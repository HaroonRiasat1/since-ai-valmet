from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pycocotools.mask as mask_util
import torch
import torchvision.ops
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def pick_device(pref: str) -> str:
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return "cuda"
    if pref in ("auto", "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_span(ids: list[int], sub: list[int], start: int) -> list[int]:
    for s in range(start, len(ids) - len(sub) + 1):
        if ids[s : s + len(sub)] == sub:
            return list(range(s, s + len(sub)))
    raise ValueError(f"prompt tokens {sub} not found in {ids}")


class GroundedSAM2:
    """Grounding DINO (Hugging Face) boxes -> SAM 2.1 masks, following Grounded-SAM-2's HF demo."""

    def __init__(self, cfg: dict[str, Any], repo_root: Path) -> None:
        dc = cfg["detection"]
        self.cfg = cfg
        self.classes = cfg["classes"]
        self.device = pick_device(dc.get("device", "auto"))
        self.processor = AutoProcessor.from_pretrained(dc["grounding_model"])
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(dc["grounding_model"]).to(self.device).eval()
        ckpt = (repo_root / dc["sam2_checkpoint"]).resolve()
        self.sam = SAM2ImagePredictor(build_sam2(dc["sam2_config"], str(ckpt), device=self.device))
        # Long prompts dilute Grounding DINO's scores, so classes are detected in groups, one pass per group.
        per_pass = int(dc.get("prompts_per_pass", len(self.classes)))
        self.groups = [list(range(i, min(i + per_pass, len(self.classes)))) for i in range(0, len(self.classes), per_pass)]
        self.texts = [" ".join(f'{self.classes[k]["prompt"].strip().lower()}.' for k in g) for g in self.groups]
        self.text = " | ".join(self.texts)
        # Keep weak candidates here; per-class box_threshold is applied when lifting.
        cand = float(dc["candidate_threshold"])
        self.thresholds = torch.tensor(
            [min(cand, float(c.get("box_threshold", dc["default_box_threshold"]))) for c in self.classes]
        )
        self.spans: list[list[list[int]] | None] = [None] * len(self.groups)

    def _class_spans(self, input_ids: torch.Tensor, group: list[int]) -> list[list[int]]:
        ids = input_ids[0].tolist()
        spans, pos = [], 0
        for k in group:
            sub = self.processor.tokenizer(self.classes[k]["prompt"].strip().lower(), add_special_tokens=False)["input_ids"]
            span = _find_span(ids, sub, pos)
            spans.append(span)
            pos = span[-1] + 1
        return spans

    @torch.inference_mode()
    def detect(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        dc = self.cfg["detection"]
        h, w = image_rgb.shape[:2]
        image = Image.fromarray(image_rgb)
        found = []
        for gi, (group, text) in enumerate(zip(self.groups, self.texts)):
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            out = self.gdino(**inputs)
            if self.spans[gi] is None:
                self.spans[gi] = self._class_spans(inputs.input_ids, group)
            probs = out.logits[0].sigmoid().float().cpu()
            cls_scores = torch.stack([probs[:, s].max(dim=1).values for s in self.spans[gi]], dim=1)
            score, local = cls_scores.max(dim=1)
            cls = torch.tensor(group)[local]
            keep = score >= self.thresholds[cls]
            found.append((out.pred_boxes[0].float().cpu()[keep], score[keep], cls[keep]))
        cxcywh = torch.cat([f[0] for f in found])
        score = torch.cat([f[1] for f in found])
        cls = torch.cat([f[2] for f in found])
        if len(score) == 0:
            return []
        boxes = torch.stack(
            [
                (cxcywh[:, 0] - cxcywh[:, 2] / 2) * w,
                (cxcywh[:, 1] - cxcywh[:, 3] / 2) * h,
                (cxcywh[:, 0] + cxcywh[:, 2] / 2) * w,
                (cxcywh[:, 1] + cxcywh[:, 3] / 2) * h,
            ],
            dim=1,
        )
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

        area_frac = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / float(w * h)
        ok = (area_frac >= float(dc["min_box_frac"])) & (area_frac <= float(dc["max_box_frac"]))
        boxes, score, cls = boxes[ok], score[ok], cls[ok]
        if len(boxes) == 0:
            return []

        keep_idx = torchvision.ops.batched_nms(boxes, score, cls, float(dc["nms_iou"]))
        boxes, score, cls = boxes[keep_idx], score[keep_idx], cls[keep_idx]

        # Same box claimed by two classes: keep the more confident label.
        iou = torchvision.ops.box_iou(boxes, boxes)
        drop = torch.zeros(len(boxes), dtype=torch.bool)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if cls[i] != cls[j] and iou[i, j] >= float(dc["cross_class_iou"]):
                    drop[j if score[j] < score[i] else i] = True
        boxes, score, cls = boxes[~drop], score[~drop], cls[~drop]

        self.sam.set_image(image_rgb)
        masks, sam_scores, _ = self.sam.predict(
            point_coords=None, point_labels=None, box=boxes.numpy(), multimask_output=False
        )
        n = len(boxes)
        masks = np.asarray(masks).reshape(n, h, w) > 0.5
        sam_scores = np.asarray(sam_scores, dtype=np.float32).reshape(n, -1)[:, 0]

        dets: list[dict[str, Any]] = []
        for i in range(n):
            m = masks[i]
            px = int(m.sum())
            if sam_scores[i] < float(dc["sam_min_score"]) or px < int(dc["min_mask_px"]):
                continue
            c = self.classes[int(cls[i])]
            dets.append(
                {
                    "class_id": int(cls[i]),
                    "class_key": c["key"],
                    "label": c["display"],
                    "score": float(score[i]),
                    "sam_score": float(sam_scores[i]),
                    "box_xyxy": [float(v) for v in boxes[i].tolist()],
                    "mask": m,
                    "mask_px": px,
                }
            )
        return dets


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_util.encode(np.asfortranarray(mask[:, :, None].astype(np.uint8)))[0]
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": rle["counts"].decode("utf-8")}


def rle_to_mask(seg: dict[str, Any]) -> np.ndarray:
    return mask_util.decode({"size": seg["size"], "counts": seg["counts"].encode("utf-8")}).astype(bool)


def detections_payload(image_path: str, dets: list[dict[str, Any]], w: int, h: int) -> dict[str, Any]:
    """Grounded-SAM-2 JSON dump format, plus class ids and SAM scores."""
    return {
        "image_path": image_path,
        "annotations": [
            {
                "class_name": d["label"],
                "class_key": d["class_key"],
                "class_id": d["class_id"],
                "bbox": d["box_xyxy"],
                "segmentation": mask_to_rle(d["mask"]),
                "score": d["score"],
                "sam_score": d["sam_score"],
                "mask_px": d["mask_px"],
            }
            for d in dets
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
    }

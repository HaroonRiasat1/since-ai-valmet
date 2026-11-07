"""Grounded diagnosis: the model only sees this asset's records, and its citations are checked."""
import json
import re

from backend.llm import Provider, chat

SYSTEM = """You are a field-service copilot for an industrial plant digital twin (Valmet x Since AI demo).
A technician has walked to an asset. Diagnose it using ONLY the evidence JSON provided.

Rules (from DOC-RAG-GUIDE and DOC-FIELD-SAFETY in the data pack):
- Name the asset. Cite evidence IDs inline in backticks (event IDs, work order IDs, case IDs, doc IDs, sheet names).
- State a confidence (low / medium / high) and say what lowers it (data-quality issues, the 3D mapping method and confidence in the evidence).
- Clearly separate OBSERVATION (what the data shows) from HYPOTHESIS (what might cause it).
- Recommend verification and escalation only. Never instruct autonomous physical action; remind about permits / lockout-tagout.
- If the evidence does not support an answer, say so. Do not invent values, IDs or procedures.

Format (markdown, concise, max ~220 words):
**Observation** (bullets) / **Hypothesis** (bullets) / **Next safe steps** (numbered) / **Confidence** (one line)."""

CITATION = re.compile(r"\b(?:EVT-\d+|WO-\d+|CASE-\d+|DQ-\d+|ASSET-\d+|DOC-[A-Z0-9_-]+)\b")
ALWAYS_ALLOWED = {"DOC-RAG-GUIDE", "DOC-FIELD-SAFETY", "DOC-3D-MAPPING"}


def evidence(scene, tag: str, nearby: list[str]) -> dict:
    """The records for one asset. Raw series are dropped, because the shifts summarise them."""
    record = dict(scene.copilot[tag])
    item = scene.by_tag.get(tag, {})
    record["latest_telemetry"] = item.get("telemetry")
    record["mapping"] = {"method": item.get("mapping"), "confidence": item.get("mapping_conf")}
    record.pop("trend", None)
    record["nearby_assets"] = [{k: scene.by_tag[t].get(k) for k in ("tag", "asset_class", "state", "health", "issue_code")}
                               for t in nearby if t in scene.by_tag and t != tag][:6]
    return record


def evidence_ids(record: dict) -> list[str]:
    ids = [record.get("asset_id"), record.get("doc")]
    ids += [e["event_id"] for e in record.get("events", [])]
    ids += [w["work_order_id"] for w in record.get("work_orders", [])]
    ids += [c["case_id"] for c in record.get("cases", [])]
    ids += [q["issue_id"] for q in record.get("data_quality", [])]
    return [i for i in ids if i]


def unknown_citations(answer: str, known: list[str]) -> list[str]:
    """IDs the answer cites that are not in the evidence, which means the model made them up."""
    return sorted(set(CITATION.findall(answer)) - set(known) - ALWAYS_ALLOWED)


def diagnose(scene, provider: Provider, tag: str, question: str = "", nearby: list[str] | None = None, model: str | None = None) -> dict:
    record = evidence(scene, tag, nearby or [])
    ask = f"Technician's question: {question}" if question else "The technician has just arrived. Give the diagnosis."
    user = f"Evidence JSON for {tag}:\n```json\n{json.dumps(record, default=str)}\n```\n\n{ask}"
    result = chat(provider, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], model=model)
    answer = result["message"]["content"]
    ids = evidence_ids(record)
    return {"answer": answer, "model": result["model"], "evidence_ids": ids, "unverified_citations": unknown_citations(answer, ids)}

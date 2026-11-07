# Field-Service Navigator

A demonstrator for the Valmet × Since AI 2026 Industrial Metaverse challenge. It turns a 3D Gaussian Splatting scan
into a navigable digital twin: assets are extracted and mapped to plant records, a technician walks a collision-free
route to a faulty device, and an AI assistant answers from the plant data with evidence it can show.

All plant data is the synthetic builder pack supplied for the challenge.

## What it does

- **Extraction:** 168 objects found in the scan, with 30 assets mapped to them. 23 of the 30 sit on an object that
  matches their class. The other 7 are shown as unconfirmed, never as fact.
- **Navigation:** two selectable grids (5 cm occupancy and a numbered 60×60 grid), From and To by tag or cell number,
  a floor overlay and a 2D map.
- **Assistant:** "go to the valve" routes from the current position. "What should we fix first?" queries the plant
  data and shows the table and the query behind it.
- **Diagnosis:** on arrival, a grounded diagnosis that cites record IDs and separates observation from hypothesis.
  Citations outside the evidence are flagged.
- **Pluggable AI:** choose Featherless or LM Studio and the model in the page. Any OpenAI-compatible server can be added.

## Quick start

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
cp .env.example .env         # add FEATHERLESS_API_KEY, or run LM Studio with a chat model loaded
uv run python -m backend --gpu
```

`--gpu` opens the app in its own window on the NVIDIA GPU (about 124 fps, against 12 on the integrated GPU). Use it on any laptop with two GPUs. Without it, the server starts alone: open http://127.0.0.1:8000. For a local model, start the LM Studio server and load a chat model that supports tools.

With Docker: `docker build -t navigator . && docker run --env-file .env -p 8000:8000 navigator`.

## Architecture

```
Browser (web/index.html: three.js + Spark, no build step)
   |  routes, chat, diagnosis
FastAPI (backend/)
   |-- routing.py    both navigation grids
   |-- agent.py      intent routing, tools, grounding checks
   |-- semantic.py   data semantic layer and query compiler
   |-- copilot.py    grounded diagnosis
   |-- vision.py     vision-model labelling and detection
   |-- llm.py        OpenAI-compatible providers
   |
   |-- web/data/scene.json        scene, grids and asset evidence
   `-- data/semantic/             layer.yaml and plant.sqlite (read-only)
```

`pipeline/` builds those files from the scan and the builder pack. It is only needed to rebuild them.

### Two semantic layers

- **Spatial** (`data/semantic_layer.json`): what is where in the scan, from extraction.
- **Data** (`data/semantic/`): what the plant records mean. Nine entities with grain, units, synonyms, rules, proven
  joins, verified reports and a glossary. Hand-written knowledge lives in `knowledge.yaml`. `layer.yaml` is generated.

### How the assistant stays grounded

The model never writes SQL and cannot set a path.

1. The model reads every request first and chooses its tools. It can combine them, for example a telemetry query
   to find the pump with the highest vibration, then a route to it.
2. Route tools are offered only when the request asks to go somewhere, so a question cannot move the technician.
3. The model fills a query spec. Every name is checked against the layer and values are bound parameters. Unknown
   columns and values come back as errors it can correct.
4. If an answer has figures or tags that no tool returned, code replaces it with what the tools found. If nothing
   was found, it says so.
5. With no model, a failed provider, or a trip request the model did not route, rules handle navigation and the
   glossary, so the demo never depends on the model being up.

On the real scene with DeepSeek-V3.2, 12 of 12 test requests were right, including combined ones. The cost is speed: a
median of 29 s and up to 66 s, against near-instant answers from the rules.

| Tool | Purpose |
|---|---|
| `search_layer`, `describe` | Find and explain entities, columns, reports and terms |
| `run_query`, `run_report` | Run a checked spec or a verified report |
| `search_assets`, `route_to`, `inspection_round` | Choose a destination and compute the path |
| `diagnose` | Grounded diagnosis with citations, shown as written |

### Choosing a model

Eight plant questions with answers taken from the database, one run each, on Featherless:

| Model | Correct |
|---|---|
| `deepseek-ai/DeepSeek-V3.2` (default) | 8 of 8, about 30 s per question |
| `moonshotai/Kimi-K2-Instruct` | 6 of 8 |
| `Qwen/Qwen2.5-7B-Instruct` | 0 of 8 |

Times were measured with several requests running at once. Set `FEATHERLESS_MODELS` to change the list.

## Measured performance

On a laptop with an RTX 5050: 1.4 to 1.9 s to load, about 124 fps in the `--gpu` window, and only 12 fps if the browser stays on the integrated Intel GPU. To use a normal browser window instead, set it to High performance in Settings > System > Display > Graphics. Route to CV-003: 32 ms on the fine grid, 2 ms on the
cell grid. A round of 13 stops: 1.9 s and 0.2 s.

## Rebuilding the data

Needs the scan and the builder pack, which are not in the repo.

```bash
uv sync --extra pipeline
python pipeline/build_scene.py "Equipment Room.ply" builder_pack.xlsx data/semantic_layer.json web
python pipeline/build_layer.py builder_pack.xlsx
```

`SKIP_SPLAT=1` skips rewriting the splat chunks. `build_layer.py` stops with an error if a join or a dashboard figure
does not hold. Extraction itself (`extract.py`, `semantic.py`, `refine_assets.py`) needs the `extraction` extra and a GPU.

## Tests

```bash
uv run pytest -q
uv run ruff check .
```

CI runs both and checks the page script. Tests use a stub in place of any AI provider.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `FEATHERLESS_API_KEY` | none | Enables Featherless |
| `FEATHERLESS_MODELS` | see `backend/llm.py` | Cloud models offered, first is the default |
| `VLM_MODEL` | `Qwen/Qwen3-VL-32B-Instruct` | Cloud vision model |
| `LMSTUDIO_URL` | `http://127.0.0.1:1234/v1` | Any OpenAI-compatible server |
| `LMSTUDIO_MODEL` | first chat model | Local model |
| `HOST`, `PORT`, `WEB_DIR`, `SEMANTIC_DIR`, `LOG_LEVEL` | see `backend/config.py` | Server settings |

Keys stay in the server environment and never reach the browser.

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health`, `/api/grids`, `/api/assets`, `/api/providers` | Status and lists |
| POST | `/api/route`, `/api/route/round` | Compute routes |
| POST | `/api/agent/chat` | Assistant |
| POST | `/api/copilot/query` | Grounded diagnosis |
| POST | `/api/vlm/label`, `/api/vlm/detect` | Vision |

Interactive docs are at `/docs`.

## Known limits

- There is no authentication, so the server binds to localhost by default. Add a gateway before exposing it.
- 7 of 30 assets are not on a matching object, and no vibration sensor is visible in the scan.
- The model comparison is one run of eight questions. It shows a large gap, not a precise ranking.
- A correct-looking query can still answer the wrong question. The checks catch invented figures, not a wrong filter,
  which is why the page always shows the table and the query.
- LM Studio with a real chat model is untested here: only a stub server has been used for the local path.
- `pipeline/` predates the lint rules and is excluded from them.

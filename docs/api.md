# API and configuration

[← Back to README](../README.md)

Interactive docs run at `http://127.0.0.1:8000/docs` while the server is up.

## Endpoints

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Status, object and asset counts, grids |
| GET | `/api/grids` | Route grid, with the walker size in cells and the cell size |
| GET | `/api/assets`, `/api/providers` | Asset list and available AI providers |
| POST | `/api/route` | One route: from a position or cell, to a tag, object or cell |
| POST | `/api/route/round` | Inspection round over every asset that needs attention |
| POST | `/api/agent/chat` | The grounded agent |
| POST | `/api/copilot/query` | Grounded diagnosis of one asset |
| POST | `/api/vlm/label`, `/api/vlm/detect` | Vision model on rendered views |
| POST | `/api/objects/{key}/remove` | Remove an object from the live scene (frees its floor) |

Errors are explicit: `422` for a route that cannot exist (with the reason), `502` when a model provider fails, `400`
for bad input.

```bash
curl -s -X POST http://127.0.0.1:8000/api/route -H 'Content-Type: application/json' \
     -d '{"grid":"cells","start_cell":1488,"target":"CV-003"}'
```

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

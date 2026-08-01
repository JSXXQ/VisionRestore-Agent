# API

All business routes are under `/api/v1/`. OpenAPI is available at `/docs`, `/redoc`, and `/openapi.json`.

## API v2 Foundation

The `/api/v2` namespace preserves `/api/v1` compatibility and adds v2-ready structures.

Implemented endpoints:

- `GET /api/v2/health`
- `GET /api/v2/system`
- `GET /api/v2/models`
- `POST /api/v2/models/scan`
- `POST /api/v2/models/{model_id}/health-check`
- `POST /api/v2/images/upload`
- `POST /api/v2/images/analyze`
- `POST /api/v2/tasks`
- `GET /api/v2/tasks/{task_id}`
- `POST /api/v2/tasks/{task_id}/cancel`
- `GET /api/v2/tasks/{task_id}/plan`
- `GET /api/v2/tasks/{task_id}/candidates`
- `GET /api/v2/tasks/{task_id}/ranking`
- `POST /api/v2/tasks/{task_id}/select-candidate`
- `GET /api/v2/tasks/{task_id}/recommendations`
- `POST /api/v2/tasks/{task_id}/postprocess/decision`
- `GET /api/v2/tasks/{task_id}/artifacts`
- `GET /api/v2/tasks/{task_id}/report`
- `WS /api/v2/tasks/{task_id}/stream`

Several endpoints currently expose and persist structured foundation records while the real v2 executor and postprocess workers are staged in.

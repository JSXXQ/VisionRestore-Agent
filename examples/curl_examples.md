# curl examples

```bash
curl http://127.0.0.1:8000/api/v1/health
curl http://127.0.0.1:8000/api/v1/models
curl -F "file=@data/cache/test_images/low_light.png" http://127.0.0.1:8000/api/v1/images/analyze
curl -X POST http://127.0.0.1:8000/api/v1/tasks -H "Content-Type: application/json" -d "{\"image_id\":\"...\",\"mode\":\"auto\",\"priority\":\"balanced\"}"
```

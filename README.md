# Employee Attendance & Analytics API

FastAPI + MongoDB. All code is in `app/main.py`.

## Run
```bash
pip install -r requirements.txt
export MONGO_URI="mongodb://localhost:27017"   # a local .env also works; real env vars win
export MONGO_DB="attendance_db"
uvicorn app.main:app --port 8000
```
Indexes are created at startup (idempotent). `GET /health` returns 200 once MongoDB answers.

## Notes
* All endpoints in `openapi.yaml` are implemented, including `/admin/explain/{endpoint}`.
* Instants are epoch milliseconds in the API and truncated to whole seconds; BSON datetimes (UTC) in MongoDB.
* Honest status: I could not run the analytics aggregation pipelines against a real MongoDB in my build environment (I tested helpers, rules and the CRUD / punch / regularize flows). Please treat the four analytics endpoints as less tested than the rest.

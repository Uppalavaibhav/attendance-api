"""
Employee Attendance & Analytics API  (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB  (a local .env is read, real environment variables win)

Layout of this file:
  1. configuration / database / indexes
  2. time, rounding and business-rule helpers (R1 - R5, R8)
  3. request models and (de)serialisation
  4. employees, attendance (punch-in / punch-out / list / regularize)
  5. analytics (MongoDB aggregation pipelines)
  6. explain endpoint
"""
import json
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from math import floor
from typing import Annotated, Any, Literal, Optional

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, OperationFailure, PyMongoError

load_dotenv()  # does not override variables that are already set in the environment

log = logging.getLogger("attendance")

# --------------------------------------------------------------------------- #
# 1. Configuration, database, indexes
# --------------------------------------------------------------------------- #
UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# tz_aware=True: PyMongo hands back aware UTC datetimes instead of naive ones.
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    tzinfo=UTC,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

_index_lock = threading.Lock()
_indexes_ready = False


def ensure_indexes() -> None:
    """Create every index we rely on. Idempotent: same spec + name is a no-op."""
    global _indexes_ready
    with _index_lock:
        if _indexes_ready:
            return
        emp, logs = db.employees, db.attendance_logs
        # employees
        emp.create_index([("emp_code", ASCENDING)], unique=True, name="emp_code_unique")
        emp.create_index([("department", ASCENDING), ("emp_code", ASCENDING)], name="dept_emp_code")
        emp.create_index([("department", ASCENDING), ("joined_on", ASCENDING)], name="dept_joined_on")
        emp.create_index([("joined_on", ASCENDING)], name="joined_on")
        # attendance_logs: the unique (emp_code, date) index is what makes punch-in race safe
        try:
            logs.create_index(
                [("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="emp_code_date_unique"
            )
        except OperationFailure:  # pre-existing duplicates in a seeded DB: still index, just not unique
            log.exception("could not build unique (emp_code, date) index")
            logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], name="emp_code_date")
        logs.create_index([("date", DESCENDING), ("emp_code", ASCENDING)], name="date_desc_emp_code")
        logs.create_index(
            [("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)], name="status_date_emp_code"
        )
        logs.create_index([("emp_code", ASCENDING), ("punch_in", DESCENDING)], name="emp_code_punch_in")
        _indexes_ready = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ensure_indexes()
    except PyMongoError:  # Mongo not reachable yet: /health keeps retrying
        log.exception("index creation failed at startup")
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


@app.exception_handler(PyMongoError)
def mongo_error_handler(_, exc: PyMongoError):
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


# --------------------------------------------------------------------------- #
# 2. Helpers: validation errors, time, rounding, business rules
# --------------------------------------------------------------------------- #
PRESENCE = ("PRESENT", "WFH", "ON_DUTY")
ALL_STATUS = ("PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY")
GRACE_SECONDS = 10 * 60
MIN_OVERTIME_MINUTES = 30
HALF_DAY_LIMIT = 4.5
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$", re.ASCII)
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$", re.ASCII)


def bad(msg: str, loc: Optional[list] = None):
    """422 in FastAPI's default validation-error shape."""
    raise HTTPException(422, detail=[{"loc": loc or ["query"], "msg": msg, "type": "value_error"}])


def parse_date(value: str, name: str = "date") -> Date:
    if not DATE_RE.match(value):
        bad(f"{name} must be YYYY-MM-DD", ["path", name])
    try:
        return Date.fromisoformat(value)
    except ValueError:
        bad(f"{name} is not a valid calendar date", ["path", name])


def parse_month(value: str):
    """-> (first_day, last_day) of a YYYY-MM month."""
    if not MONTH_RE.match(value) or int(value[:4]) < 1:
        bad("month must be YYYY-MM", ["query", "month"])
    y, m = int(value[:4]), int(value[5:7])
    first = Date(y, m, 1)
    last = Date(y, 12, 31) if m == 12 else Date(y, m + 1, 1) - timedelta(days=1)
    return first, last


def hhmm(value: str):
    h, m = value.split(":")
    return int(h), int(m)


def as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def ms_to_dt(ms: int) -> datetime:
    """Epoch ms -> aware UTC datetime, truncated to whole seconds (R1)."""
    return datetime.fromtimestamp(ms // 1000, tz=UTC)


def dt_to_ms(dt: datetime) -> int:
    delta = as_utc(dt) - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000


def ms_or_none(dt: Optional[datetime]) -> Optional[int]:
    return None if dt is None else dt_to_ms(dt)


def now_s() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _frac(x) -> Fraction:
    return Fraction(str(x)) if isinstance(x, float) else Fraction(x)


def round_half_up(x, places: int) -> float:
    """Exact half-up rounding (R4 / R8). Works on ints, Fractions and floats (via their repr)."""
    scale = 10 ** places
    n = floor(_frac(x) * scale + Fraction(1, 2))
    return float(Fraction(n, scale))


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> Date:
    """R1: IST calendar date of the punch-in; an overnight shift's early-morning punch belongs to the day before."""
    local = as_utc(punch_in).astimezone(IST)
    d = local.date()
    if hhmm(shift_end) <= hhmm(shift_start):  # overnight shift
        eh, em = hhmm(shift_end)
        if (local.hour, local.minute, local.second) < (eh, em, 0):
            d -= timedelta(days=1)
    return d


def shift_bounds(d: Date, shift_start: str, shift_end: str):
    """Shift start / end instants (IST) for the shift that belongs to attendance date `d`."""
    sh, sm = hhmm(shift_start)
    eh, em = hhmm(shift_end)
    start = datetime(d.year, d.month, d.day, sh, sm, tzinfo=IST)
    end = datetime(d.year, d.month, d.day, eh, em, tzinfo=IST)
    if end <= start:
        end += timedelta(days=1)
    return start, end


def compute_late_minutes(punch_in: datetime, shift_start_dt: datetime) -> int:
    """R2: late only if strictly more than 10:00 after shift start; then whole minutes since shift start."""
    secs = int((punch_in - shift_start_dt).total_seconds())
    return secs // 60 if secs > GRACE_SECONDS else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600, 2 decimals half-up."""
    secs = int((punch_out - punch_in).total_seconds())
    return round_half_up(Fraction(secs, 3600), 2)


def compute_overtime(punch_out: datetime, shift_end_dt: datetime) -> int:
    """R3: whole minutes after shift end, counted only when >= 30."""
    secs = int((punch_out - shift_end_dt).total_seconds())
    minutes = secs // 60
    return minutes if minutes >= MIN_OVERTIME_MINUTES else 0


def derive(emp: dict, d: Date, status: str, punch_in: Optional[datetime], punch_out: Optional[datetime]) -> dict:
    """All stored derived fields for a record (DATA_MODEL.md, 'Stored derived values')."""
    out = {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    if status in PRESENCE and punch_in is not None:
        start, end = shift_bounds(d, emp["shift_start"], emp["shift_end"])
        out["late_minutes"] = compute_late_minutes(punch_in, start)
        if punch_out is not None:
            wh = compute_work_hours(punch_in, punch_out)
            out.update(work_hours=wh, overtime_minutes=compute_overtime(punch_out, end), half_day=wh < HALF_DAY_LIMIT)
    return out


def weekdays_between(a: Date, b: Date) -> int:
    """Number of Mon-Fri days in [a, b] (at most ~31 iterations)."""
    n, d = 0, a
    while d <= b:
        n += d.weekday() < 5
        d += timedelta(days=1)
    return n


# --------------------------------------------------------------------------- #
# 3. Models and serialisation
# --------------------------------------------------------------------------- #
# strict=True rejects floats, strings and bools; bounds reject values that look like seconds.
EpochMs = Annotated[int, Field(strict=True, ge=100_000_000_000, le=4_102_444_800_000)]
HHMM = r"^([01]\d|2[0-3]):[0-5]\d$"


class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=HHMM)
    shift_end: str = Field(default="18:30", pattern=HHMM)
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def _joined_on(cls, v: str) -> str:
        if not DATE_RE.match(v):
            raise ValueError("joined_on must be YYYY-MM-DD")
        Date.fromisoformat(v)  # ValueError -> 422
        return v

    @model_validator(mode="after")
    def _shift(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str = Field(min_length=1)
    punched_at: Optional[EpochMs] = None
    status: Literal["PRESENT", "WFH", "ON_DUTY"] = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str = Field(min_length=1)
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    status: Optional[Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @model_validator(mode="before")
    @classmethod
    def _no_explicit_null(cls, data):
        # Omitted means "unchanged"; an explicit null is not part of the contract.
        if isinstance(data, dict):
            for k in ("status", "punch_in", "punch_out"):
                if k in data and data[k] is None:
                    raise ValueError(f"{k} must not be null")
        return data


def ser_history(history) -> list:
    out = []
    for h in history or []:
        changes = {}
        for k, v in (h.get("changes") or {}).items():
            if k in ("punch_in", "punch_out"):
                changes[k] = {"from": ms_or_none(v.get("from")), "to": ms_or_none(v.get("to"))}
            else:
                changes[k] = {"from": v.get("from"), "to": v.get("to")}
        out.append({"at": ms_or_none(h.get("at")), "by": h.get("by"), "reason": h.get("reason"), "changes": changes})
    return out


def ser_record(doc: dict) -> dict:
    """Stored attendance document -> API shape. Tolerates legacy docs (missing fields). Never leaks _id."""
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": ms_or_none(doc.get("punch_in")),
        "punch_out": ms_or_none(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day") or False),
        "history": ser_history(doc.get("history")),
    }


def ser_employee(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": ms_or_none(doc.get("created_at")),
    }


def get_employee(emp_code: str) -> dict:
    emp = db.employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    return emp


def skip_for(page: int, page_size: int) -> int:
    return min((page - 1) * page_size, 2 ** 62)


# --------------------------------------------------------------------------- #
# 4. Employees and attendance
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        ensure_indexes()  # no-op once done; covers "Mongo came up after the app"
    except PyMongoError:
        raise HTTPException(503, "database unavailable")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = now_s()
    try:
        db.employees.insert_one(doc)  # the unique index decides, not a prior find
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return ser_employee(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = {"department": department} if department else {}
    total = db.employees.count_documents(q)
    cur = db.employees.find(q, {"_id": 0}).sort("emp_code", ASCENDING).skip(skip_for(page, page_size)).limit(page_size)
    return {"items": [ser_employee(d) for d in cur], "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = get_employee(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_s()
    d = attendance_date(ts, emp["shift_start"], emp["shift_end"])
    doc = {
        "emp_code": body.emp_code,
        "date": d.isoformat(),
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        **derive(emp, d, body.status, ts, None),
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)  # unique (emp_code, date): exactly one concurrent insert wins
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return ser_record(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = get_employee(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_s()
    # most recent record whose punch_in <= punched_at (works across midnight: no date match needed)
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts}}, sort=[("punch_in", DESCENDING)]
    )
    if rec is None:
        raise HTTPException(404, "no punch-in found")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")
    pin = as_utc(rec["punch_in"])
    if ts <= pin:
        bad("punched_at must be after punch_in", ["body", "punched_at"])
    if ts - pin > timedelta(hours=24):
        bad("punched_at is more than 24 hours after punch_in", ["body", "punched_at"])
    derived = derive(emp, Date.fromisoformat(rec["date"]), rec["status"], pin, ts)
    # compare-and-set: only succeeds while the record is still open and unchanged
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None, "punch_in": rec["punch_in"]},
        {"$set": {"punch_out": ts, **derived}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return ser_record(updated)


def attendance_filter(emp_code, date_from, date_to, status) -> dict:
    q: dict[str, Any] = {}
    df = parse_date(date_from, "date_from") if date_from else None
    dt = parse_date(date_to, "date_to") if date_to else None
    if df and dt and df > dt:
        bad("date_from must not be after date_to", ["query", "date_from"])
    if status is not None and status not in ALL_STATUS:
        bad("invalid status", ["query", "status"])
    if emp_code:
        q["emp_code"] = emp_code
    if df or dt:
        q["date"] = {}
        if df:
            q["date"]["$gte"] = df.isoformat()
        if dt:
            q["date"]["$lte"] = dt.isoformat()
    if status:
        q["status"] = status
    return q


ATT_SORT = [("date", DESCENDING), ("emp_code", ASCENDING)]


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = attendance_filter(emp_code, date_from, date_to, status)
    total = db.attendance_logs.count_documents(q) if q else db.attendance_logs.estimated_document_count()
    cur = db.attendance_logs.find(q, {"_id": 0}).sort(ATT_SORT).skip(skip_for(page, page_size)).limit(page_size)
    return {"items": [ser_record(d) for d in cur], "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: str, body: RegularizeIn):
    parse_date(date, "date")
    emp = get_employee(emp_code)
    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if rec is None:
        raise HTTPException(404, "no attendance record for that date")

    old_status = rec["status"]
    old_in, old_out = rec.get("punch_in"), rec.get("punch_out")
    old_history = rec.get("history") or []
    new_status = body.status or old_status

    if new_status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            bad(f"{new_status} clears punch times; do not send punch_in / punch_out", ["body"])
        new_in = new_out = None
    else:
        new_in = ms_to_dt(body.punch_in) if body.punch_in is not None else old_in
        new_out = ms_to_dt(body.punch_out) if body.punch_out is not None else old_out
        if new_in is None:
            bad("a presence status requires a punch_in", ["body", "punch_in"])
        new_in = as_utc(new_in)
        new_out = as_utc(new_out) if new_out is not None else None
        if attendance_date(new_in, emp["shift_start"], emp["shift_end"]).isoformat() != date:
            bad("punch_in must stay on the record's attendance date", ["body", "punch_in"])
        if new_out is not None:
            if new_out <= new_in:
                bad("punch_out must be after punch_in", ["body", "punch_out"])
            if new_out - new_in > timedelta(hours=24):
                bad("punch_out is more than 24 hours after punch_in", ["body", "punch_out"])

    new_vals = {"status": new_status, "punch_in": new_in, "punch_out": new_out,
                **derive(emp, Date.fromisoformat(date), new_status, new_in, new_out)}
    old_vals = {
        "status": old_status, "punch_in": old_in, "punch_out": old_out,
        "work_hours": rec.get("work_hours"), "late_minutes": rec.get("late_minutes") or 0,
        "overtime_minutes": rec.get("overtime_minutes") or 0, "half_day": bool(rec.get("half_day") or False),
    }

    def same(a, b) -> bool:
        if isinstance(a, datetime) or isinstance(b, datetime):
            return a is not None and b is not None and as_utc(a) == as_utc(b)
        return a == b

    changes = {k: {"from": old_vals[k], "to": new_vals[k]} for k in old_vals if not same(old_vals[k], new_vals[k])}
    if not changes:
        bad("request changes nothing", ["body"])

    entry = {"at": datetime.now(UTC), "by": body.regularized_by, "reason": body.reason, "changes": changes}
    # optimistic concurrency: apply only if the record is exactly as we read it (same times, status, history length)
    flt = {
        "emp_code": emp_code, "date": date, "status": old_status, "punch_in": old_in, "punch_out": old_out,
        "$expr": {"$eq": [{"$size": {"$ifNull": ["$history", []]}}, len(old_history)]},
    }
    updated = db.attendance_logs.find_one_and_update(
        flt, {"$set": new_vals, "$push": {"history": entry}}, return_document=ReturnDocument.AFTER
    )
    if updated is None:
        raise HTTPException(409, "record was modified concurrently; retry")
    return ser_record(updated)


# --------------------------------------------------------------------------- #
# 5. Analytics (aggregation pipelines)
# --------------------------------------------------------------------------- #
# Reusable aggregation expressions. They read the STORED derived values, never recompute from punch times.
IS_PRESENT = {"$in": ["$status", list(PRESENCE)]}
IS_WEEKDAY = {"$in": [{"$dayOfWeek": {"$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}}}, [2, 3, 4, 5, 6]]}
IS_HALF = {"$eq": [{"$ifNull": ["$half_day", False]}, True]}
LATE_MIN = {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, "$late_minutes", 0]}
IS_LATE = {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}
HAS_HOURS = {"$and": [IS_PRESENT, {"$ne": [{"$ifNull": ["$work_hours", None]}, None]}]}
# present "half-days": full day = 2, half day = 1 (so everything stays an integer until the last step)
PRESENT_HALVES_ANY_DAY = {"$cond": [IS_PRESENT, {"$cond": [IS_HALF, 1, 2]}, 0]}
PRESENT_HALVES_WEEKDAY = {"$cond": [{"$and": [IS_PRESENT, IS_WEEKDAY]}, {"$cond": [IS_HALF, 1, 2]}, 0]}
HOURS_HUNDREDTHS = {"$cond": [HAS_HOURS, {"$round": [{"$multiply": ["$work_hours", 100]}, 0]}, 0]}


def monthly_pipeline(emp_code: str, first: Date, last: Date) -> list:
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},
        {"$group": {
            "_id": None,
            "halves": {"$sum": PRESENT_HALVES_WEEKDAY},
            "leave": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
            "late_count": {"$sum": IS_LATE},
            "late_min": {"$sum": LATE_MIN},
            "ot_min": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
        }},
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$")):
    first, last = parse_month(month)
    emp = get_employee(emp_code)
    res = next(iter(db.attendance_logs.aggregate(monthly_pipeline(emp_code, first, last))), None) or {}
    working = weekdays_between(max(first, Date.fromisoformat(emp["joined_on"])), last)
    halves = int(res.get("halves", 0))
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working,
        "present_days": halves / 2,
        "leave_days": int(res.get("leave", 0)),
        "late_count": int(res.get("late_count", 0)),
        "total_late_minutes": int(res.get("late_min", 0)),
        "total_overtime_minutes": int(res.get("ot_min", 0)),
        "attendance_pct": round_half_up(Fraction(halves, 2 * working) * 100, 2) if working else None,
    }


def summary_pipeline(first: Date, last: Date, department: Optional[str]) -> list:
    emp_match: dict[str, Any] = {"joined_on": {"$lte": last.isoformat()}}
    if department:
        emp_match["department"] = department
    sums = ("halves", "hund", "hn", "late_count", "late_min", "leave", "duty")
    return [
        # 1. employees who count (R9) - this side guarantees people with zero logs are present
        {"$match": emp_match},
        {"$project": {"_id": "$emp_code", "department": 1, "is_emp": {"$literal": True}}},
        # 2. add each employee's month of logs as one pre-summed row per emp_code
        {"$unionWith": {"coll": "attendance_logs", "pipeline": [
            {"$match": {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},
            {"$group": {
                "_id": "$emp_code",
                "halves": {"$sum": PRESENT_HALVES_WEEKDAY},
                "hund": {"$sum": HOURS_HUNDREDTHS},
                "hn": {"$sum": {"$cond": [HAS_HOURS, 1, 0]}},
                "late_count": {"$sum": IS_LATE},
                "late_min": {"$sum": LATE_MIN},
                "leave": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                "duty": {"$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}},
            }},
        ]}},
        # 3. merge the two row kinds per employee; rows without is_emp (unknown / not-yet-joined / other dept) drop out
        {"$group": {"_id": "$_id", "department": {"$max": "$department"}, "is_emp": {"$max": "$is_emp"},
                    **{s: {"$sum": f"${s}"} for s in sums}}},
        {"$match": {"is_emp": True}},
        # 4. per department
        {"$group": {"_id": "$department", "headcount": {"$sum": 1}, **{s: {"$sum": f"${s}"} for s in sums}}},
        {"$sort": {"_id": 1}},
    ]


@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"), department: Optional[str] = None
):
    first, last = parse_month(month)
    items = []
    for r in db.employees.aggregate(summary_pipeline(first, last, department)):
        items.append({
            "department": r["_id"],
            "headcount": int(r["headcount"]),
            "present_days": int(r["halves"]) / 2,
            "avg_work_hours": round_half_up(Fraction(int(r["hund"]), 100 * int(r["hn"])), 2) if r["hn"] else None,
            "late_count": int(r["late_count"]),
            "total_late_minutes": int(r["late_min"]),
            "leave_count": int(r["leave"]),
            "on_duty_count": int(r["duty"]),
        })
    return {"month": month, "items": items}


def leaderboard_pipeline(first: Date, last: Date, limit: int, department: Optional[str]) -> list:
    pipe: list[dict[str, Any]] = [
        {"$match": {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},
        {"$group": {"_id": "$emp_code", "total": {"$sum": LATE_MIN}, "late_count": {"$sum": IS_LATE}}},
        {"$match": {"total": {"$gt": 0}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "e"}},
        {"$unwind": "$e"},  # logs of unknown employees disappear here
    ]
    if department:
        pipe.append({"$match": {"e.department": department}})  # rank inside the department only
    pipe += [
        # standard competition ranking (1, 2, 2, 4) computed by the database, applied BEFORE the limit
        {"$setWindowFields": {"sortBy": {"total": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total": -1, "_id": 1}},
        {"$project": {"_id": 0, "rank": 1, "emp_code": "$_id", "name": "$e.name", "department": "$e.department",
                      "total_late_minutes": "$total", "late_count": 1}},
    ]
    return pipe


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    first, last = parse_month(month)
    items = list(db.attendance_logs.aggregate(leaderboard_pipeline(first, last, limit, department)))
    return {"month": month, "items": items}


def trend_range(frm: str, to: str):
    f, t = parse_date(frm, "from"), parse_date(to, "to")
    if t < f:
        bad("to must not be before from", ["query", "to"])
    if (t - f).days + 1 > 92:
        bad("range must not exceed 92 days", ["query", "to"])
    return f, t


def trend_pipeline(department: str, f: Date, t: Date) -> list:
    fs, ts = f.isoformat(), t.isoformat()
    lower = datetime(f.year, f.month, f.day, tzinfo=UTC)
    upper = datetime(t.year, t.month, t.day, tzinfo=UTC) + timedelta(days=1)  # densify upper bound is exclusive
    return [
        {"$match": {"department": department}},
        # each employee brings its logs in the range (index on (emp_code, date))
        {"$lookup": {"from": "attendance_logs", "localField": "emp_code", "foreignField": "emp_code",
                     "pipeline": [
                         {"$match": {"date": {"$gte": fs, "$lte": ts}}},
                         {"$project": {"_id": 0, "d": "$date", "pm": PRESENT_HALVES_ANY_DAY, "late": IS_LATE}},
                     ], "as": "logs"}},
        # turn every employee into "events" keyed by day: a seed row, a join row (+1 headcount) and one row per log
        {"$project": {"events": {"$concatArrays": [
            {"$literal": [{"d": fs, "hc": 0, "pm": 0, "late": 0}]},
            {"$cond": [{"$lte": ["$joined_on", ts]},
                       [{"d": {"$cond": [{"$lt": ["$joined_on", fs]}, fs, "$joined_on"]}, "hc": 1, "pm": 0, "late": 0}],
                       []]},
            {"$map": {"input": "$logs", "as": "l",
                      "in": {"d": "$$l.d", "hc": 0, "pm": "$$l.pm", "late": "$$l.late"}}},
        ]}}},
        {"$unwind": "$events"},
        {"$replaceRoot": {"newRoot": "$events"}},
        {"$group": {"_id": "$d", "hc": {"$sum": "$hc"}, "pm": {"$sum": "$pm"}, "late": {"$sum": "$late"}}},
        {"$addFields": {"dt": {"$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d"}}}},
        # gap filling: one document per calendar day in [from, to]
        {"$densify": {"field": "dt", "range": {"step": 1, "unit": "day", "bounds": [lower, upper]}}},
        {"$addFields": {"hc": {"$ifNull": ["$hc", 0]}, "pm": {"$ifNull": ["$pm", 0]}, "late": {"$ifNull": ["$late", 0]}}},
        # running headcount (R9): everyone who joined on or before that day
        {"$setWindowFields": {"sortBy": {"dt": 1},
                              "output": {"headcount": {"$sum": "$hc", "window": {"documents": ["unbounded", "current"]}}}}},
        {"$addFields": {"wd": {"$in": [{"$dayOfWeek": "$dt"}, [2, 3, 4, 5, 6]]}}},
        {"$addFields": {"rate": {"$cond": [{"$and": ["$wd", {"$gt": ["$headcount", 0]}]},
                                           {"$divide": ["$pm", {"$multiply": [2, "$headcount"]}]}, None]}}},
        # 7-row moving average inside the range; $avg skips null rates and yields null when none are left
        {"$setWindowFields": {"sortBy": {"dt": 1},
                              "output": {"ma": {"$avg": "$rate", "window": {"documents": [-6, 0]}}}}},
        {"$sort": {"dt": 1}},
        {"$project": {"_id": 0, "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$dt"}},
                      "wd": 1, "headcount": 1, "pm": 1, "late": 1, "ma": 1}},
    ]


@app.get("/analytics/departments/{department}/trend")
def department_trend(department: str, from_: str = Query(..., alias="from"), to: str = Query(...)):
    f, t = trend_range(from_, to)
    if db.employees.find_one({"department": department}, {"_id": 1}) is None:
        raise HTTPException(404, "unknown department")
    items = []
    for r in db.employees.aggregate(trend_pipeline(department, f, t)):
        hc, pm = int(r["headcount"]), int(r["pm"])
        items.append({
            "date": r["date"],
            "is_working_day": r["wd"],
            "headcount": hc,
            "present_count": pm / 2,
            "late_count": int(r["late"]),
            "attendance_rate": round_half_up(Fraction(pm, 2 * hc), 4) if r["wd"] and hc > 0 else None,
            "moving_avg_7d": round_half_up(r["ma"], 4) if r.get("ma") is not None else None,
        })
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# 6. Explain endpoint
# --------------------------------------------------------------------------- #
def _need(**params):
    missing = [k for k, v in params.items() if v is None]
    if missing:
        raise HTTPException(422, detail=[{"loc": ["query", k], "msg": "field required", "type": "missing"} for k in missing])


@app.get("/admin/explain/{endpoint}")
def explain(
    endpoint: Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"],
    emp_code: Optional[str] = None,
    month: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    # Each branch builds the operation with the very same builder the real endpoint uses.
    if endpoint == "attendance_list":
        q = attendance_filter(emp_code, date_from, date_to, status)
        coll = "attendance_logs"
        cmd: dict[str, Any] = {"find": coll, "filter": q, "sort": dict(ATT_SORT),
                               "skip": skip_for(page, page_size), "limit": page_size}
    else:
        if endpoint == "employee_monthly":
            _need(emp_code=emp_code, month=month)
            first, last = parse_month(month)
            coll, pipe = "attendance_logs", monthly_pipeline(emp_code, first, last)
        elif endpoint == "department_summary":
            _need(month=month)
            first, last = parse_month(month)
            coll, pipe = "employees", summary_pipeline(first, last, department)
        elif endpoint == "late_leaderboard":
            _need(month=month)
            first, last = parse_month(month)
            coll, pipe = "attendance_logs", leaderboard_pipeline(first, last, limit, department)
        else:  # department_trend
            _need(department=department, **{"from": from_, "to": to})
            f, t = trend_range(from_, to)
            coll, pipe = "employees", trend_pipeline(department, f, t)
        cmd = {"aggregate": coll, "pipeline": pipe, "cursor": {}}
    result = db.command({"explain": cmd, "verbosity": "executionStats"})
    return JSONResponse({
        "endpoint": endpoint,
        "collection": coll,
        "explain": json.loads(json_util.dumps(result, json_options=json_util.RELAXED_JSON_OPTIONS)),
    })

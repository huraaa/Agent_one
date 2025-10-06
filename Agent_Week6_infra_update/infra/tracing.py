# infra/tracing.py
import time, uuid, json, sys
from contextlib import contextmanager

def new_request_id() -> str:
    return uuid.uuid4().hex[:12]

def log(event: str, **fields):
    rec = {"event": event, **fields}
    sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")

@contextmanager
def span(event: str, **fields):
    t0 = time.time()
    rid = fields.get("request_id")
    log(event + ".start", **fields)
    try:
        yield
        dt = round(time.time() - t0, 3)
        log(event + ".end", duration_s=dt, **({k:v for k,v in fields.items() if k!="request_id"}), request_id=rid)
    except Exception as e:
        dt = round(time.time() - t0, 3)
        log(event + ".error", duration_s=dt, error=str(e), **fields)
        raise

# infra/cache.py
import os, sqlite3, json, hashlib, time
DB = os.getenv("CACHE_DB","cache.db")

def _conn(): return sqlite3.connect(DB)

def init():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS cache(
            k TEXT PRIMARY KEY, v TEXT, ts REAL)""")

def make_key(model: str, prompt: str, profile: dict) -> str:
    payload = json.dumps({"m":model, "p":prompt, "profile":profile}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def get(k: str):
    with _conn() as c:
        row = c.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
    return json.loads(row[0]) if row else None

def set_(k: str, v_obj):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO cache(k,v,ts) VALUES(?,?,?)",
                  (k, json.dumps(v_obj, ensure_ascii=False), time.time()))

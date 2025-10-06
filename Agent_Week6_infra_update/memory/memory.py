import os, sqlite3
from typing import Dict, List, Tuple

DB_PATH = os.getenv("MEMORY_DB", "memory.db")

def _conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS profiles(
            user_id TEXT, k TEXT, v TEXT, PRIMARY KEY(user_id,k))""")
        c.execute("""CREATE TABLE IF NOT EXISTS facts(
            user_id TEXT, fact TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")

def set_profile_kv(user_id: str, k: str, v: str):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO profiles(user_id,k,v) VALUES(?,?,?)",(user_id,k,str(v)))

def get_profile_dict(user_id: str) -> Dict[str,str]:
    with _conn() as c:
        rows = c.execute("SELECT k,v FROM profiles WHERE user_id=?",(user_id,)).fetchall()
    return {k:v for k,v in rows}

def add_fact(user_id: str, fact: str):
    with _conn() as c:
        c.execute("INSERT INTO facts(user_id,fact) VALUES(?,?)",(user_id,fact))

def get_recent_facts(user_id: str, n: int=5) -> List[str]:
    with _conn() as c:
        rows = c.execute("SELECT fact FROM facts WHERE user_id=? ORDER BY ts DESC LIMIT ?",(user_id,n)).fetchall()
    return [r[0] for r in rows]

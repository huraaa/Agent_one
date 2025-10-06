# agent.py — Week 6: planner agent with memory, tracing, cache, retries, and confidence gating.

import os, json
from typing import Any, Dict, List, Optional

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL", "gpt-4.1")
APP_VERSION = os.getenv("APP_VERSION", "w6.0")
USER_ID = os.getenv("USER_ID", "default")

# --- memory ---
from memory.memory import init_db, get_profile_dict, get_recent_facts, set_profile_kv, add_fact
init_db()

# --- tracing, cache, retries ---
from infra.tracing import new_request_id, log, span
from infra.cache import init as cache_init, make_key, get as cache_get, set_ as cache_set
from infra.retry import retry
cache_init()

# --- tools: core ---
from tools.calculator import calculator as _calc
from tools.retriever import query_topk as _query_topk
try:
    # confident(results) must exist in your tools/retriever.py (returns bool on top hit)
    from tools.retriever import confident as _retrieval_confident
except Exception:
    def _retrieval_confident(_): return True

try:
    from tools.websearch import web_search as _web_search
    _has_web = True
except Exception:
    _has_web = False

# optional: sentiment tool (if you added it)
try:
    from tools.sentiment import sentiment as _sent
    _has_sent = True
except Exception:
    _has_sent = False

# --- tool schema ---
TOOL_SPEC: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate arithmetic expressions.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_docs",
            "description": "Retrieve top-k relevant chunks from local PDFs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_profile",
            "description": "Read user profile key-values.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_preference",
            "description": "Persist a user preference (allowed: name, citation_style, default_k).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Store a short factual note about the user for future sessions.",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        },
    },
]

if _has_web:
    TOOL_SPEC.append({
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return top results with titles and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    })

if _has_sent:
    TOOL_SPEC.append({
        "type": "function",
        "function": {
            "name": "sentiment",
            "description": "Classify sentiment of short text (movie-review tuned).",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    })


# --- tool runner ---
def run_local_tool(name: str, args_json: str) -> Dict[str, Any]:
    args = json.loads(args_json) if isinstance(args_json, str) else args_json

    if name == "calculator":
        return _calc(**args)

    if name == "retrieve_docs":
        items = _query_topk(args["query"], args.get("k", 3))
        top_score = items[0].get("score") if items else None
        return {"results": items, "confident": _retrieval_confident(items), "top_score": top_score}

    if name == "web_search" and _has_web:
        return _web_search(**args)

    if name == "sentiment" and _has_sent:
        return _sent(**args)

    if name == "read_profile":
        return {"profile": get_profile_dict(USER_ID)}

    if name == "save_preference":
        key = args["key"].strip().lower()
        if key not in {"name", "citation_style", "default_k"}:
            return {"error": "key not allowed"}
        set_profile_kv(USER_ID, key, args["value"])
        return {"ok": True}

    if name == "remember_fact":
        add_fact(USER_ID, args["fact"])
        return {"ok": True}

    return {"error": f"Unknown tool {name}"}


# --- llm call with retries ---
def _llm_call(messages: List[Dict[str, Any]]):
    def _do():
        return client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SPEC,
            tool_choice="auto",
            timeout=30,
        )
    return retry(_do, tries=3)


# --- main entry ---
def run_agent(user_goal: str, max_rounds: int = 6) -> str:
    request_id = new_request_id()
    profile = get_profile_dict(USER_ID)
    facts = get_recent_facts(USER_ID, n=5)

    sys_content = (
        f"[version:{APP_VERSION}][request_id:{request_id}]\n"
        "Planner mode. Decide steps and call tools as needed.\n"
        "- Use retrieve_docs for local PDFs.\n"
        "- Use web_search for internet research.\n"
        "- Use calculator for arithmetic.\n"
        "Cite sources (local=paths, web=URLs). If insufficient info, say so.\n"
        "If retrieve_docs returns confident=false, answer: 'Not found in the provided documents.'\n"
        f"User profile: {profile}\n"
        f"Known user facts: {facts}\n"
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_goal},
    ]

    # cache
    cache_key = make_key(MODEL, user_goal, profile)
    cached = cache_get(cache_key)
    if cached:
        log("cache.hit", request_id=request_id)
        return cached["answer"]

    with span("agent.run", request_id=request_id, user_goal=user_goal, model=MODEL):
        for _ in range(max_rounds):
            with span("llm.call", request_id=request_id):
                resp = _llm_call(messages)
            msg = resp.choices[0].message

            if getattr(msg, "tool_calls", None):
                messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
                log("agent.tool_calls", request_id=request_id,
                    calls=[(tc.function.name, tc.function.arguments) for tc in msg.tool_calls])
                for tc in msg.tool_calls:
                    with span("tool.exec", request_id=request_id, tool=tc.function.name):
                        result = run_local_tool(tc.function.name, tc.function.arguments)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
                continue

            messages.append({"role": "assistant", "content": msg.content})
            ans = (msg.content or "").strip()
            cache_set(cache_key, {"answer": ans})
            log("cache.store", request_id=request_id)
            return ans

    return "Stopped without final answer."


# --- optional safety wrapper (uses Week-4 guard if present) ---
try:
    from safety.filter import guard_query
    def run_agent_safe(user_goal: str, max_rounds: int = 6) -> str:
        g = guard_query(user_goal)
        if g.get("blocked"):
            return f"Refused: {g.get('reason','blocked')}."
        return run_agent(user_goal, max_rounds=max_rounds)
except Exception:
    # fallback if safety not installed
    def run_agent_safe(user_goal: str, max_rounds: int = 6) -> str:
        return run_agent(user_goal, max_rounds=max_rounds)

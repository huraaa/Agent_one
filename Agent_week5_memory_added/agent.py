import os, json
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from memory.memory import init_db, get_profile_dict, get_recent_facts, set_profile_kv, add_fact
init_db()
USER_ID = os.getenv("USER_ID","default")


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL", "gpt-4.1")

# ---- Tools spec
TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate arithmetic expressions.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_docs",
            "description": "Retrieve top-k relevant document chunks for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return top results with titles and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"sentiment",
            "description":"Classify sentiment of short text (movie-review tuned).",
            "parameters":{  
            "type":"object",
            "properties":{
                "text":{"type":"string"}
            },
            "required":["text"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"read_profile",
            "description":"Read user profile key-values.",
            "parameters":{"type":"object","properties":{}}
        }
    },
    {
        "type":"function",
        "function":{
            "name":"save_preference",
            "description":"Persist a user preference (allowed: name, citation_style, default_k).",
            "parameters":{
                "type":"object",
                "properties":{
                    "key":{"type":"string"},
                    "value":{"type":"string"}
                },
                "required":["key","value"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"remember_fact",
            "description":"Store a short factual note about the user for future sessions.",
            "parameters":{
                "type":"object",
                "properties":{"fact":{"type":"string"}},
                "required":["fact"]
            }
        }
    }
]

# ---- Local tool implementations
from tools.calculator import calculator as _calc
from tools.retriever import query_topk as _query_topk
from tools.websearch import web_search as _web_search
from safety.filter import guard_query
from tools.sentiment import sentiment as _sent






def run_agent_safe(user_goal: str, max_rounds: int = 6) -> str:
    g = guard_query(user_goal)
    if g["blocked"]:
        return f"Refused: {g['reason']}. Your request violates safety checks."
    return run_agent(user_goal, max_rounds=max_rounds)


def run_local_tool(name: str, args_json: str) -> Dict[str, Any]:
    try:
        args = json.loads(args_json) if isinstance(args_json, str) else args_json
    except Exception:
        return {"error": "Bad JSON args"}
    if name == "calculator":
        return _calc(**args)
    if name == "web_search":
        return _web_search(**args)
    if name == "retrieve_docs":
        items = _query_topk(args["query"], args.get("k", 3))
        return {"results": items}
    if name == "sentiment":
        return _sent(**args)
    if name == "read_profile":
        return {"profile": get_profile_dict(USER_ID)}
    if name == "save_preference":
        key = args["key"].strip().lower()
        if key not in {"name","citation_style","default_k"}:
            return {"error":"key not allowed"}
        set_profile_kv(USER_ID, key, args["value"])
        return {"ok": True}
    if name == "remember_fact":
        add_fact(USER_ID, args["fact"])
        return {"ok": True}
    return {"error": f"Unknown tool {name}"}


def call_model(messages: List[Dict[str, Any]]):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SPEC,
        tool_choice="auto"
    )

def run_agent(user_goal: str, max_rounds: int = 6) -> str:
    profile = get_profile_dict(USER_ID)
    facts = get_recent_facts(USER_ID, n=5)

    sys_content = (
        "Planner mode. Decide steps and call tools as needed.\n"
        "- Use retrieve_docs for local PDFs.\n"
        "- Use web_search for internet research.\n"
        "- Use calculator for arithmetic.\n"
        "After tools, produce a concise answer with citations:\n"
        " - cite local as file paths\n"
        " - cite web as URLs\n"
        "If insufficient info, say so.\n"
        f"User profile: {profile}\n"
        f"Known user facts: {facts}\n"
    )
    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_goal},
    ]


    for _ in range(max_rounds):
        resp = call_model(messages)
        msg = resp.choices[0].message

        if getattr(msg, "tool_calls", None):
            messages.append({"role":"assistant","content": msg.content or "", "tool_calls": msg.tool_calls})
            print("TOOL CALLS:", [ (tc.function.name, tc.function.arguments) for tc in msg.tool_calls ])
            for tc in msg.tool_calls:
                result = run_local_tool(tc.function.name, tc.function.arguments)
                messages.append({"role":"tool","tool_call_id": tc.id,"content": json.dumps(result)})
            continue

        messages.append({"role":"assistant","content": msg.content})
        return msg.content.strip()

    return "Stopped without final answer."

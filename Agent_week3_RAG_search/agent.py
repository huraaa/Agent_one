import os, json
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv


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
    }

]

# ---- Local tool implementations
from tools.calculator import calculator as _calc
from tools.retriever import query_topk as _query_topk
from tools.websearch import web_search as _web_search

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
    return {"error": f"Unknown tool {name}"}

def call_model(messages: List[Dict[str, Any]]):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SPEC,
        tool_choice="auto"
    )

def run_agent(user_goal: str, max_rounds: int = 6) -> str:
        messages = [
        {"role": "system", "content":
         ("Planner mode. Decide steps and call tools as needed.\n"
          "- Use retrieve_docs for local PDFs.\n"
          "- Use web_search for internet research.\n"
          "- Use calculator for arithmetic.\n"
          "After tools, produce a concise answer with citations:\n"
          " - cite local as file paths\n"
          " - cite web as URLs\n"
          "If insufficient info, say so.")},
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

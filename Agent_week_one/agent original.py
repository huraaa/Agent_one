import json, os, time
from typing import Any, Dict, List
from dotenv import load_dotenv
from openai import OpenAI
from tools import search_web, fetch_url, write_file, read_file

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL", "gpt-5")

TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Web search for links given a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and extract readable text from a URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 8000}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a local file path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path","content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read text content from a local file path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 8000}
                },
                "required": ["path"]
            }
        }
    }
]

TOOLS_IMPL = {
    "search_web": search_web,
    "fetch_url": fetch_url,
    "write_file": write_file,
    "read_file": read_file
}

def call_model(messages: List[Dict[str, Any]], tools=TOOL_SPEC):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=1
    )

def save_memory(entry: Dict[str, Any], path="memory.json"):
    try:
        data = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        data.append(entry)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Memory write failed:", e)

def run_agent(user_goal: str, max_iters: int = 8, max_tokens_seen: int = 120_000):
    messages = [
        {"role": "system", "content":
         "You are a focused task agent. Plan briefly. Use tools when useful. "
         "Cite URLs in your final answer. Stop when the goal is satisfied."},
        {"role": "user", "content": user_goal}
    ]
    tokens_seen = 0

    for step in range(1, max_iters+1):
        resp = call_model(messages)
        choice = resp.choices[0].message
        tokens_seen += resp.usage.total_tokens if resp.usage else 0

        # Tool call?
        if choice.tool_calls:
            for tool_call in choice.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                if name not in TOOLS_IMPL:
                    messages.append({"role": "assistant", "content": f"Tool {name} not available."})
                    continue
                try:
                    result = TOOLS_IMPL[name](**args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": json.dumps(result) if not isinstance(result, str) else result
                    })
                    save_memory({"step": step, "tool": name, "args": args, "result_preview": str(result)[:300]})
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": f"ERROR: {e}"
                    })
        else:
            # Final or intermediate response
            text = choice.content or ""
            print(f"\n[STEP {step}] {text}\n")
            if "DONE" in text.upper() or step == max_iters:
                save_memory({"final": text, "tokens_seen": tokens_seen})
                return text

        if tokens_seen > max_tokens_seen:
            messages.append({"role": "assistant", "content": "Token budget exceeded. Summarize and finish."})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("goal", type=str, help="What should the agent accomplish?")
    parser.add_argument("--iters", type=int, default=8)
    args = parser.parse_args()
    print(run_agent(args.goal, max_iters=args.iters))

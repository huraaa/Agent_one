# agent.py
import json
import os
from typing import List, Dict, Any
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL", "gpt-4.1")  # pick a model that supports tools

# 1) Declare tools (JSON schema)
TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a simple arithmetic expression safely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g., '17*24+5/2'"
                    }
                },
                "required": ["expression"],
                "additionalProperties": False
            }
        }
    }
]

# 2) Implement tool(s)
def calculator(expression: str) -> Dict[str, Any]:
    # Very limited eval: digits + operators only
    if not all(c in "0123456789+-*/(). " for c in expression):
        return {"error": "Invalid characters in expression."}
    try:
        # Python eval for simple arithmetic only
        result = eval(expression, {"__builtins__": {}})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

LOCAL_TOOLS = {
    "calculator": calculator,
}

def run_local_tool(name: str, args_json: str) -> Dict[str, Any]:
    try:
        args = json.loads(args_json) if isinstance(args_json, str) else args_json
    except Exception:
        return {"error": "Bad JSON arguments"}
    fn = LOCAL_TOOLS.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    return fn(**args)

# 3) One step of model call, optionally with tools
def call_model(messages: List[Dict[str, Any]], enable_tools: bool = True):
    kwargs = {
        "model": MODEL,
        "messages": messages,
    }
    if enable_tools and TOOL_SPEC:
        kwargs["tools"] = TOOL_SPEC
        kwargs["tool_choice"] = "auto"

    return client.chat.completions.create(**kwargs)

# 4) Agent loop: let the model call tools, feed results back, then finalize
def run_agent(user_goal: str, max_rounds: int = 3) -> str:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a concise assistant. Use tools when helpful."},
        {"role": "user", "content": user_goal},
    ]

    for _ in range(max_rounds):
        resp = call_model(messages, enable_tools=True)
        msg = resp.choices[0].message

        # If the assistant called tools, execute and continue the loop
        if getattr(msg, "tool_calls", None):
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
            for tc in msg.tool_calls:
                result = run_local_tool(tc.function.name, tc.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)
                })
            # Continue to next loop iteration so the model can use tool results
            continue

        # No tool calls → final answer
        messages.append({"role": "assistant", "content": msg.content})
        return msg.content.strip()

    return "Reached max rounds without a final answer."

# 5) Example
if __name__ == "__main__":
    print(run_agent("Compute (17*24)+5 and explain the steps briefly."))

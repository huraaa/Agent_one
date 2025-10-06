from typing import Dict, Any

def calculator(expression: str) -> Dict[str, Any]:
    if not all(c in "0123456789.+-*/() " for c in expression):
        return {"error": "Invalid characters"}
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

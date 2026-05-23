import json
import re
from typing import Dict, Any

def extract_json(text: Any) -> Dict[str, Any]:
    """
    Extract the first JSON object from `text`.
    - Handles ```json fences.
    - Finds the first balanced {...} with brace counting.
    - Tries lightweight repairs (auto-closing braces).
    - Always returns a dict: parsed JSON or {"_raw": ..., "_reason": ...}.
    """
    # Fast paths for non-strings
    if isinstance(text, dict):
        return text
    if text is None:
        return {"_raw": "", "_reason": "none_input"}
    if not isinstance(text, str):
        return {"_raw": str(text), "_reason": f"not_a_string:{type(text).__name__}"}

    original = text
    s = text.strip()
    
    # Strip leading ``` or ```json fences and a trailing closing fence
    s = re.sub(r"^```[\w-]*\s*\n", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n```[\s\t]*$", "", s)

    # 0) Whole-string JSON
    try:
        val = json.loads(s)
        return val if isinstance(val, dict) else {"_raw": s, "_reason": "json_root_not_object"}
    except Exception:
        pass

    # 1) Find first balanced { ... }
    start = s.find("{")
    if start == -1:
        return {"_raw": s, "_reason": "no_brace_found"}

    depth = 0
    in_string = False
    escape = False
    end = None

    for i, ch in enumerate(s[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

    candidate = s[start:(end + 1) if end is not None else len(s)]

    # 2) Parse balanced candidate
    if end is not None:
        try:
            val = json.loads(candidate)
            return val if isinstance(val, dict) else {"_raw": candidate, "_reason": "json_root_not_object"}
        except Exception as e:
            return {"_raw": candidate, "_reason": f"parse_error_balanced:{type(e).__name__}"}

    # 3) Try auto-closing braces (only if not inside a string)
    if not in_string and depth > 0:
        repaired = candidate + ("}" * depth)
        try:
            val = json.loads(repaired)
            return val if isinstance(val, dict) else {"_raw": repaired, "_reason": "json_root_not_object"}
        except Exception as e:
            return {"_raw": repaired, "_reason": f"parse_error_repaired:{type(e).__name__}"}

    # 4) Last resort
    return {"_raw": original, "_reason": "unbalanced_in_string_or_unknown"}

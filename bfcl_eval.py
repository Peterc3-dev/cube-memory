#!/usr/bin/env python3
"""
BFCL (Berkeley Function Calling Leaderboard) Evaluation Harness
Evaluates local llama-server tool calling against BFCL v3 Simple and Multiple benchmarks.

Targets: Simple >= 80%, Multiple >= 60%
"""

import json
import sys
import time
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BFCL_DIR = Path("/tmp/bfcl-data")
SERVER_URL = "http://localhost:8090/v1/chat/completions"


TYPE_MAP = {
    "dict": "object",
    "float": "number",
    "tuple": "array",
    "any": "string",
    # These are already valid JSON Schema types:
    # "string", "integer", "boolean", "array", "object", "number"
}


def convert_params_for_openai(params: dict) -> dict:
    """Convert BFCL param format to OpenAI-compatible JSON Schema.
    BFCL uses non-standard types: 'dict', 'float', 'tuple', 'any'.
    Maps them to valid JSON Schema types.
    """
    if not isinstance(params, dict):
        return params

    result = dict(params)

    if "type" in result and isinstance(result["type"], str):
        result["type"] = TYPE_MAP.get(result["type"], result["type"])

    if "properties" in result:
        new_props = {}
        for k, v in result["properties"].items():
            new_props[k] = convert_params_for_openai(v)
        result["properties"] = new_props

    if "items" in result and isinstance(result["items"], dict):
        result["items"] = convert_params_for_openai(result["items"])

    return result


def load_test_cases(category: str) -> list:
    """Load BFCL test cases and ground truth for a category."""
    q_file = BFCL_DIR / f"BFCL_v3_{category}.json"
    a_file = BFCL_DIR / "possible_answer" / f"BFCL_v3_{category}.json"

    questions = []
    with open(q_file) as f:
        for line in f:
            questions.append(json.loads(line))

    answers = {}
    with open(a_file) as f:
        for line in f:
            entry = json.loads(line)
            answers[entry["id"]] = entry["ground_truth"]

    return [(q, answers[q["id"]]) for q in questions if q["id"] in answers]


def make_tools(functions: list) -> list:
    """Convert BFCL function definitions to OpenAI tools format."""
    tools = []
    for func in functions:
        params = convert_params_for_openai(func.get("parameters", {}))
        tools.append({
            "type": "function",
            "function": {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": params,
            }
        })
    return tools


def call_model(messages: list, tools: list, timeout: float = 120) -> dict:
    """Send a tool-calling request to the local llama-server."""
    # Prepend a system message to suppress verbose reasoning
    sys_msg = {
        "role": "system",
        "content": "/no_think\nYou are a helpful assistant. Call the appropriate function to answer the user's question. Be concise."
    }
    messages_with_sys = [sys_msg] + messages

    payload = {
        "model": "local",
        "messages": messages_with_sys,
        "tools": tools,
        "temperature": 0.1,
        "max_tokens": 2048,
    }

    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def extract_tool_calls(response: dict) -> list:
    """Extract function name and arguments from model response."""
    if "error" in response:
        return []

    try:
        choices = response.get("choices", [])
        if not choices:
            return []

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])

        results = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            try:
                args = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            results.append({"name": name, "arguments": args})

        # If no tool_calls but content has <tool_call> tags (fallback parsing)
        if not results and message.get("content"):
            content = message["content"]
            import re
            tool_call_matches = re.findall(
                r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL
            )
            for match in tool_call_matches:
                try:
                    parsed = json.loads(match)
                    results.append({
                        "name": parsed.get("name", ""),
                        "arguments": parsed.get("arguments", {})
                    })
                except json.JSONDecodeError:
                    pass

        return results
    except Exception:
        return []


def coerce_value(val):
    """Coerce a value for comparison - handle string/int/float mismatches."""
    if isinstance(val, str):
        # Try to parse as number
        try:
            if '.' in val:
                return float(val)
            return int(val)
        except ValueError:
            return val.strip().lower()
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val
    return val


def values_match(pred_val, acceptable_values: list) -> bool:
    """Check if a predicted value matches any acceptable value.

    Ground truth acceptable_values is a list where:
    - "" means parameter is optional (can be omitted)
    - Each non-"" entry is an acceptable value
    - For dict entries, the dict itself might use the acceptable-values-list pattern
      (each value in the dict is a list of acceptable values for that key)
    """
    for av in acceptable_values:
        if av == "":
            continue
        # If the acceptable value is a dict with list values,
        # it might be using the nested acceptable-values pattern
        if isinstance(av, dict) and isinstance(pred_val, dict):
            if _match_gt_dict(pred_val, av):
                return True
        if _deep_match(pred_val, av):
            return True
    return False


def _match_gt_dict(pred_dict: dict, gt_dict: dict) -> bool:
    """Match a predicted dict against a ground truth dict.

    In BFCL ground truth, dict values can be:
    - A list of acceptable values (the pattern): [val1, val2, ""]
    - A plain value (direct comparison)
    """
    for key, gt_val in gt_dict.items():
        if key not in pred_dict:
            # Check if optional (has "" in acceptable values)
            if isinstance(gt_val, list) and "" in gt_val:
                continue
            return False

        pred_v = pred_dict[key]

        if isinstance(gt_val, list):
            # This is a list of acceptable values
            # But could also be a literal list value...
            # Heuristic: if the list contains simple scalars or "",
            # treat it as acceptable-values list
            if _looks_like_acceptable_values_list(gt_val):
                if not values_match(pred_v, gt_val):
                    return False
            else:
                # It's a literal list
                if not _deep_match(pred_v, gt_val):
                    return False
        elif isinstance(gt_val, dict):
            if not _match_gt_dict(pred_v, gt_val) and not _deep_match(pred_v, gt_val):
                return False
        else:
            if not _deep_match(pred_v, gt_val):
                return False

    return True


def _looks_like_acceptable_values_list(lst: list) -> bool:
    """Check if a list looks like a BFCL acceptable-values list.
    These contain scalars (str, int, float, bool) or "" and typically 1-5 items.
    A literal list parameter would contain dicts or other complex types.
    """
    if not lst:
        return False
    for item in lst:
        if isinstance(item, (dict, list)):
            return False
    return True


def _deep_match(pred, expected) -> bool:
    """Deep comparison with type coercion."""
    # Direct equality
    if pred == expected:
        return True

    # Both dicts
    if isinstance(pred, dict) and isinstance(expected, dict):
        if set(pred.keys()) != set(expected.keys()):
            return False
        return all(_deep_match(pred[k], expected[k]) for k in pred)

    # Both lists
    if isinstance(pred, list) and isinstance(expected, list):
        if len(pred) != len(expected):
            return False
        return all(_deep_match(p, e) for p, e in zip(pred, expected))

    # Coerced comparison
    pc = coerce_value(pred)
    ec = coerce_value(expected)
    if pc == ec:
        return True

    # String comparison fallback
    if str(pred).lower().strip() == str(expected).lower().strip():
        return True

    return False


def check_answer(predicted: list, ground_truth: list) -> bool:
    """Check if predicted tool calls match ground truth.

    Ground truth format: [{"func_name": {"param1": [acceptable_val1, ...], ...}}]
    Predicted format: [{"name": "func_name", "arguments": {"param1": val, ...}}]

    For Simple: exactly 1 function call expected.
    For Multiple: exactly 1 function call expected (choosing from multiple available).

    A parameter value of "" in the ground truth means the parameter is optional
    (can be omitted or set to its default).
    """
    if not predicted or not ground_truth:
        return False

    # We expect exactly one tool call for both Simple and Multiple
    if len(predicted) < 1:
        return False

    pred = predicted[0]
    gt = ground_truth[0]  # First (and typically only) acceptable answer

    # Get expected function name and params
    gt_func_name = list(gt.keys())[0]
    gt_params = gt[gt_func_name]

    # Check function name
    if pred["name"] != gt_func_name:
        return False

    pred_args = pred["arguments"]

    # Check each expected parameter
    for param_name, acceptable_values in gt_params.items():
        pred_val = pred_args.get(param_name)

        # If "" is in acceptable values, the param is optional
        if "" in acceptable_values:
            if pred_val is None:
                # Parameter omitted - that's fine
                continue
            # Parameter provided - check if it matches any non-empty acceptable value
            non_empty = [v for v in acceptable_values if v != ""]
            if non_empty:
                if not values_match(pred_val, non_empty):
                    return False
        else:
            # Parameter is required
            if pred_val is None:
                return False
            if not values_match(pred_val, acceptable_values):
                return False

    return True


def run_eval(category: str, max_cases: int = 0, verbose: bool = False) -> dict:
    """Run evaluation on a BFCL category."""
    print(f"\n{'='*60}", flush=True)
    print(f"BFCL v3 {category.upper()} Evaluation", flush=True)
    print(f"{'='*60}", flush=True)

    test_cases = load_test_cases(category)
    if max_cases > 0:
        test_cases = test_cases[:max_cases]

    total = len(test_cases)
    correct = 0
    errors = 0
    failures = []
    timings = []

    for i, (question, ground_truth) in enumerate(test_cases):
        messages = question["question"][0]  # Unwrap the outer list
        tools = make_tools(question["function"])

        t0 = time.time()
        response = call_model(messages, tools)
        elapsed = time.time() - t0
        timings.append(elapsed)

        predicted = extract_tool_calls(response)
        is_correct = check_answer(predicted, ground_truth)

        if is_correct:
            correct += 1
        elif "error" in response:
            errors += 1
            failures.append({
                "id": question["id"],
                "error": response["error"],
            })
        else:
            failures.append({
                "id": question["id"],
                "expected": ground_truth,
                "predicted": predicted,
            })

        status = "PASS" if is_correct else "FAIL"
        pct = correct / (i + 1) * 100

        if verbose or not is_correct:
            gt_name = list(ground_truth[0].keys())[0] if ground_truth else "?"
            pred_name = predicted[0]["name"] if predicted else "none"
            print(f"  [{i+1:3d}/{total}] {status} | {question['id']:20s} | "
                  f"expected={gt_name} got={pred_name} | "
                  f"running={pct:.1f}% | {elapsed:.1f}s", flush=True)
        elif (i + 1) % 20 == 0:
            avg_t = sum(timings) / len(timings)
            eta = avg_t * (total - i - 1)
            print(f"  [{i+1:3d}/{total}] running accuracy: {pct:.1f}% | "
                  f"avg {avg_t:.1f}s/case | ETA {eta:.0f}s", flush=True)

    accuracy = correct / total * 100
    avg_time = sum(timings) / len(timings) if timings else 0

    print(f"\n{'─'*60}", flush=True)
    print(f"Results: {correct}/{total} = {accuracy:.1f}%", flush=True)
    print(f"Errors: {errors}", flush=True)
    print(f"Avg time per case: {avg_time:.1f}s", flush=True)
    print(f"Total time: {sum(timings):.0f}s", flush=True)

    if failures and len(failures) <= 30:
        print(f"\nFailure details:", flush=True)
        for f in failures[:30]:
            print(f"  {f['id']}: expected={f.get('expected','err')} got={f.get('predicted','err')}", flush=True)

    return {
        "category": category,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "errors": errors,
        "avg_time": avg_time,
        "failures": failures,
    }


def main():
    global SERVER_URL

    parser = argparse.ArgumentParser(description="BFCL v3 Evaluation Harness")
    parser.add_argument("--category", choices=["simple", "multiple", "both"],
                        default="both", help="Which category to evaluate")
    parser.add_argument("--max-cases", type=int, default=0,
                        help="Max test cases per category (0 = all)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show every test case result")
    parser.add_argument("--server", default=SERVER_URL,
                        help="Server URL")
    args = parser.parse_args()

    SERVER_URL = args.server

    # Quick health check
    try:
        r = requests.get("http://localhost:8090/health", timeout=5)
        if r.status_code != 200:
            print("ERROR: llama-server not healthy")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot reach llama-server: {e}")
        sys.exit(1)

    results = []

    if args.category in ("simple", "both"):
        results.append(run_eval("simple", args.max_cases, args.verbose))

    if args.category in ("multiple", "both"):
        results.append(run_eval("multiple", args.max_cases, args.verbose))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        target = 80 if r["category"] == "simple" else 60
        status = "PASS" if r["accuracy"] >= target else "FAIL"
        print(f"  {r['category']:10s}: {r['accuracy']:5.1f}% (target: {target}%) [{status}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

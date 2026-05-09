#!/usr/bin/env python3
"""Quick integration test for the Cube Memory tool-calling endpoint."""

import json
import requests
import sys

BASE_URL = "http://localhost:8090"

def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    print("  [OK] Health check")

def test_simple_tool_call():
    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": "local",
        "messages": [
            {"role": "system", "content": "/no_think\nYou are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        }],
        "temperature": 0.1,
        "max_tokens": 512,
    }, timeout=60)

    assert resp.status_code == 200, f"Request failed: {resp.status_code}"
    data = resp.json()
    msg = data["choices"][0]["message"]
    assert msg.get("tool_calls"), "No tool calls in response"
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather", f"Wrong function: {tc['function']['name']}"
    args = json.loads(tc["function"]["arguments"])
    assert "city" in args, "Missing city argument"
    assert "tokyo" in args["city"].lower(), f"Wrong city: {args['city']}"
    print(f"  [OK] Simple tool call: get_weather(city={args['city']})")

def test_multiple_tools():
    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": "local",
        "messages": [
            {"role": "system", "content": "/no_think\nYou are a helpful assistant."},
            {"role": "user", "content": "Send an email to alice@example.com with subject 'Meeting' and body 'See you at 3pm'"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send an email to a recipient",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "subject", "body"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_contacts",
                    "description": "Search contacts by name",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }, timeout=60)

    assert resp.status_code == 200
    data = resp.json()
    msg = data["choices"][0]["message"]
    assert msg.get("tool_calls"), "No tool calls"
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == "send_email", f"Wrong function: {tc['function']['name']}"
    args = json.loads(tc["function"]["arguments"])
    assert args.get("to") == "alice@example.com"
    print(f"  [OK] Multiple tools selection: send_email(to={args['to']})")

def test_tool_response_roundtrip():
    """Test that the model can process tool results and give a final answer."""
    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": "local",
        "messages": [
            {"role": "system", "content": "/no_think\nYou are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}
            }]},
            {"role": "tool", "content": '{"temperature": 18, "condition": "sunny", "humidity": 45}', "tool_call_id": "call_1"},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }],
        "temperature": 0.1,
        "max_tokens": 256,
    }, timeout=60)

    assert resp.status_code == 200
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    assert content, "No content in response after tool result"
    assert "18" in content or "sunny" in content.lower() or "paris" in content.lower(), \
        f"Response doesn't mention weather data: {content[:200]}"
    print(f"  [OK] Tool response roundtrip: '{content[:80]}...'")


if __name__ == "__main__":
    print("Cube Memory Endpoint Integration Tests")
    print("=" * 50)

    tests = [test_health, test_simple_tool_call, test_multiple_tools, test_tool_response_roundtrip]
    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed")
    sys.exit(0 if failed == 0 else 1)

"""Unified Test Runner for Agent RAG MCP.
Supports local and remote environments with multi-scenario simulations.
"""
import asyncio
import argparse
import json
import sys
from typing import Any, Dict, List
from fastmcp import Client
from fastmcp.client.transports import SSETransport

# ==============================================================================
# Scenarios
# ==============================================================================
SCENARIOS = [
    {
        "id": "python-fastapi-singleton",
        "name": "Python FastAPI Singleton",
        "language": "Python",
        "framework": "FastAPI",
        "pattern": "Singleton",
        "details": "Database connection pool management",
        "success_code": "class DB:\n  _instance = None\n  def __new__(cls):\n    if not cls._instance: cls._instance = super().__new__(cls)\n    return cls._instance",
        "error_details": "RecursionError when implementing Singleton with __init__",
    },
    {
        "id": "javascript-react-hook",
        "name": "JavaScript React Custom Hook",
        "language": "JavaScript",
        "framework": "React",
        "pattern": "Custom Hook",
        "details": "Persistent local storage state hook",
        "success_code": "function useLocalStorage(key, init) {\n  const [val, setVal] = useState(() => JSON.parse(localStorage.getItem(key)) || init);\n  useEffect(() => localStorage.setItem(key, JSON.stringify(val)), [val]);\n  return [val, setVal];\n}",
        "error_details": "Infinite loop in useEffect when setting state",
    },
    {
        "id": "skript-minecraft-cooldown",
        "name": "Skript Minecraft Cooldown",
        "language": "Skript",
        "framework": "Minecraft",
        "pattern": "Cooldown System",
        "details": "Action cooldown using metadata or variables",
        "success_code": "on right click:\n  if {cd::%player%} is not set:\n    set {cd::%player%} to now\n    add 5 seconds to {cd::%player%}\n    send \"Action!\"\n  else if now < {cd::%player%}:\n    send \"Wait!\"\n  else:\n    delete {cd::%player%}",
        "error_details": "Cooldown never expires due to date comparison mismatch",
    },
    {
        "id": "java-spring-repository",
        "name": "Java Spring Boot Repository",
        "language": "Java",
        "framework": "Spring Boot",
        "pattern": "Repository Pattern",
        "details": "Data access layer with JPA",
        "success_code": "@Repository\npublic interface UserRepository extends JpaRepository<User, Long> {\n  Optional<User> findByEmail(String email);\n}",
        "error_details": "LazyInitializationException when accessing relations outside transaction",
    }
]

# ==============================================================================
# Helper functions
# ==============================================================================
def format_toon(data: Dict[str, Any]) -> str:
    """Very simple dict to TOON-like YAML string converter."""
    import yaml
    return yaml.dump(data, allow_unicode=True, sort_keys=False)

def print_banner(text: str):
    print(f"\n{'='*80}\n {text} \n{'='*80}")

async def run_scenario(client: Client, scenario: Dict[str, Any], use_toon: bool = True):
    print(f"\n▶ Scenario: {scenario['name']}")
    
    # 1. Tell Success
    print("  [1/3] Reporting SUCCESS experience...")
    report = {
        "request": {
            "language": scenario["language"],
            "framework": scenario["framework"],
            "design_context": {"pattern": scenario["pattern"]},
            "content": {
                "result": "SUCCESS",
                "feature_details": scenario["details"],
                "code": {"success": scenario["success_code"]}
            }
        }
    }
    input_str = format_toon(report) if use_toon else json.dumps(report)
    res = await client.call_tool("tell_code_pattern", {"request_data": input_str})
    print(f"  ✅ Reported. ID: {res.content[0].text.split('ID: ')[-1].split(')')[0] if 'ID:' in res.content[0].text else 'N/A'}")

    # 2. Ask (Search)
    print(f"  [2/3] Asking for best practice of {scenario['pattern']}...")
    ask = {
        "request": {
            "language": scenario["language"],
            "framework": scenario["framework"],
            "design_context": {"pattern": scenario["pattern"]},
            "content": {"feature_details": f"How to implement {scenario['pattern']}?"}
        }
    }
    input_str = format_toon(ask) if use_toon else json.dumps(ask)
    res = await client.call_tool("ask_code_pattern", {"request_data": input_str})
    print(f"  ✅ Advice (Preview): {res.content[0].text[:150].replace('\n', ' ')}...")

    # 3. Tell Failure (Immediate Advice)
    print("  [3/3] Reporting FAILURE and getting advice...")
    failure = {
        "request": {
            "language": scenario["language"],
            "framework": scenario["framework"],
            "design_context": {"pattern": scenario["pattern"]},
            "content": {
                "result": "FAILED",
                "feature_details": scenario["error_details"]
            }
        }
    }
    input_str = format_toon(failure) if use_toon else json.dumps(failure)
    res = await client.call_tool("tell_code_pattern", {"request_data": input_str})
    print(f"  ✅ Fix Suggested (Preview): {res.content[0].text.split('過去の成功事例に基づく改善案:')[-1][:150].strip().replace('\n', ' ')}...")

async def run_rag_test(client: Client):
    print_banner("Testing Document RAG (ask_project_document)")
    question = "このプロジェクトの目的は何ですか？"
    print(f"❓ Question: {question}")
    res = await client.call_tool("ask_project_document", {"question": question})
    print(f"✅ Answer (Preview): {res.content[0].text[:300].replace('\n', ' ')}...")

# ==============================================================================
# Main
# ==============================================================================
async def main():
    parser = argparse.ArgumentParser(description="Unified Test Runner")
    parser.add_argument("--env", choices=["local", "remote"], default="local", help="Testing environment")
    parser.add_argument("--url", help="Server URL (for remote env)")
    parser.add_argument("--token", help="Auth token")
    parser.add_argument("--scenarios", help="Comma-separated list of scenario IDs to run")
    parser.add_argument("--json", action="store_true", help="Use JSON instead of TOON")
    args = parser.parse_args()

    url = args.url or ("http://127.0.0.1:8000/sse" if args.env == "local" else None)
    if not url:
        print("Error: --url is required for remote environment.")
        sys.exit(1)

    print_banner(f"Agent RAG MCP Unified Tester - Env: {args.env.upper()}")
    print(f"Server URL: {url}")
    
    transport = SSETransport(url)
    client = Client(transport, auth=args.token)
    
    try:
        async with client:
            # 1. RAG Test
            await run_rag_test(client)
            
            # 2. Scenario Tests
            print_banner("Testing Dynamic Learning Scenarios")
            target_ids = args.scenarios.split(",") if args.scenarios else None
            for scenario in SCENARIOS:
                if target_ids and scenario["id"] not in target_ids:
                    continue
                await run_scenario(client, scenario, use_toon=not args.json)
                
            print_banner("ALL TESTS COMPLETED SUCCESSFULLY")
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

import toon_format

toon_input = """
request:
  language: Python
  framework: FastMCP
  design_context:
    pattern: Tool Verification
  content:
    feature_details: Testing TOON support
""".strip()

print(f"Input:\n{toon_input}")
print("-" * 20)

try:
    parsed = toon_format.decode(toon_input)
    print("Success!")
    print(parsed)
except Exception as e:
    print(f"Failed: {e}")
    # Print traceback
    import traceback
    traceback.print_exc()

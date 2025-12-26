from google.genai import types
print("types.Tool fields:")
print(list(types.Tool.model_fields.keys()))
print("-" * 20)
print("types.FileSearch fields:")
try:
    print(list(types.FileSearch.model_fields.keys()))
except AttributeError:
    print("types.FileSearch not found")

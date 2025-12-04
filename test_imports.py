import importlib, sys

modules = [
    "model.unet",
    "model.infer",
    "utils.preprocessing",
    "utils.decode",
    "flask_app.app"
]

for m in modules:
    try:
        importlib.import_module(m)
        print(m, "OK")
    except Exception as e:
        print(m, "ERROR ->", e)

print("\nTest Completed.")

import sys
sys.path.insert(0, r"C:\Users\HPL\Desktop\vs code\predict_fifa2026\predictfifa2026\Lib\site-packages")

def test_import(name):
    print(f"Testing {name}...", flush=True)
    try:
        __import__(name)
        print(f"  {name}: SUCCESS", flush=True)
    except Exception as e:
        print(f"  {name}: FAILED ({e})", flush=True)
    except BaseException as e:
        print(f"  {name}: BASE EXCEPTION ({type(e).__name__})", flush=True)

test_import('numpy')
test_import('pandas')
test_import('scipy')
test_import('sklearn')

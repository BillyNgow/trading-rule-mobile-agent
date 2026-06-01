import os

# Must be set before screen.py is imported; it raises ValueError at module level
# if the key is absent.
os.environ.setdefault("ALPHA_VANTAGE_API_KEY", "test_key")

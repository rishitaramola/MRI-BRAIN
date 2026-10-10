"""
main.py
~~~~~~~
Root entrypoint for ASGI servers and zero-configuration Vercel detection.
Exports FastAPI `app` from src.inference.predict.
"""

from src.inference.predict import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)

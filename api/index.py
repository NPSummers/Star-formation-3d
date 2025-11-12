import os

# Enable serverless mode inside the Flask app
os.environ.setdefault("SERVERLESS", "1")

from app.web import app as flask_app  # noqa: E402

# Export a WSGI 'app' for Vercel Python runtime
app = flask_app

# Optional explicit handler (some runtimes accept this too)
def handler(environ, start_response):
    return app(environ, start_response)



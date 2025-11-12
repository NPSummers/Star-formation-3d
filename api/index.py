import os

# Enable serverless mode inside the Flask app
os.environ.setdefault("SERVERLESS", "1")

from app.web import app as flask_app  # noqa: E402


def handler(environ, start_response):
    # Flask app is a WSGI callable
    return flask_app(environ, start_response)



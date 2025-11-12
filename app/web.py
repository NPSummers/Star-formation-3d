from __future__ import annotations

import io
import os
import threading
import time
from typing import Optional

import pygame
import secrets
from flask import Flask, Response, make_response, render_template_string
from PIL import Image

from app.config import settings as C
from app.physics.engine import Simulation
from app.visualization.render import draw

# Ensure SDL doesn't try to open a desktop window
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

app = Flask(__name__)

_frame_lock = threading.Lock()
_latest_frame_jpeg: Optional[bytes] = None
_running = True


def _surface_to_jpeg_bytes(surface: pygame.Surface, quality: int = 80) -> bytes:
    raw_str = pygame.image.tostring(surface, "RGB")
    image = Image.frombytes("RGB", (surface.get_width(), surface.get_height()), raw_str)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _sim_loop() -> None:
    global _latest_frame_jpeg

    pygame.init()
    try:
        font = pygame.font.SysFont("Menlo,Consolas,Monaco,monospace", 16)
    except (NotImplementedError, AttributeError):
        try:
            font = pygame.font.Font(None, 16)
        except Exception:
            font = None

    screen = pygame.Surface((C.WINDOW_WIDTH, C.WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    sim = Simulation(seed=secrets.randbits(32))

    paused = C.PAUSED_AT_START
    show_grid = False
    show_info = True
    show_heatmap = C.HEATMAP_ENABLED
    center = (0.0, 0.0)
    zoom = C.WINDOW_HEIGHT / (2.4 * C.BOX_SIZE)

    # Target FPS for simulation/render
    target_fps = min(30, C.MAX_FPS)
    last_reset_time = time.time()

    while _running:
        if not paused:
            sim.step()

        draw(sim, screen, font, center, zoom, show_grid, show_info, show_heatmap)

        try:
            frame_bytes = _surface_to_jpeg_bytes(screen, quality=80)
            with _frame_lock:
                _latest_frame_jpeg = frame_bytes
        except Exception:
            # If encoding fails for some reason, skip this frame
            pass

        # Throttle to target FPS
        clock.tick(target_fps)

        # Auto-reset with new seed every 60 seconds
        now = time.time()
        if now - last_reset_time >= 60.0:
            sim = Simulation(seed=secrets.randbits(32))
            last_reset_time = now

    pygame.quit()


@app.route("/")
def index() -> str:
    return render_template_string(
        """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Star Formation - Canvas</title>
    <style>
      html, body {
        background: #0b0e16;
        color: #e6ebf5;
        margin: 0;
        padding: 0;
        height: 100%;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      }
      .wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100%;
        padding: 12px;
        box-sizing: border-box;
        gap: 12px;
        flex-direction: column;
      }
      canvas {
        border-radius: 8px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
        background: #05070c;
        image-rendering: pixelated;
      }
      .hint {
        opacity: 0.8;
        font-size: 14px;
      }
      .controls {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }
      button {
        background: #1d2436;
        color: #e6ebf5;
        border: 1px solid #2a344d;
        padding: 6px 10px;
        border-radius: 6px;
        cursor: pointer;
      }
      button:hover {
        background: #232c42;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <canvas id="canvas" width="{{w}}" height="{{h}}"></canvas>
    </div>

    <script>
      (function(){
        const w = {{w}};
        const h = {{h}};
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        // Avoid overlapping loads
        let loading = false;

        async function drawLoop() {
          if (!document.body) return;
          if (!loading) {
            loading = true;
            try {
              const img = new Image();
              img.onload = function() {
                try {
                  ctx.drawImage(img, 0, 0, w, h);
                } finally {
                  loading = false;
                }
              };
              img.onerror = function() { loading = false; };
              img.crossOrigin = "anonymous";
              img.src = "/frame.jpg?t=" + Date.now();
            } catch (e) {
              loading = false;
            }
          }
          requestAnimationFrame(drawLoop);
        }
        requestAnimationFrame(drawLoop);
      })();
    </script>
  </body>
  </html>
        """,
        w=C.WINDOW_WIDTH,
        h=C.WINDOW_HEIGHT,
    )


@app.route("/frame.jpg")
def frame_jpg() -> Response:
    # If the simulation hasn't produced a frame yet, wait briefly
    deadline = time.time() + 2.0
    while True:
        with _frame_lock:
            frame = _latest_frame_jpeg
        if frame is not None:
            resp = make_response(frame)
            resp.headers["Content-Type"] = "image/jpeg"
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            return resp
        if time.time() > deadline:
            # Return 503 to signal no frame ready yet
            return Response(status=503)
        time.sleep(0.01)


def _start_background_sim_once() -> None:
    # Start the simulation thread only once
    if not getattr(_start_background_sim_once, "_started", False):
        t = threading.Thread(target=_sim_loop, daemon=True)
        t.start()
        _start_background_sim_once._started = True  # type: ignore[attr-defined]


if __name__ == "__main__":
    # Start sim immediately if running this file directly
    _start_background_sim_once()
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)



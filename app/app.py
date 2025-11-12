from __future__ import annotations

import os
import sys
import time

import pygame
import secrets

from app.config import settings as C
from app.physics.engine import Simulation
from app.visualization.render import draw


def run() -> None:
    pygame.init()
    pygame.display.set_caption("2D Star Formation Simulator (SPH + Barnes-Hut)")
    screen = pygame.display.set_mode((C.WINDOW_WIDTH, C.WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont("Menlo,Consolas,Monaco,monospace", 16)
    except (NotImplementedError, AttributeError):
        try:
            font = pygame.font.Font(None, 16)
        except Exception:
            font = None

    os.makedirs(C.SCREENSHOT_DIR, exist_ok=True)

    sim = Simulation(seed=secrets.randbits(32))

    paused = C.PAUSED_AT_START
    show_grid = False
    show_info = True
    show_heatmap = C.HEATMAP_ENABLED

    center = (0.0, 0.0)
    zoom = C.WINDOW_HEIGHT / (2.4 * C.BOX_SIZE)

    dragging = False
    drag_start = (0, 0)
    center_start = center

    last_fps_stamp = time.time()
    fps = 0.0
    last_reset_time = time.time()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    sim = Simulation(seed=secrets.randbits(32))
                    last_reset_time = time.time()
                elif event.key == pygame.K_g:
                    show_grid = not show_grid
                elif event.key == pygame.K_i:
                    show_info = not show_info
                elif event.key == pygame.K_h:
                    show_heatmap = not show_heatmap
                elif event.key == pygame.K_s:
                    path = os.path.join(C.SCREENSHOT_DIR, f"screenshot_{time.time():.0f}.png")
                    pygame.image.save(screen, path)
                    print(f"Saved {path}")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    drag_start = event.pos
                    center_start = center
                elif event.button == 4:
                    zoom = min(4000.0, zoom * 1.1)
                elif event.button == 5:
                    zoom = max(20.0, zoom / 1.1)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - drag_start[0]
                dy = event.pos[1] - drag_start[1]
                center = (
                    center_start[0] - dx / zoom,
                    center_start[1] - dy / zoom,
                )

        if not paused:
            sim.step()

        draw(sim, screen, font, center, zoom, show_grid, show_info, show_heatmap)
        pygame.display.flip()

        clock.tick(C.MAX_FPS)
        now = time.time()
        # Auto-reset with a new seed every 60 seconds
        if now - last_reset_time >= 60.0:
            sim = Simulation(seed=secrets.randbits(32))
            last_reset_time = now
        if now - last_fps_stamp > 0.25:
            fps = clock.get_fps()
            last_fps_stamp = now
        pygame.display.set_caption(f"2D Star Formation Simulator - {fps:.1f} FPS")

    pygame.quit()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)


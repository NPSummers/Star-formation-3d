from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pygame

from app.config import settings as C
from app.physics.engine import Simulation, cubic_spline_kernel


def world_to_screen(x: float, y: float, center: Tuple[float, float], zoom: float) -> Tuple[int, int]:
    cx, cy = center
    sx = (x - cx) * zoom + C.WINDOW_WIDTH * 0.5
    sy = (y - cy) * zoom + C.WINDOW_HEIGHT * 0.5
    return int(sx), int(sy)


def compute_density_colors(rho: np.ndarray) -> np.ndarray:
    r = np.clip(rho / (rho.mean() + 1e-6), 0.0, 5.0) ** C.DENSITY_COLOR_SCALING
    r = (r - r.min()) / max(1e-6, (r.max() - r.min()))
    base = np.array(C.GAS_COLOR, dtype=np.float64)[None, :] / 255.0
    bright = np.array([1.0, 1.0, 1.0])[None, :]
    col = base * (1.0 - r[:, None]) + bright * r[:, None]
    return (col * 255.0).astype(np.uint8)


def get_colormap_color(value: float, colormap: str = "viridis") -> Tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    if colormap == "viridis":
        if value < 0.25:
            r, g, b = 0.267, 0.005, 0.329
            t = value / 0.25
            r, g, b = r + t * (0.208 - r), g + t * (0.112 - g), b + t * (0.577 - b)
        elif value < 0.5:
            r, g, b = 0.208, 0.112, 0.577
            t = (value - 0.25) / 0.25
            r, g, b = r + t * (0.278 - r), g + t * (0.379 - g), b + t * (0.763 - b)
        elif value < 0.75:
            r, g, b = 0.278, 0.379, 0.763
            t = (value - 0.5) / 0.25
            r, g, b = r + t * (0.298 - r), g + t * (0.651 - g), b + t * (0.761 - b)
        else:
            r, g, b = 0.298, 0.651, 0.761
            t = (value - 0.75) / 0.25
            r, g, b = r + t * (0.993 - r), g + t * (0.906 - g), b + t * (0.144 - b)
        return (int(r * 255), int(g * 255), int(b * 255))
    elif colormap == "plasma":
        if value < 0.25:
            r, g, b = 0.142, 0.0, 0.35
            t = value / 0.25
            r, g, b = r + t * (0.396 - r), g + t * (0.0 - g), b + t * (0.12 - b)
        elif value < 0.5:
            r, g, b = 0.396, 0.0, 0.12
            t = (value - 0.25) / 0.25
            r, g, b = r + t * (0.743 - r), g + t * (0.132 - g), b + t * (0.186 - b)
        elif value < 0.75:
            r, g, b = 0.743, 0.132, 0.186
            t = (value - 0.5) / 0.25
            r, g, b = r + t * (0.988 - r), g + t * (0.493 - g), b + t * (0.053 - b)
        else:
            r, g, b = 0.988, 0.493, 0.053
            t = (value - 0.75) / 0.25
            r, g, b = r + t * (0.94 - r), g + t * (0.978 - g), b + t * (0.999 - b)
        return (int(r * 255), int(g * 255), int(b * 255))
    elif colormap == "hot":
        if value < 0.33:
            r = value / 0.33
            g, b = 0.0, 0.0
        elif value < 0.66:
            r = 1.0
            g = (value - 0.33) / 0.33
            b = 0.0
        else:
            r, g = 1.0, 1.0
            b = (value - 0.66) / 0.34
        return (int(r * 255), int(g * 255), int(b * 255))
    elif colormap == "cool":
        r = value
        g = 1.0 - value
        b = 1.0
        return (int(r * 255), int(g * 255), int(b * 255))
    elif colormap == "inferno":
        if value < 0.33:
            r = value * 3.0 * 0.402
            g, b = 0.0, value * 3.0 * 0.389
        elif value < 0.66:
            r = 0.402 + (value - 0.33) * 3.0 * (0.931 - 0.402)
            g = (value - 0.33) * 3.0 * 0.574
            b = 0.389 + (value - 0.33) * 3.0 * (0.998 - 0.389)
        else:
            r = 0.931 + (value - 0.66) * 3.0 * (1.0 - 0.931)
            g = 0.574 + (value - 0.66) * 3.0 * (1.0 - 0.574)
            b = 0.998
        return (int(min(1.0, r) * 255), int(min(1.0, g) * 255), int(min(1.0, b) * 255))
    else:
        if value < 0.33:
            r = value * 3.0 * 0.462
            g = 0.0
            b = value * 3.0 * 0.634
        elif value < 0.66:
            r = 0.462 + (value - 0.33) * 3.0 * (0.929 - 0.462)
            g = (value - 0.33) * 3.0 * 0.487
            b = 0.634 + (value - 0.33) * 3.0 * (0.768 - 0.634)
        else:
            r = 0.929 + (value - 0.66) * 3.0 * (1.0 - 0.929)
            g = 0.487 + (value - 0.66) * 3.0 * (1.0 - 0.487)
            b = 0.768 + (value - 0.66) * 3.0 * (1.0 - 0.768)
        return (int(min(1.0, r) * 255), int(min(1.0, g) * 255), int(min(1.0, b) * 255))


def render_heatmap(sim: Simulation, screen: pygame.Surface, center: Tuple[float, float], zoom: float) -> None:
    if sim.N == 0:
        return
    sim.grid.build(sim.gas.pos)
    res = C.HEATMAP_RESOLUTION
    h = sim.h
    heatmap_surf = pygame.Surface((C.WINDOW_WIDTH, C.WINDOW_HEIGHT), pygame.SRCALPHA)
    inv_zoom = 1.0 / zoom
    world_left = center[0] - C.WINDOW_WIDTH * 0.5 * inv_zoom
    world_right = center[0] + C.WINDOW_WIDTH * 0.5 * inv_zoom
    world_top = center[1] - C.WINDOW_HEIGHT * 0.5 * inv_zoom
    world_bottom = center[1] + C.WINDOW_HEIGHT * 0.5 * inv_zoom
    world_dx = (world_right - world_left) / res
    world_dy = (world_bottom - world_top) / res
    screen_dx = C.WINDOW_WIDTH / res
    screen_dy = C.WINDOW_HEIGHT / res
    if C.HEATMAP_SHOW_DENSITY:
        field_values = sim.gas.rho
    else:
        field_values = sim.gas.u
    if field_values.size == 0:
        return
    field_min = float(np.min(field_values))
    field_max = float(np.max(field_values))
    field_range = max(1e-6, field_max - field_min)
    for j in range(res):
        for i in range(res):
            world_x = world_left + (i + 0.5) * world_dx
            world_y = world_top + (j + 0.5) * world_dy
            if abs(world_x) > C.BOX_SIZE or abs(world_y) > C.BOX_SIZE:
                continue
            value = 0.0
            weight_sum = 0.0
            neighbors = sim.grid.neighbors(world_x, world_y)
            for k in neighbors:
                dx = world_x - float(sim.gas.pos[k, 0])
                dy = world_y - float(sim.gas.pos[k, 1])
                r = math.hypot(dx, dy)
                if r < 2.0 * h:
                    # SPH interpolation: f(x) ≈ Σ m_j f_j W(|x - x_j|, h) / Σ m_j W(|x - x_j|, h)
                    w = cubic_spline_kernel(r, h)
                    mass = sim.mass
                    field_val = float(field_values[k])
                    value += mass * field_val * w
                    weight_sum += mass * w
            if weight_sum > 1e-8:
                value /= weight_sum
            else:
                value = field_min
            normalized = (value - field_min) / field_range
            color = get_colormap_color(normalized, C.HEATMAP_COLORMAP)
            alpha = int(C.HEATMAP_OPACITY * 255)
            color_with_alpha = (*color, alpha)
            screen_x = int(i * screen_dx)
            screen_y = int(j * screen_dy)
            screen_w = int(math.ceil(screen_dx)) + 1
            screen_h = int(math.ceil(screen_dy)) + 1
            pygame.draw.rect(heatmap_surf, color_with_alpha, (screen_x, screen_y, screen_w, screen_h))
    screen.blit(heatmap_surf, (0, 0))


def draw(sim: Simulation, screen: pygame.Surface, font: pygame.font.Font | None, center: Tuple[float, float], zoom: float, show_grid: bool, show_info: bool, show_heatmap: bool) -> None:
    screen.fill(C.BACKGROUND_COLOR)
    if show_grid:
        step = max(40, int(zoom * 0.1))
        for x in range(0, C.WINDOW_WIDTH, step):
            pygame.draw.line(screen, C.GRID_COLOR, (x, 0), (x, C.WINDOW_HEIGHT), 1)
        for y in range(0, C.WINDOW_HEIGHT, step):
            pygame.draw.line(screen, C.GRID_COLOR, (0, y), (C.WINDOW_WIDTH, y), 1)
    if show_heatmap:
        render_heatmap(sim, screen, center, zoom)
    if sim.N > 0:
        colors = compute_density_colors(sim.gas.rho) if C.DRAW_SPH_GRADIENT else np.array([C.GAS_COLOR] * sim.N, dtype=np.uint8)
        pts = [world_to_screen(float(sim.gas.pos[i, 0]), float(sim.gas.pos[i, 1]), center, zoom) for i in range(sim.N)]
        for (x, y), col in zip(pts, colors):
            pygame.draw.circle(screen, tuple(int(c) for c in col), (x, y), C.PARTICLE_DRAW_SIZE)
    for s in sim.sinks:
        x, y = world_to_screen(s.x, s.y, center, zoom)
        pygame.draw.circle(screen, C.STAR_COLOR, (x, y), C.STAR_DRAW_SIZE)
    if show_info and font is not None:
        field_type = "density" if C.HEATMAP_SHOW_DENSITY else "temperature"
        info_lines = [
            f"t={sim.time:.3f}  N_gas={sim.N}  N_star={len(sim.sinks)}",
            f"dt={sim.dt:.4g}  h={sim.h:.3f}  theta={sim.theta:.2f}",
            f"Heatmap: {'ON' if show_heatmap else 'OFF'} ({field_type}, {C.HEATMAP_COLORMAP})",
            "Controls: Space=Pause, R=Reset, G=Grid, I=Info, H=Heatmap, S=Screenshot, Wheel=Zoom, Drag=Pan",
        ]
        y0 = 8
        for line in info_lines:
            text_surf = font.render(line, True, C.TEXT_COLOR)
            screen.blit(text_surf, (8, y0))
            y0 += 20


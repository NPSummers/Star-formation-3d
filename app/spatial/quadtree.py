from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Body:
    x: float
    y: float
    mass: float


class Quad:
    __slots__ = ("cx", "cy", "half")

    def __init__(self, cx: float, cy: float, half: float) -> None:
        self.cx = cx
        self.cy = cy
        self.half = half

    def contains(self, x: float, y: float) -> bool:
        return (
            (self.cx - self.half) <= x <= (self.cx + self.half)
            and (self.cy - self.half) <= y <= (self.cy + self.half)
        )

    def subdivide(self) -> Tuple["Quad", "Quad", "Quad", "Quad"]:
        h2 = self.half * 0.5
        return (
            Quad(self.cx - h2, self.cy - h2, h2),
            Quad(self.cx + h2, self.cy - h2, h2),
            Quad(self.cx - h2, self.cy + h2, h2),
            Quad(self.cx + h2, self.cy + h2, h2),
        )


class Node:
    __slots__ = (
        "quad",
        "body",
        "mass",
        "com_x",
        "com_y",
        "sw",
        "se",
        "nw",
        "ne",
    )

    def __init__(self, quad: Quad) -> None:
        self.quad = quad
        self.body: Optional[Body] = None
        self.mass = 0.0
        self.com_x = 0.0
        self.com_y = 0.0
        self.sw: Optional[Node] = None
        self.se: Optional[Node] = None
        self.nw: Optional[Node] = None
        self.ne: Optional[Node] = None

    def is_external(self) -> bool:
        return self.sw is None and self.se is None and self.nw is None and self.ne is None

    def _update_com(self, b: Body) -> None:
        m = self.mass + b.mass
        if m <= 0.0:
            return
        self.com_x = (self.com_x * self.mass + b.x * b.mass) / m
        self.com_y = (self.com_y * self.mass + b.y * b.mass) / m
        self.mass = m

    def insert(self, b: Body) -> None:
        if self.body is None and self.is_external() and self.mass == 0.0:
            self.body = b
            self.mass = b.mass
            self.com_x = b.x
            self.com_y = b.y
            return

        if not self.is_external():
            self._update_com(b)
            self._put_into_child(b)
            return

        existing = self.body
        self.body = None
        self.sw, self.se, self.nw, self.ne = self.quad.subdivide()
        self.sw = Node(self.sw)
        self.se = Node(self.se)
        self.nw = Node(self.nw)
        self.ne = Node(self.ne)

        if existing is not None:
            self._put_into_child(existing)
        self._put_into_child(b)

        self.mass = 0.0
        self.com_x = 0.0
        self.com_y = 0.0
        if existing is not None:
            self._update_com(existing)
        self._update_com(b)

    def _put_into_child(self, b: Body) -> None:
        x, y = b.x, b.y
        if self.sw.quad.contains(x, y):
            self.sw.insert(b)
        elif self.se.quad.contains(x, y):
            self.se.insert(b)
        elif self.nw.quad.contains(x, y):
            self.nw.insert(b)
        else:
            self.ne.insert(b)

    def calc_force(self, x: float, y: float, theta: float, G: float, eps2: float) -> Tuple[float, float]:
        # Barnes–Hut criterion: use cell COM when (cell size / distance) < θ
        # Softened Newtonian force: a = G M r / (|r|^2 + ε^2)^{3/2}
        if self.mass == 0.0:
            return 0.0, 0.0

        dx = self.com_x - x
        dy = self.com_y - y
        dist2 = dx * dx + dy * dy + eps2
        dist = dist2 ** 0.5

        size = self.quad.half * 2.0
        if self.is_external() or size / dist < theta:
            inv_r3 = 1.0 / (dist2 * dist)
            s = G * self.mass * inv_r3
            return dx * s, dy * s

        ax, ay = 0.0, 0.0
        if self.sw is not None:
            sx, sy = self.sw.calc_force(x, y, theta, G, eps2)
            ax += sx; ay += sy
        if self.se is not None:
            sx, sy = self.se.calc_force(x, y, theta, G, eps2)
            ax += sx; ay += sy
        if self.nw is not None:
            sx, sy = self.nw.calc_force(x, y, theta, G, eps2)
            ax += sx; ay += sy
        if self.ne is not None:
            sx, sy = self.ne.calc_force(x, y, theta, G, eps2)
            ax += sx; ay += sy
        return ax, ay


class BarnesHut:
    def __init__(self, cx: float, cy: float, half: float) -> None:
        self.root = Node(Quad(cx, cy, half))

    def insert_all(self, bodies: List[Body]) -> None:
        self.root = Node(self.root.quad)
        for b in bodies:
            if self.root.quad.contains(b.x, b.y):
                self.root.insert(b)

    def acceleration(self, x: float, y: float, theta: float, G: float, eps: float) -> Tuple[float, float]:
        return self.root.calc_force(x, y, theta, G, eps * eps)


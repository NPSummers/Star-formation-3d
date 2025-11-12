from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import secrets

from app.spatial.quadtree import BarnesHut, Body
from app.config import settings as C


@dataclass
class Gas:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    u: np.ndarray
    rho: np.ndarray


@dataclass
class Sink:
    x: float
    y: float
    vx: float
    vy: float
    mass: float


class UniformGrid:
    def __init__(self, box_half: float, h: float) -> None:
        self.box_half = box_half
        self.h = h
        self.cell_size = h
        self.n = max(2, int(math.ceil(2 * box_half / self.cell_size)))
        self.inv_cell = 1.0 / self.cell_size
        self.cells: List[List[int]] = [[] for _ in range(self.n * self.n)]

    def _index(self, x: float, y: float) -> Tuple[int, int, int]:
        i = int((x + self.box_half) * self.inv_cell)
        j = int((y + self.box_half) * self.inv_cell)
        i = min(max(i, 0), self.n - 1)
        j = min(max(j, 0), self.n - 1)
        return i, j, j * self.n + i

    def build(self, pos: np.ndarray) -> None:
        for bucket in self.cells:
            bucket.clear()
        for idx, (x, y) in enumerate(pos):
            _, __, k = self._index(float(x), float(y))
            self.cells[k].append(idx)

    def neighbors(self, x: float, y: float) -> List[int]:
        i, j, _ = self._index(x, y)
        result: List[int] = []
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                ii = min(max(i + di, 0), self.n - 1)
                jj = min(max(j + dj, 0), self.n - 1)
                result.extend(self.cells[jj * self.n + ii])
        return result


def cubic_spline_kernel(r: float, h: float) -> float:
    # 2D cubic spline kernel (Monaghan):
    # W(q) = σ [1 - 1.5 q^2 + 0.75 q^3] for 0 ≤ q < 1; W(q) = σ 0.25 (2 - q)^3 for 1 ≤ q < 2; 0 otherwise
    # with σ = 10 / (7 π h^2) in 2D
    q = r / h
    sigma = 10.0 / (7.0 * math.pi * h * h)
    if q < 1.0:
        return sigma * (1 - 1.5 * q * q + 0.75 * q * q * q)
    if q < 2.0:
        t = 2.0 - q
        return sigma * 0.25 * t * t * t
    return 0.0


def cubic_spline_gradient(dx: float, dy: float, h: float) -> Tuple[float, float]:
    # ∇W for the 2D cubic spline; returns components (∂W/∂x, ∂W/∂y)
    # Uses radial derivative dW/dr and projects along (dx, dy)
    r = math.hypot(dx, dy)
    if r == 0.0:
        return 0.0, 0.0
    q = r / h
    sigma = 10.0 / (7.0 * math.pi * h * h)
    factor = 0.0
    if q < 1.0:
        factor = sigma * (-3.0 * q + 2.25 * q * q) / h
    elif q < 2.0:
        t = 2.0 - q
        factor = -sigma * 0.75 * t * t / h
    s = factor / r
    return dx * s, dy * s


class Simulation:
    def __init__(self, seed: int | None = None) -> None:
        # Choose a random 32-bit seed if none provided; clamp to NumPy range
        if seed is None:
            seed = secrets.randbits(32)
        self.seed = int(seed) & 0xFFFFFFFF
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.N = C.NUM_GAS_PARTICLES
        self.mass = C.GAS_PARTICLE_MASS

        self.gas = Gas(
            pos=self._init_positions(self.N),
            vel=self._init_velocities(self.N),
            acc=np.zeros((self.N, 2), dtype=np.float64),
            u=np.full(self.N, 0.5, dtype=np.float64),
            rho=np.full(self.N, 1.0, dtype=np.float64),
        )

        self.sinks: List[Sink] = []

        self.h = C.SPH_SMOOTHING_LENGTH
        self.grid = UniformGrid(C.BOX_SIZE, self.h)
        self.theta = C.BARNES_HUT_THETA
        self.G = C.GRAVITATIONAL_CONSTANT
        self.soft = C.SOFTENING_LENGTH

        self.time = 0.0
        self.dt = C.DT

    def _init_positions(self, n: int) -> np.ndarray:
        pts = []
        centers = [
            (random.uniform(-0.6, 0.6), random.uniform(-0.6, 0.6))
            for _ in range(C.INIT_DENSITY_CLUMPS)
        ]
        weights = np.array([random.random() + 0.2 for _ in centers], dtype=np.float64)
        weights /= weights.sum()
        std = 0.12
        for _ in range(n):
            cx, cy = random.choices(centers, weights=weights)[0]
            x = random.gauss(cx, std)
            y = random.gauss(cy, std)
            x = max(-C.BOX_SIZE * 0.95, min(C.BOX_SIZE * 0.95, x))
            y = max(-C.BOX_SIZE * 0.95, min(C.BOX_SIZE * 0.95, y))
            pts.append((x, y))
        return np.array(pts, dtype=np.float64)

    def _init_velocities(self, n: int) -> np.ndarray:
        amp = C.INIT_VELOCITY_AMPLITUDE
        if amp <= 0.0:
            return np.zeros((n, 2), dtype=np.float64)
        v = np.random.normal(0.0, amp, size=(n, 2))
        v -= v.mean(axis=0, keepdims=True)
        return v

    def step(self) -> None:
        self.grid.build(self.gas.pos)
        self._compute_density()
        a_hydro = self._compute_hydro_acceleration()
        a_grav = self._compute_gravity()
        self.gas.acc[:] = a_hydro + a_grav
        self._update_internal_energy()
        self._integrate()
        self._handle_boundaries()
        self._star_formation_and_accretion()
        self._update_timestep()
        self.time += self.dt

    def _compute_density(self) -> None:
        rho = self.gas.rho
        pos = self.gas.pos
        h = self.h
        for i in range(self.N):
            x, y = float(pos[i, 0]), float(pos[i, 1])
            neighbors = self.grid.neighbors(x, y)
            acc_rho = 0.0
            for j in neighbors:
                dx = x - float(pos[j, 0])
                dy = y - float(pos[j, 1])
                r = math.hypot(dx, dy)
                acc_rho += self.mass * cubic_spline_kernel(r, h)
            rho[i] = max(1e-6, acc_rho)

    def _compute_pressure(self) -> np.ndarray:
        # Ideal gas law: P = (γ - 1) ρ u, scaled by PRESSURE_COEFFICIENT
        return (C.GAMMA - 1.0) * self.gas.rho * self.gas.u * C.PRESSURE_COEFFICIENT

    def _compute_hydro_acceleration(self) -> np.ndarray:
        pos = self.gas.pos
        vel = self.gas.vel
        rho = self.gas.rho
        P = self._compute_pressure()
        h = self.h
        N = self.N

        a = np.zeros((N, 2), dtype=np.float64)
        for i in range(N):
            xi, yi = float(pos[i, 0]), float(pos[i, 1])
            vi_x, vi_y = float(vel[i, 0]), float(vel[i, 1])
            Pi, rhoi = float(P[i]), float(rho[i])
            neighbors = self.grid.neighbors(xi, yi)
            ax, ay = 0.0, 0.0
            for j in neighbors:
                if j == i:
                    continue
                dx = xi - float(pos[j, 0])
                dy = yi - float(pos[j, 1])
                dvx = vi_x - float(vel[j, 0])
                dvy = vi_y - float(vel[j, 1])
                gradWx, gradWy = cubic_spline_gradient(dx, dy, h)
                Pj, rhoj = float(P[j]), float(rho[j])
                # Symmetric SPH pressure term: a_i ≈ - Σ m_j (P_i/ρ_i^2 + P_j/ρ_j^2) ∇W_ij
                press = -self.mass * (Pi / (rhoi * rhoi) + Pj / (rhoj * rhoj))
                ax += press * gradWx
                ay += press * gradWy
                # Monaghan artificial viscosity: Π_ij ∝ (-ν μ)/(r^2 + ε h^2) (c_i + c_j)/2 for μ < 0
                mu = dx * dvx + dy * dvy
                if mu < 0.0:
                    r2 = dx * dx + dy * dy + 0.01 * h * h
                    nu = C.ARTIFICIAL_VISCOSITY
                    c_i = math.sqrt(max(1e-8, C.GAMMA * Pi / rhoi))
                    c_j = math.sqrt(max(1e-8, C.GAMMA * Pj / rhoj))
                    pi_ij = (-nu * mu) / r2 * (c_i + c_j) * 0.5
                    # a_visc ≈ - m_j Π_ij / (ρ_i ρ_j) ∇W_ij
                    visc = -self.mass * pi_ij * (1.0 / (rhoi * rhoj))
                    ax += visc * gradWx
                    ay += visc * gradWy

            a[i, 0] = ax
            a[i, 1] = ay
        return a

    def _compute_gravity(self) -> np.ndarray:
        # Gas and sinks as point masses for Barnes–Hut approximation
        bodies: List[Body] = []
        for i in range(self.N):
            bodies.append(Body(float(self.gas.pos[i, 0]), float(self.gas.pos[i, 1]), self.mass))
        for s in self.sinks:
            bodies.append(Body(s.x, s.y, s.mass))

        tree = BarnesHut(0.0, 0.0, C.BOX_SIZE)
        tree.insert_all(bodies)

        a = np.zeros((self.N, 2), dtype=np.float64)
        for i in range(self.N):
            xi = float(self.gas.pos[i, 0])
            yi = float(self.gas.pos[i, 1])
            # Softened Newtonian acceleration: a = G M r / (|r|^2 + ε^2)^{3/2}, aggregated via BH
            ax, ay = tree.acceleration(xi, yi, self.theta, self.G, self.soft)
            a[i, 0] = ax
            a[i, 1] = ay
        return a

    def _update_internal_energy(self) -> None:
        pos = self.gas.pos
        vel = self.gas.vel
        rho = self.gas.rho
        u = self.gas.u
        h = self.h

        du = np.zeros_like(u)
        for i in range(self.N):
            xi, yi = float(pos[i, 0]), float(pos[i, 1])
            vi_x, vi_y = float(vel[i, 0]), float(vel[i, 1])
            neighbors = self.grid.neighbors(xi, yi)
            div_v = 0.0
            for j in neighbors:
                if j == i:
                    continue
                dx = xi - float(pos[j, 0])
                dy = yi - float(pos[j, 1])
                dvx = vi_x - float(vel[j, 0])
                dvy = vi_y - float(vel[j, 1])
                gradWx, gradWy = cubic_spline_gradient(dx, dy, h)
                div_v += self.mass * (dvx * gradWx + dvy * gradWy) / float(rho[j])
            # PdV heating/cooling: du/dt = - (P/ρ) ∇·v
            Pi = (C.GAMMA - 1.0) * float(rho[i]) * float(u[i]) * C.PRESSURE_COEFFICIENT
            du[i] = -Pi / max(1e-8, float(rho[i])) * div_v
        # Linear cooling toward floor temperature: du/dt = -λ (u - u_min)
        cooling = -C.COOLING_RATE * (u - C.MIN_TEMPERATURE)
        self.gas.u += self.dt * (du + cooling)
        np.maximum(self.gas.u, C.MIN_TEMPERATURE, out=self.gas.u)

    def _integrate(self) -> None:
        # Leapfrog KDK: v^{n+1/2} = v^n + (dt/2) a^n; x^{n+1} = x^n + dt v^{n+1/2}; then second kick
        dt = self.dt
        self.gas.vel += 0.5 * dt * self.gas.acc
        self.gas.pos += dt * self.gas.vel

        a_grav = self._compute_gravity()
        self.gas.acc[:] = self.gas.acc * 0.0 + a_grav + self._compute_hydro_acceleration()
        self.gas.vel += 0.5 * dt * self.gas.acc

        if self.sinks:
            sink_bodies = [Body(s.x, s.y, s.mass) for s in self.sinks]
            all_bodies = [Body(float(self.gas.pos[i, 0]), float(self.gas.pos[i, 1]), self.mass) for i in range(self.N)] + sink_bodies
            tree = BarnesHut(0.0, 0.0, C.BOX_SIZE)
            tree.insert_all(all_bodies)
            for s in self.sinks:
                ax, ay = tree.acceleration(s.x, s.y, self.theta, self.G, self.soft)
                s.vx += dt * ax
                s.vy += dt * ay
                s.x += dt * s.vx
                s.y += dt * s.vy

    def _handle_boundaries(self) -> None:
        return

    def _star_formation_and_accretion(self) -> None:
        created: List[int] = []
        for i in range(self.N):
            if self.gas.rho[i] < C.DENSITY_THRESHOLD:
                continue
            if not self._is_bound_region(i):
                continue
            x, y = float(self.gas.pos[i, 0]), float(self.gas.pos[i, 1])
            vx, vy = float(self.gas.vel[i, 0]), float(self.gas.vel[i, 1])
            m = self.mass
            self.sinks.append(Sink(x=x, y=y, vx=vx, vy=vy, mass=m))
            created.append(i)

        if created:
            mask = np.ones(self.N, dtype=bool)
            mask[created] = False
            self.gas.pos = self.gas.pos[mask]
            self.gas.vel = self.gas.vel[mask]
            self.gas.acc = self.gas.acc[mask]
            self.gas.u = self.gas.u[mask]
            self.gas.rho = self.gas.rho[mask]
            self.N = int(self.gas.pos.shape[0])

        if not self.sinks or self.N == 0:
            return
        to_remove: List[int] = []
        for i in range(self.N):
            xi, yi = float(self.gas.pos[i, 0]), float(self.gas.pos[i, 1])
            for s in self.sinks:
                dx = xi - s.x
                dy = yi - s.y
                r2 = dx * dx + dy * dy
                if r2 < C.SINK_ACCRETION_RADIUS * C.SINK_ACCRETION_RADIUS:
                    s.vx = (s.vx * s.mass + self.gas.vel[i, 0] * self.mass) / (s.mass + self.mass)
                    s.vy = (s.vy * s.mass + self.gas.vel[i, 1] * self.mass) / (s.mass + self.mass)
                    s.mass += self.mass
                    to_remove.append(i)
                    break
        if to_remove:
            mask = np.ones(self.N, dtype=bool)
            mask[to_remove] = False
            self.gas.pos = self.gas.pos[mask]
            self.gas.vel = self.gas.vel[mask]
            self.gas.acc = self.gas.acc[mask]
            self.gas.u = self.gas.u[mask]
            self.gas.rho = self.gas.rho[mask]
            self.N = int(self.gas.pos.shape[0])

        merged: List[int] = []
        for a in range(len(self.sinks)):
            if a in merged:
                continue
            for b in range(a + 1, len(self.sinks)):
                if b in merged:
                    continue
                sa, sb = self.sinks[a], self.sinks[b]
                dx = sa.x - sb.x
                dy = sa.y - sb.y
                if dx * dx + dy * dy < C.SINK_MERGE_RADIUS * C.SINK_MERGE_RADIUS:
                    mtot = sa.mass + sb.mass
                    vx = (sa.vx * sa.mass + sb.vx * sb.mass) / mtot
                    vy = (sa.vy * sa.mass + sb.vy * sb.mass) / mtot
                    x = (sa.x * sa.mass + sb.x * sb.mass) / mtot
                    y = (sa.y * sa.mass + sb.y * sb.mass) / mtot
                    self.sinks[a] = Sink(x=x, y=y, vx=vx, vy=vy, mass=mtot)
                    merged.append(b)
        if merged:
            self.sinks = [s for idx, s in enumerate(self.sinks) if idx not in merged]

    def _is_bound_region(self, i: int) -> bool:
        # Bound criterion: E_kin + α E_th < |E_grav|
        xi, yi = float(self.gas.pos[i, 0]), float(self.gas.pos[i, 1])
        vi_x, vi_y = float(self.gas.vel[i, 0]), float(self.gas.vel[i, 1])
        neighbors = self.grid.neighbors(xi, yi)

        m_sum = 0.0
        x_cm = 0.0
        y_cm = 0.0
        vx_cm = 0.0
        vy_cm = 0.0
        for j in neighbors:
            m_sum += self.mass
            x_cm += self.mass * float(self.gas.pos[j, 0])
            y_cm += self.mass * float(self.gas.pos[j, 1])
            vx_cm += self.mass * float(self.gas.vel[j, 0])
            vy_cm += self.mass * float(self.gas.vel[j, 1])
        if m_sum <= 0.0:
            return False
        x_cm /= m_sum; y_cm /= m_sum
        vx_cm /= m_sum; vy_cm /= m_sum

        E_kin = 0.0
        E_th = 0.0
        E_grav = 0.0
        for j in neighbors:
            dvx = float(self.gas.vel[j, 0]) - vx_cm
            dvy = float(self.gas.vel[j, 1]) - vy_cm
            # E_kin = 1/2 m |v - v_cm|^2; E_th ∝ m u (proxy for thermal energy)
            E_kin += 0.5 * self.mass * (dvx * dvx + dvy * dvy)
            E_th += self.mass * float(self.gas.u[j])
            dx = float(self.gas.pos[j, 0]) - x_cm
            dy = float(self.gas.pos[j, 1]) - y_cm
            r = math.hypot(dx, dy) + 1e-3
            # Approximate potential: E_grav ≈ - G m m_tot / r
            E_grav -= self.G * self.mass * m_sum / r
        return (E_kin + C.BOUNDNESS_ALPHA * E_th) < abs(E_grav)

    def _update_timestep(self) -> None:
        # Signal speed ≈ c_s + |v| with c_s^2 = γ(γ-1) u
        c_s = np.sqrt(np.maximum(1e-8, C.GAMMA * (C.GAMMA - 1.0) * self.gas.u))
        vmag = np.linalg.norm(self.gas.vel, axis=1)
        signal = np.max(c_s + vmag)
        # CFL: dt ≤ C h / signal
        dt_cfl = C.CFL_FACTOR * self.h / max(1e-6, signal)

        amax = float(np.max(np.linalg.norm(self.gas.acc, axis=1))) if self.N > 0 else 0.0
        # Acceleration limiter: dt ∝ sqrt(h / |a|)
        dt_acc = 0.25 * math.sqrt(self.h / max(1e-8, amax)) if amax > 0 else C.MAX_DT

        self.dt = max(C.MIN_DT, min(C.MAX_DT, min(dt_cfl, dt_acc)))


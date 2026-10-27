"""Constant-mass, flat-earth 6-DOF mechanics, without a fitted aircraft model.

State: [N,E,D,u,v,w,q0,q1,q2,q3,p,q_body,r].
Quaternion: Hamilton, scalar first, body -> NED. All moments about CG.
Forces passed to rhs EXCLUDE gravity. SI units throughout.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _array(value, shape, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


@dataclass(frozen=True)
class RigidBody:
    """Actual mass and inertia must be supplied; no DIAM-20K defaults."""

    mass_kg: float
    inertia_kg_m2: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError("mass_kg must be finite and positive")
        inertia = _array(self.inertia_kg_m2, (3, 3), "inertia")
        if not np.allclose(inertia, inertia.T, rtol=0, atol=1e-12):
            raise ValueError("inertia must be symmetric")
        principal = np.linalg.eigvalsh(inertia)
        if principal[0] <= 0:
            raise ValueError("inertia must be positive definite")
        if principal[2] > principal[0] + principal[1] + 1e-12 * principal[2]:
            raise ValueError("principal inertias must satisfy the triangle inequality")
        inertia.setflags(write=False)
        object.__setattr__(self, "inertia_kg_m2", inertia)


def _quaternion(value):
    q = _array(value, (4,), "quaternion")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion norm must be nonzero and finite")
    return q / norm


def rotation(quaternion):
    """Return an orthogonal body -> NED matrix; normalize input quaternion."""
    s, x, y, z = _quaternion(quaternion)
    v = np.array([x, y, z])
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (s*s - v @ v) * np.eye(3) + 2*np.outer(v, v) + 2*s*skew


def rhs(state, body, force_body_n, moment_body_nm, *, gravity_m_s2=9.80665):
    """State derivative under total non-gravitational force and CG moment.

    Loads are actual forces/moments, not motor commands. No aerodynamics,
    ground contact, fuel consumption or rotor angular momentum is inferred.
    """
    x = _array(state, (13,), "state")
    force = _array(force_body_n, (3,), "force")
    moment = _array(moment_body_nm, (3,), "moment")
    if not np.isfinite(gravity_m_s2) or gravity_m_s2 < 0:
        raise ValueError("gravity must be finite and nonnegative")
    velocity, omega = x[3:6], x[10:13]
    q = _quaternion(x[6:10])
    rot = rotation(q)
    result = np.zeros(13)
    result[:3] = rot @ velocity
    result[3:6] = (force / body.mass_kg + rot.T @ [0, 0, gravity_m_s2]
                   - np.cross(omega, velocity))
    result[6] = -0.5 * (q[1:] @ omega)
    result[7:10] = 0.5 * (q[0]*omega + np.cross(q[1:], omega))
    inertia = body.inertia_kg_m2
    result[10:13] = np.linalg.solve(inertia, moment - np.cross(omega, inertia @ omega))
    return result


def point_load(position_body_m, force_body_n, reaction_moment_body_nm):
    """Force and CG moment from an application point and a free couple.

    Position is measured FROM CG. Reaction moment is explicitly supplied,
    so rotor rotation signs and thrust directions are never guessed.
    """
    position = _array(position_body_m, (3,), "position")
    force = _array(force_body_n, (3,), "force")
    couple = _array(reaction_moment_body_nm, (3,), "reaction moment")
    return force, np.cross(position, force) + couple


def air_data(state, wind_ned_m_s, density_kg_m3):
    """Air-relative velocity, speed, dynamic pressure and angles.

    Wind points in the direction the air moves. At zero speed alpha/beta
    are None; these angles must not silently be used in an aero model.
    """
    x = _array(state, (13,), "state")
    wind = _array(wind_ned_m_s, (3,), "wind")
    if not np.isfinite(density_kg_m3) or density_kg_m3 <= 0:
        raise ValueError("density must be finite and positive")
    relative = x[3:6] - rotation(x[6:10]).T @ wind
    speed = float(np.linalg.norm(relative))
    return {
        "velocity_body_m_s": relative,
        "speed_m_s": speed,
        "dynamic_pressure_pa": 0.5 * density_kg_m3 * speed**2,
        "alpha_rad": float(np.arctan2(relative[2], relative[0])) if speed > 0 else None,
        "beta_rad": float(np.arcsin(np.clip(relative[1]/speed, -1, 1))) if speed > 0 else None,
    }


def step_rk4(state, time_s, dt_s, body, loads: Callable, *, gravity_m_s2=9.80665):
    """One RK4 step. loads(t, state) -> (force_B, moment_B), excluding gravity.

    The callback is evaluated at each intermediate state and receives a copy
    with a unit quaternion. Quaternion normalization also follows each step.
    """
    if not np.isfinite(time_s) or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("time must be finite; dt must be finite and positive")
    x = _array(state, (13,), "state")
    x[6:10] = _quaternion(x[6:10])

    def derivative(t, value):
        normalized = value.copy()
        normalized[6:10] = _quaternion(normalized[6:10])
        force, moment = loads(t, normalized.copy())
        return rhs(normalized, body, force, moment, gravity_m_s2=gravity_m_s2)

    k1 = derivative(time_s, x)
    k2 = derivative(time_s + dt_s/2, x + dt_s*k1/2)
    k3 = derivative(time_s + dt_s/2, x + dt_s*k2/2)
    k4 = derivative(time_s + dt_s, x + dt_s*k3)
    result = x + dt_s*(k1 + 2*k2 + 2*k3 + k4)/6
    result[6:10] = _quaternion(result[6:10])
    return _array(result, (13,), "integrated state")

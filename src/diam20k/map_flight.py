"""Generate a standalone browser replay of a simulated flight over DEMO_MAP."""

import argparse
import json
from dataclasses import asdict
from importlib.resources import files
from math import cos, pi, sin
from pathlib import Path

import numpy as np

from .dynamics import RigidBody, rotation, step_rk4
from .geography import MapOrigin
from .obstacle_map import DEMO_GOAL_NED, DEMO_MAP, DEMO_START_NED

DEMO_ORIGIN = MapOrigin(55.75, 37.62, 150.0)
GRAVITY_M_S2 = 9.80665


def simulate_demo(origin: MapOrigin = DEMO_ORIGIN) -> dict:
    """Integrate a scripted force and moment profile with a collision check each step."""
    body = RigidBody(2.0, np.diag([2., 3., 4.]))
    dt = 0.05
    duration = 40.0
    frequency = 2 * pi / duration
    climb_amplitude = 30.0
    pitch_amplitude = np.deg2rad(12.0)
    state = np.zeros(13)
    state[:3] = DEMO_START_NED
    state[3] = (DEMO_GOAL_NED[0] - DEMO_START_NED[0]) / duration
    state[6] = 1
    state[11] = pitch_amplitude * frequency

    def loads(time_s, current):
        phase = frequency * time_s
        down_acceleration = -climb_amplitude * frequency**2 * cos(phase)
        force_ned = body.mass_kg * np.array([0, 0, down_acceleration - GRAVITY_M_S2])
        pitch_acceleration = -pitch_amplitude * frequency**2 * sin(phase)
        moment_body = np.array([0, body.inertia_kg_m2[1, 1] * pitch_acceleration, 0])
        return rotation(current[6:10]).T @ force_ned, moment_body

    samples = []
    status = "Маршрут завершён"
    if not DEMO_MAP.is_position_free(state[:3]):
        raise ValueError("Начальная точка занята или вне карты")
    for step in range(round(duration / dt) + 1):
        time_s = step * dt
        attitude = rotation(state[6:10])
        samples.append({
            "time": time_s,
            "ned": state[:3].tolist(),
            "geo": origin.to_geodetic(state[:3]),
            "attitude": attitude.tolist(),
            "velocity": (attitude @ state[3:6]).tolist(),
        })
        if step * dt >= duration:
            break
        next_state = step_rk4(state, time_s, dt, body, loads, gravity_m_s2=GRAVITY_M_S2)
        if not DEMO_MAP.is_segment_free(state[:3], next_state[:3]):
            status = "Полёт прерван: препятствие или граница карты"
            break
        state = next_state

    return {**asdict(DEMO_MAP), "goal": DEMO_GOAL_NED, "samples": samples, "status": status}


def main():
    parser = argparse.ArgumentParser(description="Создать HTML с воспроизведением полёта 6-DOF.")
    parser.add_argument("--origin", type=float, nargs=3, metavar=("LAT", "LON", "H"),
                        default=(DEMO_ORIGIN.latitude_deg, DEMO_ORIGIN.longitude_deg,
                                 DEMO_ORIGIN.ellipsoid_height_m),
                        help="широта, долгота и высота уровня карты над эллипсоидом WGS 84")
    parser.add_argument("--output", type=Path, default=Path("map-flight.html"))
    args = parser.parse_args()
    try:
        replay = simulate_demo(MapOrigin(*args.origin))
        template = files("diam20k").joinpath("map_flight.html").read_text(encoding="utf-8")
        with args.output.open("x", encoding="utf-8") as output:
            output.write(template.replace("/* FLIGHT_DATA */", json.dumps(replay, allow_nan=False)))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(args.output.resolve())


if __name__ == "__main__":
    main()

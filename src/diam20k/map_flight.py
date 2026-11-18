"""Generate a standalone browser replay of a simulated flight over DEMO_MAP."""

import argparse
import json
from dataclasses import asdict
from importlib.resources import files
from math import ceil, pi, sin
from pathlib import Path

import numpy as np

from .course import plan_course
from .dynamics import RigidBody, rotation, step_rk4
from .geography import MapOrigin
from .obstacle_map import DEMO_GOAL_NED, DEMO_MAP, DEMO_START_NED

DEMO_ORIGIN = MapOrigin(55.75, 37.62, 150.0)
GRAVITY_M_S2 = 9.80665


def crosswind(time_s: float) -> float:
    """Восточная компонента ветра, м/с. Плавная смена направления за 40 с."""
    if 20.0 < time_s < 60.0:
        return 4.0 * sin(2 * pi * (time_s - 20.0) / 40.0)
    return 0.0


def flight_summary(samples, world, blockers):
    positions = np.array([sample["ned"] for sample in samples])
    closest_m = None
    for box in world.obstacles:
        # Внутри интервала расстояние до коробки по соответствующей оси равно нулю.
        offsets = np.maximum(box.minimum_ned - positions, positions - box.maximum_ned)
        distance = float(np.linalg.norm(np.maximum(offsets, 0), axis=1).min())
        if closest_m is None or distance < closest_m:
            closest_m = distance
    return {
        "duration_s": samples[-1]["time"] - samples[0]["time"],
        "distance_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        "passed": sum(positions[-1, 0].item() > world.obstacles[i].maximum_ned[0]
                      for i in blockers),
        "total": len(blockers),
        "closest_sample_m": closest_m,
    }


def simulate_demo(origin: MapOrigin = DEMO_ORIGIN, *, wind: bool = True) -> dict:
    """Рассчитывает полёт по проходам карты при заданных внешних нагрузках."""
    body = RigidBody(2.0, np.diag([2., 3., 4.]))
    dt = 0.05
    speed = 20.0
    course = plan_course(DEMO_MAP, DEMO_START_NED, DEMO_GOAL_NED)
    collision_map = DEMO_MAP.with_margin(3.0)
    duration = (DEMO_GOAL_NED[0] - DEMO_START_NED[0]) / speed
    state = np.zeros(13)
    state[:3] = DEMO_START_NED
    state[3] = speed
    state[6] = 1

    def loads(time_s, current):
        north = DEMO_START_NED[0] + speed * time_s
        _, velocity, acceleration = course.motion(north, speed)
        attitude = rotation(current[6:10])
        ground_velocity = attitude @ current[3:6]
        wind_speed = crosswind(time_s) if wind else 0.0
        # Боковое сопротивление пропорционально скорости относительно воздуха.
        drag = (wind_speed - ground_velocity[1]) / 2.0
        east_accel = acceleration[0] + velocity[0] / 2.0 + drag
        force_ned = body.mass_kg * np.array([0, east_accel, acceleration[1] - GRAVITY_M_S2])

        # Учебный закон момента разворачивает продольную ось вдоль скорости.
        direction_body = current[3:6] / np.linalg.norm(ground_velocity)
        angular_error = np.array([0, -direction_body[2], direction_body[1]])
        omega = current[10:13]
        inertia = body.inertia_kg_m2
        moment = inertia @ (4 * angular_error - 4 * omega) + np.cross(omega, inertia @ omega)
        return attitude.T @ force_ned, moment

    samples = []
    status = "Время расчёта истекло, цель не достигнута"
    for step in range(ceil(duration / dt) + 1):
        time_s = min(step * dt, duration)
        attitude = rotation(state[6:10])
        samples.append({
            "time": time_s,
            "wind_east": crosswind(time_s) if wind else 0.0,
            "ned": state[:3].tolist(),
            "geo": origin.to_geodetic(state[:3]),
            "attitude": attitude.tolist(),
            "velocity": (attitude @ state[3:6]).tolist(),
        })
        if time_s >= duration:
            if np.linalg.norm(state[:3] - DEMO_GOAL_NED) < 1.0:
                status = "Маршрут завершён"
            break
        next_state = step_rk4(state, time_s, min(dt, duration - time_s), body, loads, gravity_m_s2=GRAVITY_M_S2)
        if not collision_map.is_segment_free(state[:3], next_state[:3]):
            status = "Полёт прерван: препятствие или граница карты"
            break
        state = next_state

    blockers = sorted(
        (i for i, box in enumerate(DEMO_MAP.obstacles)
         if box.intersects_segment(DEMO_START_NED, DEMO_GOAL_NED)),
        key=lambda i: DEMO_MAP.obstacles[i].minimum_ned[0],
    )
    return {
        **asdict(DEMO_MAP), "blockers": blockers, "goal": DEMO_GOAL_NED,
        "samples": samples, "status": status,
        "summary": flight_summary(samples, DEMO_MAP, blockers),
    }


def main():
    parser = argparse.ArgumentParser(description="Создать HTML с воспроизведением полёта 6-DOF.")
    parser.add_argument("--origin", type=float, nargs=3, metavar=("LAT", "LON", "H"),
                        default=(DEMO_ORIGIN.latitude_deg, DEMO_ORIGIN.longitude_deg,
                                 DEMO_ORIGIN.ellipsoid_height_m),
                        help="широта, долгота и высота уровня карты над эллипсоидом WGS 84")
    parser.add_argument("--output", type=Path, default=Path("map-flight.html"))
    parser.add_argument("--calm", action="store_true", help="воспроизвести маршрут в штиль")
    args = parser.parse_args()
    try:
        replay = simulate_demo(MapOrigin(*args.origin), wind=not args.calm)
        template = files("diam20k").joinpath("map_flight.html").read_text(encoding="utf-8")
        with args.output.open("x", encoding="utf-8") as output:
            output.write(template.replace("/* FLIGHT_DATA */", json.dumps(replay, allow_nan=False)))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(args.output.resolve())


if __name__ == "__main__":
    main()

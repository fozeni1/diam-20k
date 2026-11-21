"""Расчёт полёта над полигоном и запись телеметрии для HTML-отчёта."""

import argparse
import json
from dataclasses import asdict, replace
from importlib.resources import files
from math import pi, sin
from pathlib import Path

import numpy as np

from .course import plan_course
from .dynamics import rotation
from .instruments import battery_energy, compare_recording, downward_range, fuel_usage, readings
from .inertial_lab import audit_replay
from .flight_audit import mechanical_sample, work_balance, fuel_inventory, obstacle_passes, lab_drive
from .parameters import load_body
from .scenarios import GRAVITY_M_S2, SCENARIOS, TEST_BODY, Scenario, flight_states
from .relief import terrain_layer, utah_surface, lab_surface
from .geography import MapOrigin
from .obstacle_map import DEMO_GOAL_NED, DEMO_MAP, DEMO_START_NED

DEMO_ORIGIN = MapOrigin(37.45, -110.61, 1300.0)

# Периоды в тактах по 5 мс для режима записи приборных каналов.
SENSOR_STRIDES = {"gyro_rad_s": 1, "accel_m_s2": 1, "mag_ut": 4, "baro_pa": 8,
                  "range_down_m": 4, "dynamic_pressure_pa": 4, "gnss_deg_deg_m": 40, "gnss_velocity_ned_m_s": 40}


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


def route_scenario(world, body, wind_at):
    dt = 0.05
    speed = 20.0
    course = plan_course(world, DEMO_START_NED, DEMO_GOAL_NED)
    duration = (DEMO_GOAL_NED[0] - DEMO_START_NED[0]) / speed
    state = np.zeros(13)
    state[:3] = DEMO_START_NED
    state[3] = speed
    state[6] = 1

    def loads(time_s, current):
        north = current[0]
        _, velocity, acceleration = course.motion(north, speed)
        attitude = rotation(current[6:10])
        ground_velocity = attitude @ current[3:6]
        wind_speed = wind_at(time_s)[1]
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

    return Scenario("route", "Маршрут с заданными результирующими нагрузками",
                    body, state, loads, duration, dt)


def simulate_demo(origin: MapOrigin = DEMO_ORIGIN, *, wind=True, name="route",
                  world=None, body=TEST_BODY, fuel_log=None, engine_log=None, field_ned_ut=None,
                  instrument_rates=False, hull_m=None, inertial_check=False, fuel_rate_l_h=None, battery_log=None,
                  surface="flat", gnss_outage=None, initial_fuel_l=None, drive_profile=None):
    if drive_profile is not None and (name != "route" or
            any(v is not None for v in (fuel_rate_l_h, fuel_log, engine_log))):
        raise ValueError("Профиль учебного привода применяется к маршруту как отдельный источник расхода")
    if fuel_rate_l_h is not None:
        if not np.isfinite(fuel_rate_l_h) or fuel_rate_l_h < 0:
            raise ValueError("Расход должен быть конечным и неотрицательным")
        if fuel_log is not None or engine_log is not None:
            raise ValueError("Выберите журнал или оценку по постоянному расходу")
    world = DEMO_MAP if world is None else world
    if surface not in ("flat", "lab", "utah"):
        raise ValueError("Неизвестная поверхность полигона")
    ground, mesh = utah_surface(origin, world.bounds) if surface == "utah" else (None, None)
    if surface == "lab":
        ground, mesh = lab_surface(world.bounds)
    if gnss_outage is not None:
        start, end = gnss_outage
        if not np.isfinite([start, end]).all() or not 0 <= start < end:
            raise ValueError("Нужен возрастающий интервал потери GNSS в секундах")
        inertial_check = True
    if field_ned_ut is not None:
        field_ned_ut = np.asarray(field_ned_ut, dtype=float)
        if field_ned_ut.shape != (3,) or not np.all(np.isfinite(field_ned_ut)):
            raise ValueError("Магнитное поле должно иметь три конечные компоненты NED")

    def wind_at(time_s):
        east = 0.0
        if wind and name == "route":
            east = crosswind(time_s)
        elif wind and name == "wind":
            east = 5.0
        return (0.0, east, 0.0)

    if name == "route":
        scenario = route_scenario(world, body, wind_at)
    elif name == "free-fall":
        # Исходный тест начинается на 1000 м, выше верхней границы полигона.
        scenario = SCENARIOS[name](body, altitude_m=120, duration_s=4)
    elif name == "wind":
        scenario = SCENARIOS[name](body, wind_ned_m_s=wind_at(0))
    else:
        scenario = SCENARIOS[name](body)
    if gnss_outage is not None and gnss_outage[1] > scenario.duration_s:
        raise ValueError("Интервал потери GNSS выходит за время сценария")
    collision_map = world.with_margin(3.0)
    previous = scenario.initial_state[:3]
    if not collision_map.is_position_free(previous):
        raise ValueError("Начальное положение вне свободного объёма полигона")

    previous_rotation = rotation(scenario.initial_state[6:10])
    if hull_m is not None:
        half = np.asarray(hull_m, dtype=float) / 2
        if half.shape != (3,) or not np.all(np.isfinite(half)) or np.any(half <= 0):
            raise ValueError("Нужны три положительных габарита в метрах")
        radius = np.linalg.norm(half)
        if not collision_map.envelope_free(previous, previous, abs(previous_rotation) @ half):
            raise ValueError("Габаритная оболочка в начальной точке выходит за свободный объём")
    samples = []
    series = {}
    held = {}
    stopped_at = None
    extents = np.zeros(3)
    frame_stride = 1
    instrument_rates = instrument_rates or inertial_check
    if instrument_rates:
        # У всех сценариев карты длительность кратна 5 мс.
        frame_stride = round(scenario.dt_s * 200)
        scenario = replace(scenario, dt_s=0.005)
    for tick, (time_s, state) in enumerate(flight_states(scenario)):
        attitude = rotation(state[6:10])
        if hull_m is not None:
            # Огибающая при линейном переносе и кратчайшем повороте между отсчётами.
            angle = np.arccos(np.clip((np.trace(previous_rotation.T @ attitude) - 1) / 2, -1, 1))
            extents = np.maximum(abs(previous_rotation) @ half, abs(attitude) @ half)
            extents += radius * angle / 2
            free = collision_map.envelope_free(previous, state[:3], extents)
        else:
            free = collision_map.is_segment_free(previous, state[:3])
        if ground is not None and ground.clearance_bound(previous, state[:3], extents + 3.) <= 0:
            if not samples:
                raise ValueError("Стартовая оболочка с запасом касается рельефа")
            stopped_at = time_s
            status = "Полёт прерван: недостаточный зазор над рельефом"
            break
        if not free:
            if not samples:
                raise ValueError("Начальная оболочка с запасом вне свободного объёма")
            stopped_at = time_s
            status = "Полёт прерван: препятствие или граница карты"
            break
        previous_rotation = attitude
        previous = state[:3]
        wind_ned = wind_at(time_s)
        force, moment = scenario.loads(time_s, state)
        sensors = readings(state, force, body, origin, wind_ned, field_ned_ut)
        # Эталон для разбора берём до пропуска GNSS. В журнал прибора пропуск попадёт как null.
        geo, velocity = sensors["gnss_deg_deg_m"], sensors["gnss_velocity_ned_m_s"]
        sensors["range_down_m"] = downward_range(world, state[:3], attitude, ground)
        if instrument_rates:
            outage = gnss_outage is not None and gnss_outage[0] <= time_s < gnss_outage[1]
            for channel, stride in SENSOR_STRIDES.items():
                if tick % stride == 0:
                    value = None if outage and channel.startswith("gnss_") else sensors[channel]
                    held[channel] = value
                    if channel != "mag_ut" or value is not None:
                        series.setdefault(channel, []).append([time_s, value])
            sensors = held.copy()
        ground_h = float(ground.height(*state[:2])) if ground is not None else 0.
        frame = {
            "time": time_s, "ned": state[:3].tolist(), "attitude": attitude.tolist(),
            "wind_east": wind_ned[1], "geo": geo, "velocity": velocity, "sensors": sensors,
            "ground_height_m": ground_h, "agl_m": float(-state[2] - ground_h),
            **mechanical_sample(state, body, force, moment),
        }
        if tick % frame_stride == 0:
            samples.append(frame)
    else:
        status = "Сценарий завершён"
        if name == "route":
            status = ("Маршрут завершён" if np.linalg.norm(previous - DEMO_GOAL_NED) < 1
                      else "Время расчёта истекло, цель не достигнута")

    # Остановка может попасть между кадрами анимации. Сохраняем последний принятый шаг.
    if frame["time"] > samples[-1]["time"]:
        samples.append(frame)

    blockers = sorted(
        (i for i, box in enumerate(world.obstacles)
         if name == "route" and box.intersects_segment(DEMO_START_NED, DEMO_GOAL_NED)),
        key=lambda i: world.obstacles[i].minimum_ned[0],
    )
    summary = flight_summary(samples, world, blockers)
    summary["min_agl_m"] = min(r["agl_m"] for r in samples)
    summary["work_balance"] = work_balance(samples)
    summary["first_rejected_s"] = stopped_at
    if inertial_check:
        summary.update(audit_replay(samples, series, origin))
    summary["fuel_used_kg"] = None
    summary["fuel_used_l"] = None
    if fuel_log is not None and engine_log is not None:
        raise ValueError("Выберите один источник расхода")
    if fuel_log is not None or engine_log is not None:
        used, unit = fuel_usage(engine_log or fuel_log, [row["time"] for row in samples],
                                engine=engine_log is not None)
        for row, amount in zip(samples, used):
            row["fuel_used_" + unit] = float(amount)
        summary["fuel_used_" + unit] = float(used[-1])
    if battery_log is not None:
        summary["battery"] = battery_energy(battery_log, samples[0]["time"], samples[-1]["time"])
    if drive_profile is not None:
        summary["drive"] = lab_drive(drive_profile, samples)
        summary["fuel_used_l"] = samples[-1]["fuel_used_l"]
    estimate = None
    if drive_profile is None and fuel_log is None and engine_log is None and (name == "route" or fuel_rate_l_h is not None):
        # Пример O.S. GF40: 300 мл на 10–15 минут обычного полёта.
        rates = [1.2, 1.8] if fuel_rate_l_h is None else [fuel_rate_l_h, fuel_rate_l_h]
        hours = summary["duration_s"] / 3600
        estimate = {"rate_l_h": rates, "total_ml": [r * hours * 1000 for r in rates],
                    "source": "gf40" if fuel_rate_l_h is None else "user"}
    inventory = fuel_inventory(initial_fuel_l, samples, summary, estimate)
    if inventory is not None and inventory["shortfall_l"][1] > 0:
        status += (" · нехватка топлива" if inventory["shortfall_l"][0] > 0 else " · возможна нехватка топлива")
    if drive_profile is not None and summary["drive"]["unserved_wh"] > 0:
        status += " · превышена доступная мощность учебного привода"
    return {
        "fuel_estimate": estimate, "inventory": inventory,
        "relief": None if surface == "lab" else terrain_layer(origin, world.bounds),
        "surface": surface, "terrain_grid": mesh, "gnss_outage": gnss_outage,
        "passes": obstacle_passes(samples, world, blockers),
        "scenario": scenario.title, "mass_kg": body.mass_kg,
        "hull_m": None if hull_m is None else list(hull_m),
        "test_body": body is TEST_BODY, "sensor_series": series,
        "fuel_source": ("Расчёт по учебной нагрузочной характеристике" if drive_profile is not None else
                        "журнал режимов и удельного расхода" if engine_log else
                        "журнал расходомера" if fuel_log else None),
        **asdict(world), "blockers": blockers, "goal": DEMO_GOAL_NED if name == "route" else None,
        "samples": samples, "status": status,
        "summary": summary,
    }


def write_replay(path, replay):
    template = files("diam20k").joinpath("map_flight.html").read_text(encoding="utf-8")
    if template.count("/* FLIGHT_DATA */") != 1:
        raise ValueError("В шаблоне карты должна быть одна метка данных полёта")
    payload = json.dumps(replay, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    with Path(path).open("x", encoding="utf-8") as output:
        output.write(template.replace("/* FLIGHT_DATA */", payload))


def main():
    parser = argparse.ArgumentParser(description="Создать HTML с воспроизведением полёта 6-DOF.")
    parser.add_argument("--origin", type=float, nargs=3, metavar=("LAT", "LON", "H"),
                        default=(DEMO_ORIGIN.latitude_deg, DEMO_ORIGIN.longitude_deg,
                                 DEMO_ORIGIN.ellipsoid_height_m),
                        help="широта, долгота и высота уровня карты над эллипсоидом WGS 84")
    parser.add_argument("--output", type=Path, default=Path("map-flight.html"))
    parser.add_argument("--surface", choices=("flat", "lab", "utah"), default="flat", help="плоскость, учебные гряды или участок Юты")
    parser.add_argument("--gnss-outage", type=float, nargs=2, metavar=("FROM", "TO"),
                        help="исключить спутниковые измерения на интервале [FROM, TO), с; включает проверку ИНС")
    parser.add_argument("--initial-fuel-l", type=float, help="заданный объём топлива в начале записи, л")
    parser.add_argument("--calm", action="store_true", help="воспроизвести маршрут в штиль")
    parser.add_argument("--reference-log", type=Path, help="синхронные измерения высоты и скорости в NED")
    parser.add_argument("--scenario", choices=["route", *SCENARIOS], default="route")
    parser.add_argument("--hull-m", type=float, nargs=3, metavar=("L", "B", "H"),
                        help="габариты учебной оболочки вокруг ЦМ, м")
    parser.add_argument("--body-config", type=Path, help="JSON с измеренными массой и тензором инерции")
    parser.add_argument("--inertial-check", action="store_true", help="послеполётное счисление по профилю BMI088")
    parser.add_argument("--instrument-rates", action="store_true",
                        help="интегрировать с шагом 5 мс и записать каналы на проектных частотах")
    parser.add_argument("--battery-log", type=Path, help="журнал time_s,voltage_v,current_a (разряд положителен)")
    fuel = parser.add_mutually_exclusive_group()
    fuel.add_argument("--drive-profile", type=Path, help="нагрузочная характеристика учебного привода маршрута, JSON")
    fuel.add_argument("--fuel-rate-l-h", type=float, help="оценка при заданном постоянном расходе, л/ч")
    fuel.add_argument("--engine-log", type=Path, help="журнал time_s,rpm,torque_nm,bsfc_g_kwh")
    fuel.add_argument("--fuel-log", type=Path, help="синхронный журнал time_s,flow_kg_s или time_s,flow_l_h")
    parser.add_argument("--field-ned-ut", type=float, nargs=3, help="местное магнитное поле NED, мкТл")
    args = parser.parse_args()
    try:
        replay = simulate_demo(MapOrigin(*args.origin), wind=not args.calm, name=args.scenario,
                               body=load_body(args.body_config) if args.body_config else TEST_BODY,
                               fuel_log=args.fuel_log, engine_log=args.engine_log,
                               field_ned_ut=args.field_ned_ut, instrument_rates=args.instrument_rates, hull_m=args.hull_m,
                               inertial_check=args.inertial_check, fuel_rate_l_h=args.fuel_rate_l_h,
                               battery_log=args.battery_log, surface=args.surface, gnss_outage=args.gnss_outage,
                               initial_fuel_l=args.initial_fuel_l, drive_profile=args.drive_profile)
        if args.reference_log is not None:
            replay["comparison"] = compare_recording(args.reference_log, replay["samples"])
        write_replay(args.output, replay)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(args.output.resolve())


if __name__ == "__main__":
    main()

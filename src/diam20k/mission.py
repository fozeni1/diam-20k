"""Учебный стенд с вертикальными направляющими и двумя ветвями горизонтального пути."""

from dataclasses import asdict, dataclass
from math import dist, isfinite, sqrt

import numpy as np

from .instruments import downward_range
from .obstacle_map import Box, ObstacleMap
from .scenarios import GRAVITY_M_S2, TEST_BODY, rotor_loads


HEIGHT = 3.0
STEP_S = 0.25
STATIONS = {"start": (0., 0.), "fork": (4., 0.), "side": (4., 3.),
            "across": (8., 3.), "merge": (8., 0.), "finish": (12., 0.)}
NEXT = {"start": "fork", "side": "across", "across": "merge", "merge": "finish"}
BOUNDS = Box((-1,-1,-5), (13,4,0))
MAIN_GATE = Box((5.8,-0.4,-4), (6.2,0.4,0))
SIDE_GATE = Box((3.6,1.3,-4), (4.4,1.7,0))
IDENTITY = np.eye(3)
LABELS = {"landed": "Посадка на финишной площадке", "fuel": "Закончилось топливо маршевого привода",
          "battery": "Разрядилась батарея", "no_sensor": "Нет показания датчика",
          "no_lift": "Подъёмной тяги недостаточно для отрыва",
          "blocked": "Обе ветви закрыты", "contact": "Шаг пересекает препятствие",
          "timeout": "Истекло время опыта"}


@dataclass(frozen=True)
class Experiment:
    gates: str = "main"
    close_at_s: float = 4.
    fuel_ml: float = 1.5
    battery_wh: float = 1.
    missing_sensor: str | None = None
    disabled_rotor: int | None = None

    def __post_init__(self):
        if self.gates not in ("open", "main", "both") or self.missing_sensor not in (None, "fork", "height"):
            raise ValueError("Неизвестная конфигурация стенда")
        if self.disabled_rotor is not None and (type(self.disabled_rotor) is not int or self.disabled_rotor not in range(1, 5)):
            raise ValueError("Подъёмные моторы пронумерованы от 1 до 4")
        if not all(isfinite(x) and x >= 0 for x in (self.close_at_s, self.fuel_ml, self.battery_wh)):
            raise ValueError("Время и запасы должны быть конечными и неотрицательными")


@dataclass(frozen=True)
class Signals:
    range_m: float | None
    marker: str | None
    at_top: bool
    on_pad: bool
    main_clear: bool | None
    side_clear: bool | None


def observe(world, point, missing_sensor):
    marker = next((name for name, xy in STATIONS.items() if tuple(point[:2]) == xy), None)
    height = None if missing_sensor == "height" else downward_range(world, point, IDENTITY)
    main = side = None
    if marker == "fork" and missing_sensor != "fork":
        # Два датчика занятости на развилке макета, по одному на каждую ветвь.
        main = world.is_segment_free(point, (*STATIONS["merge"], point[2]))
        side = world.is_segment_free(point, (*STATIONS["side"], point[2]))
    return Signals(height, marker, bool(point[2] == -HEIGHT), bool(point[2] == 0.), main, side)


@dataclass
class Sequencer:
    phase: str = "takeoff"
    target: str = "fork"
    branch: str = "main"
    decided: bool = False

    def read(self, signals):
        if self.phase in ("takeoff", "landing") and signals.range_m is None:
            return "no_sensor"
        if self.phase == "takeoff" and signals.range_m >= HEIGHT and signals.at_top:
            self.phase = "cruise"
        if self.phase == "cruise":
            if signals.marker == "finish":
                self.phase = "landing"
            elif signals.marker == "fork" and not self.decided:
                if signals.main_clear is None or signals.side_clear is None:
                    return "no_sensor"
                if not signals.main_clear and not signals.side_clear:
                    return "blocked"
                self.branch = "main" if signals.main_clear else "side"
                self.target = "merge" if signals.main_clear else "side"
                self.decided = True
            elif signals.marker in NEXT:
                self.target = NEXT[signals.marker]
        if self.phase == "landing" and signals.range_m == 0 and signals.on_pad and signals.marker == "finish":
            return "landed"
        return None


def simulate_cycle(profile, case=Experiment()):
    point = np.array([*STATIONS["start"], 0.])
    # На стенде вертикальная скорость определяется избытком тяги и вязким сопротивлением направляющей.
    weight = TEST_BODY.mass_kg * GRAVITY_M_S2
    lift_sets = {phase: tuple(0. if motor == case.disabled_rotor else force for motor in range(1,5))
                 for phase, force in (("takeoff", 6.), ("landing", 4.), ("cruise", 0.))}
    vertical_speeds = {phase: (-rotor_loads(thrust)[0][2]-weight)/4.
                       for phase, thrust in lift_sets.items() if phase != "cruise"}
    # На горизонтальной направляющей P·η = F·v, F = b·v. В поворотах работает сама направляющая.
    speed = sqrt(profile.base_w*profile.efficiency / 20.)
    if speed == 0:
        raise ValueError("Для движения по горизонтальной ветви нужна положительная мощность")
    flow_ml_s = float(profile.flow_at(profile.base_w)) / 3.6
    if flow_ml_s <= 0:
        raise ValueError("Для включённого маршевого привода нужен положительный расход")
    closed_world = ObstacleMap(BOUNDS, {"open": (), "main": (MAIN_GATE,), "both": (MAIN_GATE, SIDE_GATE)}[case.gates])
    open_world = ObstacleMap(BOUNDS)
    controller = Sequencer()
    rows, events = [], []
    fuel, battery = case.fuel_ml, case.battery_wh
    time = 0.
    gates_reported = False
    while time < 60.:
        world = open_world if time < case.close_at_s else closed_world
        if world is closed_world and case.gates != "open" and not gates_reported:
            events.append((time, "gate_" + case.gates))
            gates_reported = True
        signals = observe(world, point, case.missing_sensor)
        old_phase, decided = controller.phase, controller.decided
        reason = controller.read(signals)
        if old_phase != controller.phase:
            events.append((time, controller.phase))
        if not decided and controller.decided:
            events.append((time, controller.branch))
        if reason:
            break

        phase = controller.phase
        cruise = phase == "cruise"
        lift = lift_sets[phase]
        # Стендовая характеристика электрического канала: 4 Вт на ньютон тяги.
        electric_w = profile.auxiliary_w + sum(lift)*4.
        until_empty = battery*3600/electric_w if electric_w > 0 else float("inf")
        if cruise and fuel <= 0:
            reason = "fuel"
            break
        if until_empty <= 0:
            reason = "battery"
            break

        destination = point.copy()
        if cruise:
            destination[:2] = STATIONS[controller.target]
            travel_speed = speed
        else:
            destination[2] = -HEIGHT if phase == "takeoff" else 0.
            travel_speed = vertical_speeds[phase] if phase == "takeoff" else -vertical_speeds[phase]
            if travel_speed <= 0:
                reason = "no_lift"
                break
        length = dist(point, destination)
        # Концевики, закрытие ворот и исчерпание ресурса могут попасть внутрь такта.
        to_target = length/travel_speed
        until_dry = fuel/flow_ml_s if cruise else float("inf")
        dt = min(STEP_S, to_target, until_empty, until_dry, 60.-time)
        if time < case.close_at_s:
            dt = min(dt, case.close_at_s-time)
        # При точном остатке топлива ошибка округления не должна оставить каретку перед концевиком.
        advanced = destination if to_target-dt <= 1e-12 else point+(destination-point)*(dt/to_target)
        if cruise and not world.is_segment_free(point, advanced):
            reason = "contact"
            break

        fuel = 0. if dt == until_dry else max(0., fuel - (flow_ml_s*dt if cruise else 0.))
        battery = 0. if dt == until_empty else max(0., battery-electric_w*dt/3600)
        rows.append({"start_s": time, "end_s": time+dt, "phase": phase, "signals": asdict(signals),
                     "from_ned": point.tolist(), "to_ned": advanced.tolist(), "lift_n": lift,
                     "pusher_n": 20.*speed if cruise else 0., "electric_w": electric_w,
                     "flow_ml_s": flow_ml_s if cruise else 0., "fuel_ml": fuel, "battery_wh": battery,
                     "branch": controller.branch})
        point = advanced
        time += dt
    else:
        reason = "timeout"
    world = open_world if time < case.close_at_s else closed_world
    return {"available_lift_n": float(sum(lift_sets["takeoff"])), "weight_n": weight,
            "final_lift_n": (0.,)*4, "final_pusher_n": 0.,
            "case": asdict(case), "steps": rows, "events": events, "stop": reason, "time_s": time,
            "signals_at_stop": asdict(observe(world, point, case.missing_sensor)), "end_ned": point.tolist(),
            "fuel_ml": fuel, "battery_wh": battery, "branch": controller.branch,
            "distance_m": sum(dist(row["from_ned"][:2], row["to_ned"][:2]) for row in rows)}

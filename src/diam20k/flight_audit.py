"""Послеполётные проверки по записи учебного сценария."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .scenarios import GRAVITY_M_S2


def mechanical_sample(state, body, force, moment):
    velocity, omega = state[3:6], state[10:13]
    kinetic = (body.mass_kg * (velocity @ velocity) + omega @ body.inertia_kg_m2 @ omega) / 2
    # Ось D направлена вниз, поэтому потенциальная энергия равна −mgD.
    return {"mechanical_energy_j": float(kinetic - body.mass_kg * GRAVITY_M_S2 * state[2]),
            "external_power_w": float(force @ velocity + moment @ omega)}


def work_balance(samples):
    time = np.array([r["time"] for r in samples])
    energy = np.array([r["mechanical_energy_j"] for r in samples])
    power = np.array([r["external_power_w"] for r in samples])
    work = np.r_[0, np.cumsum(np.diff(time) * (power[:-1] + power[1:]) / 2)]
    delta = energy - energy[0]
    return {"energy_change_j": float(delta[-1]), "external_work_j": float(work[-1]),
            "max_residual_j": float(np.max(np.abs(work-delta)))}


def fuel_inventory(initial_l, samples, summary, estimate):
    if initial_l is None:
        return None
    if not np.isfinite(initial_l) or initial_l < 0:
        raise ValueError("Начальный объём топлива должен быть конечным и неотрицательным")
    if summary["fuel_used_l"] is None and estimate is None:
        raise ValueError("Для остатка в литрах нужен объёмный журнал или расход в л/ч")
    for row in samples:
        if estimate is None:
            low = high = row["fuel_used_l"]
        else:
            low, high = (rate * (row["time"]-samples[0]["time"])/3600 for rate in estimate["rate_l_h"])
        row["fuel_remaining_l"] = [max(0., initial_l-high), max(0., initial_l-low)]
    return {"initial_l": initial_l, "remaining_l": samples[-1]["fuel_remaining_l"],
            "shortfall_l": [max(0., low-initial_l), max(0., high-initial_l)]}


def obstacle_passes(samples, world, blockers):
    """Разбор проходов вдоль северной оси с интерполяцией на входной и выходной гранях."""
    positions = np.array([r["ned"] for r in samples])
    times = np.array([r["time"] for r in samples])
    north = positions[:, 0]
    if blockers and np.any(np.diff(north) <= 0):
        raise ValueError("Для разбора проходов нужен маршрут с продвижением на север")
    passes = []
    for index in blockers:
        box = world.obstacles[index]
        lo, hi = box.minimum_ned, box.maximum_ned
        if north[-1] <= hi[0]:
            passes.append({"obstacle": index, "direction": "Проход начат" if north[-1] >= lo[0]
                           else "Не достигнуто", "exit_s": None, "face_gap_m": None})
            continue
        nodes = np.r_[lo[0], north[(north > lo[0]) & (north < hi[0])], hi[0]]
        east = np.interp(nodes, north, positions[:, 1])
        down = np.interp(nodes, north, positions[:, 2])
        gaps = [(float(east.min()-hi[1]), "Восточнее"), (float(lo[1]-east.max()), "Западнее"),
                (float(lo[2]-down.max()), "Сверху"), (float(down.min()-hi[2]), "Снизу")]
        gap, direction = max(gaps)
        passes.append({"obstacle": index, "direction": direction if gap > 0 else "С нескольких сторон",
                       "exit_s": float(np.interp(hi[0], north, times)),
                       "face_gap_m": gap if gap > 0 else None})
    return passes


@dataclass(frozen=True)
class DriveProfile:
    base_w: float
    efficiency: float
    auxiliary_w: float
    shaft_w: np.ndarray
    flow_l_h: np.ndarray

    def demand_at(self, mechanical_w):
        return self.base_w + max(mechanical_w, 0.) / self.efficiency

    def flow_at(self, power_w):
        return np.interp(power_w, self.shaft_w, self.flow_l_h)


def load_drive_profile(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        base, efficiency, auxiliary = [float(config[k]) for k in
                                      ("base_shaft_w", "transmission_efficiency", "auxiliary_w")]
        shaft = np.asarray(config["shaft_w"], dtype=float)
        flow = np.asarray(config["flow_l_h"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Неверные параметры профиля привода") from error
    if (shaft.ndim != 1 or len(shaft) < 2 or flow.shape != shaft.shape or
            not np.isfinite(np.r_[shaft, flow, base, efficiency, auxiliary]).all() or
            shaft[0] != 0 or np.any(np.diff(shaft) <= 0) or np.any(flow < 0) or
            not 0 <= base <= shaft[-1] or not 0 < efficiency <= 1 or auxiliary < 0):
        raise ValueError("Проверьте диапазоны мощности, расхода и КПД в профиле привода")

    return DriveProfile(base, efficiency, auxiliary, shaft, flow)


class DriveMeter:
    """Интеграл расхода и дефицита мощности по принятым шагам динамики."""

    def __init__(self, profile):
        self.profile = profile
        self.thresholds = [0.] + [float((w-profile.base_w)*profile.efficiency)
                                  for w in profile.shaft_w if w > profile.base_w]
        self.last_time = None
        self.first_time = None
        self.last_power = 0.
        self.used_l = self.missing_wh = self.peak_w = 0.

    def _point(self, mechanical_w):
        demand = self.profile.demand_at(mechanical_w)
        supplied = min(demand, self.profile.shaft_w[-1])
        return demand, demand-supplied, float(self.profile.flow_at(supplied))

    def accept(self, time, mechanical_w, available_l=None):
        # Между соседними отсчётами меняются наклон таблицы расхода и предел мощности.
        first = self.last_power if self.last_time is not None else mechanical_w
        dt = time-self.last_time if self.last_time is not None else 0.
        if dt < 0:
            raise ValueError("Время учёта привода должно возрастать")
        cuts = sorted((w-first)/(mechanical_w-first) for w in self.thresholds
                      if min(first, mechanical_w) < w < max(first, mechanical_w))
        _, previous_deficit, previous_rate = self._point(first)
        previous_u = litres = missing = 0.
        for u in [*cuts, 1.]:
            demand, deficit, rate = self._point(first + u*(mechanical_w-first))
            seconds = dt*(u-previous_u)
            litres += seconds*(rate+previous_rate)/7200
            missing += seconds*(deficit+previous_deficit)/7200
            previous_u, previous_deficit, previous_rate = u, deficit, rate
        if available_l is not None and self.used_l+litres > available_l:
            return None
        if self.first_time is None:
            self.first_time = time
        self.used_l += litres
        self.missing_wh += missing
        self.peak_w = max(self.peak_w, demand)
        self.last_time, self.last_power = time, mechanical_w
        return {"shaft_demand_w": float(demand), "fuel_rate_l_h": rate, "fuel_used_l": float(self.used_l)}

    def summary(self):
        p = self.profile
        return {"base_shaft_w": p.base_w, "efficiency": p.efficiency, "max_shaft_w": float(p.shaft_w[-1]),
                "peak_demand_w": float(self.peak_w), "unserved_wh": float(self.missing_wh),
                "auxiliary_w": p.auxiliary_w, "auxiliary_wh": float((self.last_time-self.first_time)*p.auxiliary_w/3600)}


def lab_drive(path, samples):
    """Расход по внешней записи механической мощности."""
    meter = DriveMeter(load_drive_profile(path))
    for row in samples:
        row.update(meter.accept(row["time"], row["external_power_w"]))
    return meter.summary()

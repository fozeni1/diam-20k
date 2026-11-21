"""Послеполётные проверки по записи учебного сценария."""

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


def lab_drive(path, samples):
    """Энергетический эквивалент учебного маршрута по положительной механической работе."""
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

    time = np.array([r["time"] for r in samples])
    mechanical = np.array([r["external_power_w"] for r in samples])
    # Между двумя кадрами мощность может пересечь несколько строк таблицы.
    # Добавляем моменты пересечения, чтобы интеграл учёл каждый участок характеристики.
    breaks = [time]
    for level in np.r_[0., (shaft[shaft > base]-base)*efficiency]:
        a, b = mechanical[:-1], mechanical[1:]
        crosses = ((a < level) & (b > level)) | ((b < level) & (a > level))
        breaks.append(time[:-1][crosses] + np.diff(time)[crosses]*(level-a[crosses])/(b[crosses]-a[crosses]))
    grid = np.unique(np.concatenate(breaks))
    demand = base + np.maximum(np.interp(grid, time, mechanical), 0)/efficiency
    supplied = np.minimum(demand, shaft[-1])
    rates = np.interp(supplied, shaft, flow)
    dt = np.diff(grid)
    litres = np.r_[0., np.cumsum(dt*(rates[1:]+rates[:-1])/7200)]
    deficit = demand-supplied
    missing_wh = np.sum(dt*(deficit[:-1]+deficit[1:])/7200)
    for i, row in enumerate(samples):
        j = np.searchsorted(grid, time[i])
        row.update(shaft_demand_w=float(demand[j]), fuel_rate_l_h=float(rates[j]), fuel_used_l=float(litres[j]))
    return {"base_shaft_w": base, "efficiency": efficiency, "max_shaft_w": float(shaft[-1]),
            "peak_demand_w": float(demand.max()), "unserved_wh": float(missing_wh),
            "auxiliary_w": auxiliary, "auxiliary_wh": float((time[-1]-time[0])*auxiliary/3600)}

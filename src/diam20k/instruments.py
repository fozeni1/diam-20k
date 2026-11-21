"""Идеальные измерения в центре масс и интеграл массового расхода по журналу."""

import csv
from pathlib import Path

import numpy as np

from .dynamics import air_data, rotation


def readings(state, force_body_n, body, origin, wind_ned, field_ned_ut=None):
    attitude = rotation(state[6:10])
    # Изотермическая атмосфера с заданным давлением на уровне карты.
    temperature_k = 288.15
    gas_constant = 287.05
    pressure_pa = 101325.0 * np.exp(9.80665 * state[2] / (gas_constant * temperature_k))
    density = pressure_pa / (gas_constant * temperature_k)
    air = air_data(state, wind_ned, density)
    return {
        "gyro_rad_s": state[10:13].tolist(),
        "accel_m_s2": (np.asarray(force_body_n) / body.mass_kg).tolist(),
        "mag_ut": None if field_ned_ut is None else (attitude.T @ field_ned_ut).tolist(),
        "baro_pa": float(pressure_pa),
        "dynamic_pressure_pa": air["dynamic_pressure_pa"],
        "gnss_deg_deg_m": origin.to_geodetic(state[:3]),
        "gnss_velocity_ned_m_s": (attitude @ state[3:6]).tolist(),
    }


def downward_range(world, position, attitude, surface=None):
    """Расстояние по оси +Z корпуса до первого объекта или земли, до 40 м."""
    direction = attitude[:, 2]
    end = position + 40.0 * direction
    hits = []
    ground = None
    if surface is not None:
        fraction = surface.first_hit(position, end)
        if fraction is not None:
            ground = 40.0 * fraction
    elif direction[2] > 0:
        ground = -position[2] / direction[2]
    if ground is not None and 0 <= ground <= 40:
        point = position + ground * direction
        if all(world.bounds.minimum_ned[i] <= point[i] <= world.bounds.maximum_ned[i] for i in (0, 1)):
            hits.append(ground)
    for box in world.obstacles:
        fraction = box.entry_fraction(position, end)
        if fraction is not None:
            hits.append(40.0 * fraction)
    return round(float(min(hits)), 2) if hits else None


def fuel_usage(path, times, *, engine=False):
    """Интеграл линейно интерполированного расхода с учётом всех узлов журнала."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = (["time_s", "rpm", "torque_nm", "bsfc_g_kwh"] if engine
                   else ["time_s", "flow_kg_s"])
        volume = not engine and reader.fieldnames == ["time_s", "flow_l_h"]
        if volume:
            columns = reader.fieldnames
        if reader.fieldnames != columns:
            raise ValueError("Требуются столбцы " + ",".join(columns))
        try:
            rows = [[float(row[c]) for c in columns] for row in reader]
        except (TypeError, ValueError) as exc:
            raise ValueError("Неверное число в журнале топлива") from exc
    if len(rows) < 2 or not np.all(np.isfinite(rows)):
        raise ValueError("Нужны хотя бы две конечные записи журнала топлива")
    values = np.array(rows).T
    t = values[0]
    if engine:
        rpm, torque, bsfc = values[1:]
        if np.any(values[1:] <= 0):
            raise ValueError("Удельный расход применим только к положительной мощности под нагрузкой")
        power_kw = rpm * (2 * np.pi / 60) * torque / 1000
        rate = power_kw * bsfc / 3_600_000
    else:
        rate = values[1] / 3600 if volume else values[1]
    if np.any(np.diff(t) <= 0) or np.any(rate < 0):
        raise ValueError("Время должно возрастать, расход должен быть неотрицательным")
    if t[0] > times[0] or t[-1] < times[-1]:
        raise ValueError("Журнал расходомера не покрывает весь интервал записи")
    knots = np.union1d(times, t[(t > times[0]) & (t < times[-1])])
    rates = np.interp(knots, t, rate)
    used = np.r_[0.0, np.cumsum(np.diff(knots) * (rates[1:] + rates[:-1]) / 2)]
    return used[np.searchsorted(knots, times)], "l" if volume else "kg"


def compare_recording(path, samples):
    """Расхождения на временных отметках независимого журнала, в СИ."""
    columns = ["time_s", "height_above_map_m", "vn_m_s", "ve_m_s", "vd_m_s"]
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError("Столбцы журнала для сравнения: " + ",".join(columns))
        try:
            data = np.asarray([[float(row[k]) for k in columns] for row in reader])
        except (TypeError, ValueError) as exc:
            raise ValueError("Неверное число в журнале для сравнения") from exc
    if len(data) < 2 or not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("Нужны конечные измерения хотя бы в два последовательных момента времени")
    t = data[:, 0]
    grid = np.array([s["time"] for s in samples])
    if t[0] < grid[0] or t[-1] > grid[-1]:
        raise ValueError("Измерения выходят за интервал записанной симуляции")
    values = np.array([[-s["ned"][2], *s["velocity"]] for s in samples])
    delta = np.column_stack([np.interp(t, grid, channel) for channel in values.T]) - data[:, 1:]
    return {"count": len(t), "first_s": float(t[0]), "last_s": float(t[-1]),
            "height_rmse_m": float(np.sqrt(np.mean(delta[:, 0]**2))),
            "velocity_vector_rmse_m_s": float(np.sqrt(np.mean(np.sum(delta[:, 1:]**2, axis=1))))}


def battery_energy(path, start, end):
    """Отданная и принятая батареей энергия по синхронному журналу клеммной мощности."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        cols = ["time_s", "voltage_v", "current_a"]
        if reader.fieldnames != cols:
            raise ValueError("Столбцы журнала батареи: " + ",".join(cols))
        try:
            rows = [[float(row[c]) for c in cols] for row in reader]
        except (TypeError, ValueError) as exc:
            raise ValueError("Неверное число в журнале батареи") from exc
    if len(rows) < 2 or not np.isfinite(rows).all():
        raise ValueError("Нужны хотя бы два конечных отсчёта батареи")
    time, voltage, current = np.array(rows).T
    if np.any(np.diff(time) <= 0) or np.any(voltage <= 0):
        raise ValueError("Время должно возрастать, напряжение должно быть положительным")
    if time[0] > start or time[-1] < end or end < start:
        raise ValueError("Журнал батареи не покрывает интервал полёта")
    knots = np.unique(np.r_[start, time[(time > start) & (time < end)], end])
    power = np.interp(knots, time, voltage * current)
    # При смене знака разделяем участок в точке нулевой мощности.
    i = np.flatnonzero(power[:-1] * power[1:] < 0)
    zeros = knots[i] - power[i] * (knots[i+1] - knots[i]) / (power[i+1] - power[i])
    grid = np.sort(np.r_[knots, zeros])
    p = np.interp(grid, knots, power)
    energy = np.diff(grid) * (p[:-1] + p[1:]) / 7200
    return {"discharged_wh": float(np.maximum(energy, 0).sum()),
            "charged_wh": float(np.maximum(-energy, 0).sum())}

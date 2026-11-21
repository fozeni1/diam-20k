"""Послеполётное сравнение инерциального счисления с траекторией сценария."""

import numpy as np

from .scenarios import GRAVITY_M_S2


def bmi088_samples(gyro, accel, rng):
    """Модель белого шума и 16-битного преобразования при 200 отсчётах/с."""
    def digitize(values, density, limit):
        # Односторонняя эквивалентная полоса идеального фильтра при 200 Гц.
        noisy = values + rng.normal(0, density * np.sqrt(100), values.shape)
        step = limit / 32768
        return np.clip(np.rint(noisy / step), -32768, 32767) * step

    return (digitize(gyro, np.deg2rad(0.014), np.deg2rad(250)),
            digitize(accel, 175e-6 * GRAVITY_M_S2, 6 * GRAVITY_M_S2))


def displacement(time, gyro, specific_force, initial_rotation, initial_velocity):
    """Смещение NED по синхронной записи IMU, известной начальной ориентации и скорости."""
    angles = np.diff(time)[:, None] * (gyro[1:] + gyro[:-1]) / 2
    rotation = initial_rotation.copy()
    velocity = initial_velocity.copy()
    positions = np.zeros((len(time), 3))
    for i, (angle, dt) in enumerate(zip(angles, np.diff(time))):
        x, y, z = angle
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        theta = np.linalg.norm(angle)
        # Через sinc формула Родрига работает и при нулевом угле.
        increment = (np.eye(3) + np.sinc(theta / np.pi) * skew +
                     0.5 * np.sinc(theta / (2*np.pi))**2 * (skew @ skew))
        next_rotation = rotation @ increment
        accel = (rotation @ specific_force[i] + next_rotation @ specific_force[i+1]) / 2
        accel[2] += GRAVITY_M_S2
        positions[i+1] = positions[i] + velocity * dt + accel * dt**2 / 2
        velocity = velocity + accel * dt
        rotation = next_rotation
    return positions


def reconstruct_recording(time, offsets, fix_times, fix_positions):
    """Привязка записанной IMU к двум соседним отметкам GNSS, с обеих сторон пропуска."""
    corrected = np.full_like(offsets, np.nan, dtype=float)
    if not len(fix_times):
        return corrected
    indices = np.searchsorted(time, fix_times)
    residual = np.asarray(fix_positions) - offsets[indices]
    # Для интерполяции нужна отметка с каждой стороны. Края записи оставляем пустыми.
    inside = (time >= fix_times[0]) & (time <= fix_times[-1])
    correction = np.column_stack([np.interp(time[inside], fix_times, axis) for axis in residual.T])
    corrected[inside] = offsets[inside] + correction
    return corrected


def audit_replay(samples, series, origin):
    """Сравнивает смещение по IMU с траекторией от старта и от последней отметки GNSS."""
    def values(channel):
        return np.array([v for _, v in series[channel]])

    t = np.array([t for t, _ in series["gyro_rad_s"]])
    gyro, accel = bmi088_samples(values("gyro_rad_s"), values("accel_m_s2"), np.random.default_rng(42))
    first = samples[0]
    offsets = displacement(t, gyro, accel, np.array(first["attitude"]), np.array(first["velocity"]))
    for name, measured in (("gyro_rad_s", gyro), ("accel_m_s2", accel)):
        series[name] = [[float(stamp), value] for stamp, value in zip(t, measured.tolist())]

    fixes = [(stamp, origin.to_ned(*geo)) for stamp, geo in series["gnss_deg_deg_m"] if geo is not None]
    reconstructed = reconstruct_recording(t, offsets, [f[0] for f in fixes], [f[1] for f in fixes])
    reconstruction_errors = []
    fix_index = -1
    max_relative_error = None
    for row in samples:
        i = np.searchsorted(t, row["time"])
        row["sensors"]["gyro_rad_s"] = series["gyro_rad_s"][i][1]
        row["sensors"]["accel_m_s2"] = series["accel_m_s2"][i][1]
        row["ins_displacement_ned_m"] = offsets[i].tolist()
        row["reconstructed_ned_m"] = row["reconstruction_error_m"] = None
        if np.isfinite(reconstructed[i]).all():
            row["reconstructed_ned_m"] = reconstructed[i].tolist()
            error = float(np.linalg.norm(reconstructed[i] - row["ned"]))
            row["reconstruction_error_m"] = error
            reconstruction_errors.append(error)
        row["ins_error_m"] = float(np.linalg.norm(offsets[i] + first["ned"] - row["ned"]))
        while fix_index+1 < len(fixes) and fixes[fix_index+1][0] <= row["time"]:
            fix_index += 1
            fix_time, fix_ned = fixes[fix_index]
            fix_offset = offsets[np.searchsorted(t, fix_time)]
        row["fix_age_s"] = row["ins_increment_error_m"] = None
        if fix_index >= 0:
            row["fix_age_s"] = row["time"] - fix_time
            error = np.linalg.norm((offsets[i] - fix_offset) - (row["ned"] - np.asarray(fix_ned)))
            row["ins_increment_error_m"] = float(error)
            max_relative_error = max(float(error), max_relative_error or 0.)
    return {"ins_final_error_m": samples[-1]["ins_error_m"],
            "ins_increment_max_m": max_relative_error,
            "reconstruction_max_m": max(reconstruction_errors, default=None),
            "reconstruction_frames": len(reconstruction_errors),
            "gnss_received": len(fixes), "gnss_expected": len(series["gnss_deg_deg_m"])}

"""Графики движения проверочного тела и отклонений от аналитических решений."""

import argparse
from dataclasses import replace
import json
from importlib.resources import files
from math import ceil
from pathlib import Path

import numpy as np

from .dynamics import rotation
from .inertial_lab import bmi088_samples, displacement
from .scenarios import SCENARIOS, run

# Прореживаем ряд только при подготовке графика. Ошибку считаем на полной записи.
MAX_POINTS = 600

# На логарифмической оси нельзя показать ноль. Отклонения меньше порога рисуются
# на пороге; он взят рядом с машинной точностью double (около 2,2e-16).
DEVIATION_FLOOR = 1e-16

PLACEHOLDER = "/* REPORT_DATA */"

DESCRIPTIONS = {
    "free-fall": "Внешних сил нет, действует только тяжесть: высота падает по параболе, "
                 "скорость растёт линейно.",
    "hover": "Сумма тяг равна весу, суммарный момент равен нулю: аппарат остаётся на месте.",
    "climb": "Тяга меняется по командам: висение, разгон, подъём с постоянной скоростью, "
             "торможение и снова висение на новой высоте.",
    "wind": "Тяга уравновешивает вес, воздух движется на восток: аппарат сносит, "
            "его скорость приближается к скорости ветра.",
}


def euler_deg(quaternion):
    """Крен, тангаж и курс в градусах для кватерниона «корпус → NED».

    Углы нужны только для показа: человеку они понятнее кватерниона. Порядок
    поворотов Z-Y-X (курс, тангаж, крен), как принято для связанных осей
    «вперёд, вправо, вниз». Вблизи тангажа ±90° крен и курс неразличимы; для
    сценариев с горизонтальным аппаратом это не важно.
    """
    matrix = rotation(quaternion)
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    # Ошибки округления могут вывести элемент матрицы за [-1, 1], и arcsin вернёт NaN.
    pitch = -np.arcsin(np.clip(matrix[2, 0], -1.0, 1.0))
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.degrees([roll, pitch, yaw])


def sample_indices(count, limit=MAX_POINTS):
    """Номера отсчётов для графика: равномерно, не больше limit + 1, с первым и последним.

    Последний отсчёт обязателен: по нему считаются итоговые числа, и график
    должен заканчиваться там же, где расчёт.
    """
    stride = max(1, ceil((count - 1) / limit))
    indices = list(range(0, count, stride))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return np.array(indices)


def _rounded(values):
    """Список чисел с семью значащими цифрами: этого хватает графику, а файл втрое меньше."""
    return [float(f"{value:.7g}") for value in values]


def _chart(title, unit, series, *, min_span=1.0, scale="linear", note=""):
    """Описание одного графика для шаблона.

    min_span задаёт наименьший размах оси. Без него ряд, который должен быть
    нулевым, растянулся бы на весь график, и шум округления порядка 1e-13
    выглядел бы как настоящие колебания.
    """
    return {"title": title, "unit": unit, "scale": scale, "min_span": min_span, "note": note,
            # Шаблон подписывает значения на пороге как «≤ порога», а не как точные.
            "floor": DEVIATION_FLOOR if scale == "log" else None,
            "series": [{"label": label, "values": _rounded(values)} for label, values in series]}


def _tile(label, value, unit, *, number_format="fixed"):
    """Одно итоговое число над графиками. Формат power печатает малые числа как 1,5·10⁻¹⁰."""
    return {"label": label, "value": float(value), "unit": unit, "format": number_format}


def build_report(scenario, *, inertial=False, seed=42):
    """Запускает сценарий и возвращает данные его страницы: числа итогов и ряды графиков."""
    if inertial:
        scenario = replace(scenario, dt_s=0.005)
    full_time_s, states = run(scenario)
    picked = sample_indices(len(full_time_s))
    time_s, picked_states = full_time_s[picked], states[picked]

    # Скорость в состоянии записана в связанных осях. Для графика она переводится
    # в оси NED, чтобы «на север» и «вверх» не зависели от ориентации корпуса.
    velocity_ned = np.array([rotation(state[6:10]) @ state[3:6] for state in picked_states])
    angles_deg = np.array([euler_deg(state[6:10]) for state in picked_states])
    forces_n = np.array([scenario.loads(moment_s, state)[0]
                         for moment_s, state in zip(time_s, picked_states)])
    altitude_m = -picked_states[:, 2]
    offset_m = picked_states[:, :2] - scenario.initial_state[:2]

    charts = [
        _chart("Высота", "м", [("высота", altitude_m)]),
        _chart("Скорость", "м/с", [
            ("на север", velocity_ned[:, 0]),
            ("на восток", velocity_ned[:, 1]),
            ("вверх", -velocity_ned[:, 2]),
        ]),
        _chart("Смещение от точки старта", "м", [
            ("на север", offset_m[:, 0]),
            ("на восток", offset_m[:, 1]),
        ]),
        _chart("Ориентация", "°", [
            ("крен", angles_deg[:, 0]),
            ("тангаж", angles_deg[:, 1]),
            ("курс", angles_deg[:, 2]),
        ]),
        # Тяга вверх — это отрицательная сила по оси z корпуса; знак меняется для показа.
        _chart("Внешние силы в осях корпуса", "Н", [
            ("вперёд", forces_n[:, 0]),
            ("вправо", forces_n[:, 1]),
            ("вверх", -forces_n[:, 2]),
        ], note="Без силы тяжести."),
    ]
    tiles = [
        _tile("Длительность", time_s[-1], "с"),
        _tile("Высота в конце", altitude_m[-1], "м"),
        _tile("Смещение в конце", np.linalg.norm(offset_m[-1]), "м"),
        _tile("Скорость в конце", np.linalg.norm(velocity_ned[-1]), "м/с"),
    ]

    if scenario.exact is not None:
        # Отклонение берётся по всем отсчётам расчёта, а не только по показанным:
        # наибольшее значение не должно зависеть от прореживания.
        gap = np.abs(states - scenario.exact(full_time_s)).max(axis=1)
        charts.append(_chart(
            "Отклонение от точного решения", "ед. СИ",
            [("наибольшее по 13 компонентам", np.maximum(gap[picked], DEVIATION_FLOOR))],
            scale="log",
            note="Логарифмическая шкала. Значения ниже 10⁻¹⁶ показаны на этом уровне."))
        tiles.append(_tile("Наибольшее отклонение", gap.max(), "ед. СИ", number_format="power"))

    if inertial:
        orientation = rotation(states[0, 6:10])
        velocity = orientation @ states[0, 3:6]
        gyro = states[:, 10:13]
        specific_force = np.array([scenario.loads(t, x)[0] / scenario.body.mass_kg
                                   for t, x in zip(full_time_s, states)])
        reference = states[:, :3] - states[0, :3]
        # BMP390: пример влияния постоянной ошибки +50 Па при известном давлении у земли.
        h_scale = 287.05 * 288.15 / 9.80665
        pressure = 101325 * np.exp(-altitude_m / h_scale)
        indicated = -h_scale * np.log((pressure + 50.0) / 101325)
        charts.append(_chart("Высота по барометру", "м",
                             [("Эталон", altitude_m), ("BMP390: смещение +50 Па", indicated)],
                             note="Численный эксперимент при постоянной температуре 15 °C."))
        curves = []
        for label, rates, forces in [
                ("Идеальная IMU", gyro, specific_force),
                ("Профиль BMI088", *bmi088_samples(gyro, specific_force, np.random.default_rng(seed)))]:
            estimate = displacement(full_time_s, rates, forces, orientation, velocity)
            error = np.linalg.norm(estimate - reference, axis=1)
            curves.append((label, error[picked]))
            tiles.append(_tile(label + " · ошибка в конце", error[-1], "м"))
        charts.append(_chart("Ошибка инерциального счисления", "м", curves, min_span=0.01,
                             note=f"Начальные ориентация и скорость известны. Частота 200 Гц, seed={seed}."))

    return {"name": scenario.name, "title": scenario.title,
            "description": DESCRIPTIONS.get(scenario.name, ""),
            "time": _rounded(time_s), "tiles": tiles, "charts": charts}


def render(scenarios, *, inertial=False, seed=42):
    """Возвращает готовый HTML: шаблон с подставленными данными всех сценариев."""
    data = {"scenarios": [build_report(scenario, inertial=inertial, seed=seed) for scenario in scenarios]}
    # allow_nan=False превращает NaN или бесконечность в ошибку здесь, а не в пустой
    # график в браузере. Замена «</» не даёт строке данных закрыть тег script раньше времени.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    template = files("diam20k").joinpath("scenario_report.html").read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise ValueError("Шаблон отчёта повреждён: метка данных должна встречаться один раз")
    return template.replace(PLACEHOLDER, payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Создать HTML с графиками сценариев.")
    parser.add_argument("names", nargs="*", metavar="сценарий",
                        help="имена: " + ", ".join(SCENARIOS) + "; без имён строятся все")
    parser.add_argument("--output", type=Path, default=Path("scenario-report.html"),
                        help="куда записать отчёт; существующий файл заменяется")
    parser.add_argument("--inertial", action="store_true", help="сравнить счисление по IMU с эталоном")
    parser.add_argument("--seed", type=int, default=42, help="начальное значение генератора шума")
    args = parser.parse_args(argv)
    unknown = [name for name in args.names if name not in SCENARIOS]
    if unknown:
        parser.error("неизвестный сценарий: " + ", ".join(unknown))
    try:
        html = render([SCENARIOS[name]() for name in args.names or SCENARIOS],
                      inertial=args.inertial, seed=args.seed)
        args.output.write_text(html, encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(args.output.resolve())


if __name__ == "__main__":
    main()

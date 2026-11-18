"""Воспроизводимые сценарии полёта для механического ядра 6-DOF.

Сценарий задаёт начальное состояние из 13 компонентов и функцию
loads(time_s, state) -> (сила, момент): связанные оси, без тяжести, моменты
относительно ЦМ, единицы СИ. Рядом хранится точное решение, чтобы расчёт
можно было сравнить с формулой.

Масса, инерция, координаты винтов и площади сопротивления здесь условные.
Они нужны для проверки расчёта и НЕ являются параметрами ДИАМ-20К.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil

import numpy as np

from .dynamics import RigidBody, air_data, point_load, step_rk4

GRAVITY_M_S2 = 9.80665

# Проверочные значения до появления массы, инерции и геометрии в configs/diam20k.json.
TEST_BODY = RigidBody(mass_kg=2.0, inertia_kg_m2=np.diag([2.0, 3.0, 4.0]))
# Точки приложения тяги подъёмных винтов от ЦМ (x вперёд, y вправо, z вниз):
# передний правый, передний левый, задний правый, задний левый. Нумерация условная.
TEST_ROTORS_BODY_M = ((0.5, 0.7, 0.0), (0.5, -0.7, 0.0), (-0.5, 0.7, 0.0), (-0.5, -0.7, 0.0))
# Произведения C_D·A по связанным осям x, y, z для сопротивления корпуса в висении.
TEST_DRAG_AREA_M2 = (0.2, 0.4, 0.6)


def _positive(value, label):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label}: нужно конечное положительное число")
    return float(value)


def _vector(value, size, label):
    result = np.array(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}: нужно {size} конечных чисел")
    return result


def _rotors(value):
    rotors = np.array(value, dtype=float)
    if rotors.ndim != 2 or rotors.shape[1] != 3 or not np.all(np.isfinite(rotors)):
        raise ValueError("Координаты винтов: нужны конечные тройки (x, y, z)")
    return rotors


@dataclass(frozen=True)
class Scenario:
    """Всё, что нужно для повторения расчёта.

    exact(time_s) возвращает состояния точного решения формы (n, 13)
    или отсутствует, если формулы для сценария нет.
    """

    name: str
    title: str
    body: RigidBody
    initial_state: np.ndarray
    loads: Callable
    duration_s: float
    dt_s: float
    exact: Callable | None = None

    def __post_init__(self):
        state = _vector(self.initial_state, 13, "Начальное состояние")
        norm = np.linalg.norm(state[6:10])
        if norm < 1e-12:
            raise ValueError("Начальное состояние: кватернион не может быть нулевым")
        state[6:10] /= norm
        _positive(self.duration_s, "Длительность сценария")
        _positive(self.dt_s, "Шаг интегрирования")
        state.setflags(write=False)
        object.__setattr__(self, "initial_state", state)


def level_state(altitude_m=0.0):
    """Неподвижный горизонтальный аппарат на высоте h = −D, нос на север."""
    state = np.zeros(13)
    state[2] = -altitude_m
    state[6] = 1.0
    return state


def rotor_loads(thrusts_n, rotors_body_m=TEST_ROTORS_BODY_M):
    """Суммарные сила и момент подъёмных винтов; тяга каждого направлена по −z.

    Временная замена модели силовой установки. Реактивные моменты винтов
    приняты нулевыми: их знаки и величины для аппарата неизвестны.
    """
    rotors = _rotors(rotors_body_m)
    thrusts = _vector(thrusts_n, len(rotors), "Тяги винтов")
    if thrusts.min() < 0:
        raise ValueError("Тяга винта не может быть отрицательной")
    force, moment = np.zeros(3), np.zeros(3)
    for position, thrust in zip(rotors, thrusts):
        rotor_force, rotor_moment = point_load(position, [0.0, 0.0, -thrust], [0.0, 0.0, 0.0])
        force += rotor_force
        moment += rotor_moment
    return force, moment


def hover_thrusts(body, rotors_body_m=TEST_ROTORS_BODY_M):
    """Тяги винтов для равновесия: их сумма равна весу, момент относительно ЦМ нулевой.

    Винт с тягой T в точке (x, y, z) создаёт момент (−y·T, x·T, 0). Уравнений три,
    тяг обычно четыре, поэтому берётся решение с наименьшей суммой квадратов.
    """
    rotors = _rotors(rotors_body_m)
    equations = np.vstack([np.ones(len(rotors)), -rotors[:, 1], rotors[:, 0]])
    weight_n = body.mass_kg * GRAVITY_M_S2
    target = np.array([weight_n, 0.0, 0.0])
    thrusts = np.linalg.lstsq(equations, target, rcond=None)[0]
    if not np.allclose(equations @ thrusts, target, rtol=0, atol=1e-9 * weight_n):
        raise ValueError("Равновесие невозможно: винты не уравновешивают вес и момент")
    if thrusts.min() < 0:
        raise ValueError("Равновесие требует отрицательной тяги: ЦМ далеко от центра винтов")
    return thrusts


def free_fall(body=TEST_BODY, *, altitude_m=1000.0, duration_s=10.0, dt_s=0.01):
    """Свободное падение из покоя: внешних нагрузок нет, действует только тяжесть."""
    if altitude_m < GRAVITY_M_S2 * duration_s**2 / 2:
        raise ValueError("Аппарат окажется ниже земли, а контакта с землёй в модели нет")
    start = level_state(altitude_m)

    def exact(time_s):
        states = np.tile(start, (len(time_s), 1))
        states[:, 2] += GRAVITY_M_S2 * time_s**2 / 2
        states[:, 5] = GRAVITY_M_S2 * time_s
        return states

    return Scenario("free-fall", "Свободное падение", body, start,
                    lambda time_s, state: (np.zeros(3), np.zeros(3)), duration_s, dt_s, exact)


def hover(body=TEST_BODY, rotors_body_m=TEST_ROTORS_BODY_M, *,
          altitude_m=50.0, duration_s=30.0, dt_s=0.01):
    """Равновесие: горизонтальный аппарат, сумма тяг равна весу, суммарный момент нулевой."""
    force, moment = rotor_loads(hover_thrusts(body, rotors_body_m), rotors_body_m)
    start = level_state(altitude_m)
    return Scenario("hover", "Висение в равновесии", body, start,
                    lambda time_s, state: (force, moment), duration_s, dt_s,
                    lambda time_s: np.tile(start, (len(time_s), 1)))


def _ramp_motion(knots_s, acceleration, time_s):
    """Точные скорость и путь при кусочно-линейном ускорении и старте из покоя."""
    span = np.diff(knots_s)
    speed = np.concatenate([[0.0], np.cumsum((acceleration[:-1] + acceleration[1:]) * span / 2)])
    path = np.concatenate([[0.0], np.cumsum(
        speed[:-1] * span + (2 * acceleration[:-1] + acceleration[1:]) * span**2 / 6)])
    i = np.clip(np.searchsorted(knots_s, time_s, side="right") - 1, 0, len(span) - 1)
    tau = time_s - knots_s[i]
    slope = (acceleration[i + 1] - acceleration[i]) / span[i]
    return (speed[i] + acceleration[i] * tau + slope * tau**2 / 2,
            path[i] + speed[i] * tau + acceleration[i] * tau**2 / 2 + slope * tau**3 / 6)


def climb(body=TEST_BODY, rotors_body_m=TEST_ROTORS_BODY_M, *, altitude_m=50.0,
          height_gain_m=60.0, climb_rate_m_s=4.0, acceleration_m_s2=1.0,
          ramp_s=1.0, hold_s=5.0, dt_s=0.01):
    """Набор высоты по командам тяги: висение, разгон, подъём, торможение, висение.

    Тяги всех винтов меняются одинаково и линейно между узлами по времени.
    Импульс разгона имеет форму трапеции с фронтами ramp_s, импульс торможения
    зеркален ему, поэтому аппарат останавливается ровно на новой высоте.
    """
    for value, label in ((height_gain_m, "Набор высоты"), (climb_rate_m_s, "Скорость подъёма"),
                         (acceleration_m_s2, "Ускорение"), (ramp_s, "Фронт команды"),
                         (hold_s, "Время висения")):
        _positive(value, label)
    if acceleration_m_s2 >= GRAVITY_M_S2:
        raise ValueError("Ускорение должно быть меньше g: иначе тяга при торможении отрицательна")
    plateau_s = climb_rate_m_s / acceleration_m_s2 - ramp_s
    cruise_s = height_gain_m / climb_rate_m_s - (plateau_s + 2 * ramp_s)
    if min(plateau_s, cruise_s) < -1e-9:
        raise ValueError("Профиль подъёма невыполним: нужны меньшая скорость или фронт "
                         "либо больший набор высоты")

    spans = np.array([hold_s, ramp_s, max(plateau_s, 0.0), ramp_s, max(cruise_s, 0.0),
                      ramp_s, max(plateau_s, 0.0), ramp_s, hold_s])
    keep = np.concatenate([[True], spans > 0])  # Участок нулевой длины не добавляет узла.
    knots_s = np.concatenate([[0.0], np.cumsum(spans)])[keep]
    # Заданное вертикальное ускорение (вверх) в узлах, м/с².
    profile = acceleration_m_s2 * np.array([0.0, 0, 1, 1, 0, 0, -1, -1, 0, 0])[keep]
    balance = hover_thrusts(body, rotors_body_m)
    start = level_state(altitude_m)

    def loads(time_s, state):
        thrusts = balance * (1 + np.interp(time_s, knots_s, profile) / GRAVITY_M_S2)
        return rotor_loads(thrusts, rotors_body_m)

    def exact(time_s):
        speed, path = _ramp_motion(knots_s, profile, time_s)
        states = np.tile(start, (len(time_s), 1))
        states[:, 2] -= path
        states[:, 5] = -speed
        return states

    return Scenario("climb", "Набор высоты", body, start, loads, float(knots_s[-1]), dt_s, exact)


def wind(body=TEST_BODY, rotors_body_m=TEST_ROTORS_BODY_M, *, wind_ned_m_s=(0.0, 5.0, 0.0),
         drag_area_m2=TEST_DRAG_AREA_M2, density_kg_m3=1.225,
         altitude_m=50.0, duration_s=30.0, dt_s=0.01):
    """Висение в постоянном ветре: аппарат сносит, его скорость стремится к скорости ветра.

    wind_ned_m_s показывает, куда движется воздух. Сопротивление корпуса считается
    по осям: F = −½·ρ·C_D·A·v_a·|v_a|, приложено в ЦМ и момента не создаёт.
    """
    wind_ned = _vector(wind_ned_m_s, 3, "Ветер")
    drag_area = _vector(drag_area_m2, 3, "Площади сопротивления")
    if drag_area.min() < 0:
        raise ValueError("Площади сопротивления не могут быть отрицательными")
    factor = 0.5 * _positive(density_kg_m3, "Плотность воздуха") * drag_area
    thrust_force, thrust_moment = rotor_loads(hover_thrusts(body, rotors_body_m), rotors_body_m)
    start = level_state(altitude_m)

    def loads(time_s, state):
        air = air_data(state, wind_ned, density_kg_m3)["velocity_body_m_s"]
        return thrust_force - factor * air * np.abs(air), thrust_moment

    def exact(time_s):
        states = np.tile(start, (len(time_s), 1))
        for axis in range(3):  # Аппарат горизонтален, поэтому связанные оси совпадают с NED.
            rate = factor[axis] * abs(wind_ned[axis]) / body.mass_kg
            if rate > 0:
                states[:, 3 + axis] = wind_ned[axis] * rate * time_s / (1 + rate * time_s)
                states[:, axis] += wind_ned[axis] * (time_s - np.log1p(rate * time_s) / rate)
        return states

    return Scenario("wind", "Висение в постоянном ветре", body, start, loads,
                    duration_s, dt_s, exact)


SCENARIOS = {"free-fall": free_fall, "hover": hover, "climb": climb, "wind": wind}


def run(scenario):
    """Интегрирует сценарий методом RK4. Возвращает время (n,) и состояния (n, 13).

    Число шагов округляется вверх: шаг не больше dt_s, сетка равномерная,
    последняя точка приходится точно на конец сценария.
    """
    steps = max(1, ceil(scenario.duration_s / scenario.dt_s - 1e-9))
    time_s = np.linspace(0.0, scenario.duration_s, steps + 1)
    states = np.empty((steps + 1, 13))
    states[0] = scenario.initial_state
    for i in range(steps):
        states[i + 1] = step_rk4(states[i], time_s[i], time_s[i + 1] - time_s[i], scenario.body,
                                 scenario.loads, gravity_m_s2=GRAVITY_M_S2)
    return time_s, states


def deviation(scenario, time_s, states):
    """Наибольшее по модулю отличие расчёта от точного решения по всем 13 компонентам."""
    if scenario.exact is None:
        return None
    return float(np.abs(states - scenario.exact(time_s)).max())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Рассчитать сценарии и сравнить с точным решением.")
    parser.add_argument("names", nargs="*", metavar="сценарий",
                        help="имена: " + ", ".join(SCENARIOS) + "; без имён считаются все")
    args = parser.parse_args(argv)
    unknown = [name for name in args.names if name not in SCENARIOS]
    if unknown:
        parser.error("неизвестный сценарий: " + ", ".join(unknown))
    print("Проверочное тело и условная геометрия, не параметры ДИАМ-20К.")
    for name in args.names or SCENARIOS:
        scenario = SCENARIOS[name]()
        time_s, states = run(scenario)
        north, east, down = states[-1, :3]
        error = deviation(scenario, time_s, states)
        print(f"{name}: {scenario.title}. Время {time_s[-1]:g} с, высота {-down:.3f} м, "
              f"смещение на север {north:.3f} м, на восток {east:.3f} м, "
              "отклонение от точного решения "
              + ("не определено" if error is None else f"{error:.1e}"))


if __name__ == "__main__":
    main()

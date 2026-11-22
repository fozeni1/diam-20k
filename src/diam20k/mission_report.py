"""Сравнение опытов на учебной развилке."""
import json
from dataclasses import asdict
from importlib.resources import files

from .mission import Experiment, HEIGHT, LABELS, MAIN_GATE, SIDE_GATE, STATIONS, simulate_cycle


def render_missions(profile):
    experiments = {
        "Закрытие основного прохода": Experiment(),
        "Свободный проход": Experiment(gates="open"),
        "Обе ветви закрыты": Experiment(gates="both"),
        "Пропало показание на развилке": Experiment(missing_sensor="fork"),
        "Малый запас топлива": Experiment(fuel_ml=.15),
        "Малый заряд батареи": Experiment(battery_wh=.05),
        "Отключён второй подъёмный мотор": Experiment(disabled_rotor=2),
    }
    data = {"stations": STATIONS, "height_m": HEIGHT, "gates": [asdict(MAIN_GATE), asdict(SIDE_GATE)], "labels": LABELS,
            "runs": {label: simulate_cycle(profile, case) for label, case in experiments.items()}}
    template = files("diam20k").joinpath("mission.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    return template.replace("/* MISSION_DATA */", payload)

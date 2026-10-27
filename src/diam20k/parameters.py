"""Load only explicitly known mechanical parameters from a JSON document."""

import json
from pathlib import Path

from .dynamics import RigidBody


def load_body(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    parameters = config["model_parameters"]
    required = ("mass_kg", "inertia_cg_kg_m2")
    missing = [key for key in required if parameters.get(key, {}).get("value") is None]
    if missing:
        raise ValueError("Не заданы реальные параметры: " + ", ".join(missing))
    return RigidBody(parameters["mass_kg"]["value"], parameters["inertia_cg_kg_m2"]["value"])

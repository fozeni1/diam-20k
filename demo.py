"""Демонстрационные полёты проектной группы."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from diam20k.flight_audit import load_drive_profile
from diam20k.mission_report import render_missions
from diam20k.map_flight import simulate_demo, write_replay
from diam20k.scenario_report import render
from diam20k.scenarios import SCENARIOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parent = parser.parse_args().output_dir
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(prefix=datetime.now().strftime("flight-%Y%m%d-%H%M%S-"), dir=parent))
    drive_path = ROOT / "configs" / "lab-drive.json"
    mission_path = folder / "mission.html"
    mission_path.write_text(render_missions(load_drive_profile(drive_path)), encoding="utf-8")
    print(mission_path.resolve(), flush=True)
    for surface, filename in (("lab", "training.html"), ("utah", "utah.html")):
        print(f"Расчёт {filename}…", flush=True)
        replay = simulate_demo(surface=surface, hull_m=(2, 4, 1), initial_fuel_l=1,
                               drive_profile=drive_path,
                               field_ned_ut=(20, 0, 40),
                               gnss_outage=(60., 80.) if surface == "lab" else None)
        replay["has_cycle_report"] = True
        path = folder / filename
        write_replay(path, replay)
        print(f"{replay['status']}. {path.resolve()}", flush=True)
    path = folder / "scenarios.html"
    path.write_text(render([make() for make in SCENARIOS.values()], inertial=True), encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()

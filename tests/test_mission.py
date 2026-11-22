from dataclasses import replace
from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from diam20k.flight_audit import DriveMeter, load_drive_profile
from diam20k.map_flight import simulate_demo
from diam20k.mission_report import render_missions
from diam20k.mission import Experiment, HEIGHT, Sequencer, Signals, simulate_cycle


PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs" / "lab-drive.json"
PROFILE = load_drive_profile(PROFILE_PATH)


class CycleTests(unittest.TestCase):
    def test_one_cycle_uses_separate_drives_and_returns_to_ground(self):
        run = simulate_cycle(PROFILE)
        self.assertEqual(run["end_ned"], [12., 0., 0.])
        self.assertEqual(run["stop"], "landed")
        self.assertTrue(run["signals_at_stop"]["on_pad"])
        self.assertEqual(run["steps"][0]["from_ned"], [0, 0, 0])
        self.assertEqual({s["phase"] for s in run["steps"]}, {"takeoff", "cruise", "landing"})
        used_ml = used_wh = 0.
        for row in run["steps"]:
            dt = row["end_s"]-row["start_s"]
            self.assertGreater(dt, 0)
            start, end = row["from_ned"], row["to_ned"]
            if row["phase"] == "cruise":
                self.assertEqual(end[2], -HEIGHT)
                self.assertEqual(sum(row["lift_n"]), 0)
                self.assertGreater(row["pusher_n"], 0)
                self.assertGreater(row["flow_ml_s"], 0)
            else:
                self.assertEqual(start[:2], end[:2])
                self.assertTrue(all(v > 0 for v in row["lift_n"]))
                self.assertEqual(len(row["lift_n"]), 4)
                self.assertEqual(row["pusher_n"], 0)
                self.assertEqual(row["flow_ml_s"], 0)
            used_ml += row["flow_ml_s"]*dt
            used_wh += row["electric_w"]*dt/3600
            self.assertAlmostEqual(run["case"]["fuel_ml"]-row["fuel_ml"], used_ml)
            self.assertAlmostEqual(run["case"]["battery_wh"]-row["battery_wh"], used_wh)

    def test_barrier_after_launch_changes_branch_and_consumption(self):
        clear = simulate_cycle(PROFILE, Experiment(gates="open"))
        detour = simulate_cycle(PROFILE)
        for run, distance in ((clear,12.),(detour,18.)):
            self.assertEqual(run["stop"], "landed")
            self.assertAlmostEqual(run["distance_m"], distance)
        selected = next(t for t, event in detour["events"] if event == "side")
        self.assertGreater(selected, detour["case"]["close_at_s"])
        self.assertLess(detour["events"][0][0], detour["case"]["close_at_s"])
        self.assertEqual([s for s in clear["steps"] if s["end_s"] <= selected],
                         [s for s in detour["steps"] if s["end_s"] <= selected])
        self.assertGreater(clear["fuel_ml"], detour["fuel_ml"])
        self.assertGreater(clear["battery_wh"], detour["battery_wh"])

    def test_decision_depends_on_readings(self):
        sequencer = Sequencer()
        top = Signals(2., "start", True, False, None, None)
        sequencer.read(top)
        self.assertEqual(sequencer.phase,"takeoff")
        sequencer.read(replace(top, range_m=HEIGHT))
        self.assertEqual(sequencer.phase,"cruise")
        sequencer.read(Signals(HEIGHT,"fork",True,False,False,True))
        self.assertEqual(sequencer.target,"side")
        for case, reason in ((Experiment(gates="both"),"blocked"),
                             (Experiment(missing_sensor="fork"),"no_sensor")):
            run = simulate_cycle(PROFILE,case)
            self.assertEqual(run["stop"],reason)
            self.assertEqual(run["end_ned"],[4,0,-HEIGHT])
        self.assertEqual(simulate_cycle(PROFILE,Experiment(missing_sensor="height"))["steps"],[])

    def test_no_jump_when_gate_closes_after_choice(self):
        r = simulate_cycle(PROFILE, Experiment(close_at_s=5.1))
        self.assertEqual(r["branch"], "main")
        self.assertEqual(r["stop"], "contact")
        self.assertLess(r["end_ned"][0], 5.8)

    def test_fuel_empty_interrupts_propulsion_mid_step(self):
        run = simulate_cycle(PROFILE, Experiment(fuel_ml=.15))
        self.assertEqual(run["stop"],"fuel")
        self.assertEqual(run["fuel_ml"],0)
        self.assertEqual(run["final_pusher_n"],0)
        self.assertGreater(run["battery_wh"],0)
        self.assertGreater(run["end_ned"][1],0)
        self.assertLess(run["end_ned"][1],3)
        self.assertNotIn("landing", [s["phase"] for s in run["steps"]])
        empty = simulate_cycle(PROFILE, Experiment(fuel_ml=0))
        self.assertEqual(empty["end_ned"], [0.,0.,-HEIGHT])

    def test_exact_fuel_for_horizontal_section_allows_electric_landing(self):
        for gates, travel_s in (("open", 6.), ("main",9.)):
            volume = travel_s*PROFILE.flow_at(PROFILE.base_w)/3.6
            run = simulate_cycle(PROFILE, Experiment(gates=gates, fuel_ml=volume))
            self.assertEqual(run["stop"], "landed")
            self.assertAlmostEqual(run["fuel_ml"],0)
            self.assertTrue(run["signals_at_stop"]["on_pad"])

    def test_battery_and_fourth_lift_channel_influence_takeoff(self):
        for case, why in ((Experiment(battery_wh=.05),"battery"),(Experiment(disabled_rotor=4),"no_lift")):
            r = simulate_cycle(PROFILE,case)
            self.assertEqual(r["stop"],why)
            self.assertEqual(r["fuel_ml"],case.fuel_ml)
            self.assertEqual(r["end_ned"][:2],[0,0])
            self.assertGreater(r["end_ned"][2],-HEIGHT)

    def test_common_drive_profile_changes_consumption_and_progress(self):
        case=Experiment(fuel_ml=.15)
        base = simulate_cycle(PROFILE,case)
        fast_drain = simulate_cycle(replace(PROFILE, flow_l_h=2*PROFILE.flow_l_h),case)
        self.assertAlmostEqual(fast_drain["distance_m"]*2,base["distance_m"])
        self.assertLess(fast_drain["time_s"],base["time_s"])

    def test_report_contains_observed_gate_events_and_resource_failures(self):
        page = render_missions(PROFILE)
        data, _ = json.JSONDecoder().raw_decode(page.split("const data = ", 1)[1])
        self.assertEqual(len(data["runs"]), 7)
        run = data["runs"]["Закрытие основного прохода"]
        events = {code: time for time, code in run["events"]}
        self.assertLess(events["cruise"], events["gate_main"])
        self.assertLess(events["gate_main"], events["side"])
        self.assertLess(events["side"], events["landing"])
        self.assertEqual(data["runs"]["Малый запас топлива"]["stop"], "fuel")
        self.assertEqual(data["runs"]["Отключён второй подъёмный мотор"]["available_lift_n"], 18)


class RouteBudgetTests(unittest.TestCase):
    def test_rejected_interval_does_not_change_meter(self):
        meter = DriveMeter(PROFILE)
        meter.accept(0,0)
        before = meter.summary()
        self.assertIsNone(meter.accept(5,400, available_l=0))
        self.assertEqual(meter.summary(),before)
        self.assertEqual(meter.used_l,0)
        self.assertEqual(meter.last_time,0)
        self.assertGreater(meter.accept(1,0)["fuel_used_l"],0)

    def test_route_rejects_a_step_above_drive_capacity(self):
        with TemporaryDirectory() as folder:
            path = Path(folder, "limited.json")
            config = json.loads(PROFILE_PATH.read_text())
            config.update(shaft_w=[0,101], flow_l_h=[.08,.17])
            path.write_text(json.dumps(config))
            r = simulate_demo(drive_profile=path, initial_fuel_l=1)
        self.assertIn("потребную мощность", r["status"])
        self.assertGreater(r["summary"]["drive"]["rejected_demand_w"], 101)
        self.assertLess(r["summary"]["duration_s"], r["summary"]["first_rejected_s"])
        self.assertGreater(r["inventory"]["remaining_l"][0], .99)

    def test_explicit_constant_flow_cuts_before_overdraft(self):
        r = simulate_demo(fuel_rate_l_h=3.6, initial_fuel_l=.00012)
        self.assertIn("топлива",r["status"])
        self.assertAlmostEqual(r["summary"]["duration_s"],.1)
        self.assertAlmostEqual(r["summary"]["first_rejected_s"],.15)

    def test_route_budget_matches_last_accepted_sensor_step(self):
        for litres in (0, .0000014):
            r = simulate_demo(drive_profile=PROFILE_PATH,initial_fuel_l=litres,inertial_check=True)
            self.assertIn("топлива",r["status"])
            last = r["samples"][-1]
            self.assertEqual(last["time"],r["sensor_series"]["gyro_rad_s"][-1][0])
            self.assertAlmostEqual(last["time"]+.005,r["summary"]["first_rejected_s"])
            self.assertLessEqual(last["fuel_used_l"],litres)
            self.assertEqual(last["fuel_used_l"],r["summary"]["fuel_used_l"])
            self.assertAlmostEqual(r["inventory"]["remaining_l"][0],litres-last["fuel_used_l"])
            self.assertTrue(np.isfinite(last["ins_error_m"]))

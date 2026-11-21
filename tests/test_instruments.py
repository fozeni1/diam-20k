from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from diam20k.dynamics import RigidBody
from diam20k.geography import MapOrigin
from diam20k.instruments import battery_energy, fuel_usage, readings
from diam20k.map_flight import DEMO_ORIGIN, simulate_demo
from diam20k.obstacle_map import Box, DEMO_MAP, ObstacleMap
from diam20k.scenarios import climb, free_fall, GRAVITY_M_S2, level_state, run, TEST_BODY


class InstrumentTests(unittest.TestCase):
    def test_shared_solver_matches_new_climb_scenario(self):
        replay = simulate_demo(name="climb")
        time, states = run(climb())
        np.testing.assert_array_equal([r["time"] for r in replay["samples"]], time)
        np.testing.assert_array_equal([r["ned"] for r in replay["samples"]], states[:, :3])
        self.assertIsNone(replay["goal"])
        self.assertIsNone(replay["summary"]["fuel_used_kg"])

    def test_obstacle_stops_climb_inside_a_step(self):
        shelf = Box((-1, -1, -75.001), (1, 1, -75))
        replay = simulate_demo(name="climb", world=ObstacleMap(DEMO_MAP.bounds, (shelf,)))
        self.assertIn("прерван", replay["status"])
        self.assertLess(replay["summary"]["duration_s"], 30)
        self.assertAlmostEqual(-replay["samples"][-1]["ned"][2], 72, delta=0.05)

    def test_input_mass_changes_response_to_same_force(self):
        def fixed_force(body):
            from dataclasses import replace
            case = free_fall(body, altitude_m=120, duration_s=1)
            return replace(case, loads=lambda t, x: (np.array([4., 0, 0]), np.zeros(3)))
        with patch.dict("diam20k.map_flight.SCENARIOS", {"test": fixed_force}):
            for mass in (2, 4):
                body = RigidBody(mass, np.eye(3))
                replay = simulate_demo(name="test", body=body)
                self.assertAlmostEqual(replay["samples"][-1]["ned"][0], 2/mass)
                self.assertAlmostEqual(replay["samples"][0]["sensors"]["accel_m_s2"][0], 4/mass)

    def test_free_fall_instrument_clocks_and_pressure(self):
        replay = simulate_demo(name="free-fall", field_ned_ut=(10, 2, 40), instrument_rates=True)
        streams = replay["sensor_series"]
        for channel, hz in [("gyro_rad_s", 200), ("accel_m_s2", 200), ("mag_ut", 50),
                            ("dynamic_pressure_pa", 50), ("baro_pa", 25), ("gnss_deg_deg_m", 5)]:
            np.testing.assert_allclose(np.array(streams[channel], dtype=object)[:, 0].astype(float),
                                       np.arange(4*hz + 1) / hz)
        first, last = replay["samples"][0]["sensors"], replay["samples"][-1]["sensors"]
        np.testing.assert_allclose(last["accel_m_s2"], 0)
        np.testing.assert_allclose(last["gyro_rad_s"], 0)
        np.testing.assert_allclose(last["mag_ut"], [10, 2, 40])
        self.assertGreater(last["baro_pa"], first["baro_pa"])
        self.assertAlmostEqual(last["gnss_velocity_ned_m_s"][2], GRAVITY_M_S2 * 4)
        np.testing.assert_allclose(last["gnss_deg_deg_m"][2],
                                   DEMO_ORIGIN.ellipsoid_height_m + 120 - GRAVITY_M_S2*8)

    def test_rotated_field_wind_and_specific_force(self):
        state = level_state()
        state[6:10] = [2**-0.5, 0, 0, 2**-0.5]
        state[10:13] = [0.1, -0.2, 0.3]
        sensor = readings(state, [0, 0, -2*GRAVITY_M_S2], TEST_BODY, DEMO_ORIGIN,
                          [0, 5, 0], [20, 0, 40])
        np.testing.assert_allclose(sensor["gyro_rad_s"], state[10:13])
        np.testing.assert_allclose(sensor["mag_ut"], [0, -20, 40], atol=1e-12)
        np.testing.assert_allclose(sensor["accel_m_s2"], [0, 0, -GRAVITY_M_S2])
        self.assertAlmostEqual(sensor["baro_pa"], 101325)
        self.assertAlmostEqual(sensor["dynamic_pressure_pa"], 0.5 * 101325/(287.05*288.15)*25)


class FuelTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "fuel.csv"

    def test_flow_knots_and_partial_intervals(self):
        self.path.write_text("time_s,flow_kg_s\n0,0\n2,0.02\n4,0\n")
        np.testing.assert_allclose(fuel_usage(self.path, [1, 3, 4])[0], [0, 0.03, 0.035])
        replay = simulate_demo(name="free-fall", fuel_log=self.path)
        self.assertIsNone(replay["fuel_estimate"])
        self.assertAlmostEqual(replay["summary"]["fuel_used_kg"], 0.04)
        self.assertAlmostEqual(replay["samples"][-1]["fuel_used_kg"], 0.04)

    def test_volumetric_log_keeps_litres(self):
        self.path.write_text("time_s,flow_l_h\n0,0.9\n4,0.9\n")
        result = simulate_demo(name="free-fall", fuel_log=self.path)["summary"]
        self.assertAlmostEqual(result["fuel_used_l"], 0.001)
        self.assertIsNone(result["fuel_used_kg"])

    def test_brake_power_unit_conversion(self):
        torque = 1000 / (2*np.pi*60/60)  # 1 кВт при 60 об/мин.
        self.path.write_text(f"time_s,rpm,torque_nm,bsfc_g_kwh\n0,60,{torque},300\n3600,60,{torque},300\n")
        np.testing.assert_allclose(fuel_usage(self.path, [0, 3600], engine=True)[0], [0, .3])

    def test_invalid_logs_are_not_extrapolated(self):
        for rows in ("0,0\n1,0", "0,1\n0,2", "0,nan\n4,1", "0,1\n4,-1"):
            self.path.write_text("time_s,flow_kg_s\n" + rows)
            with self.assertRaises(ValueError):
                fuel_usage(self.path, [0, 4])

    def test_battery_charge_change_and_partial_interval(self):
        self.path.write_text("time_s,voltage_v,current_a\n0,12,1\n3600,12,-1\n")
        result = battery_energy(self.path, 900, 2700)
        self.assertAlmostEqual(result["discharged_wh"], .75)
        self.assertAlmostEqual(result["charged_wh"], .75)
        with self.assertRaises(ValueError):
            battery_energy(self.path, 0, 3601)

    def test_explicit_fuel_estimate_and_geographic_fallback(self):
        r = simulate_demo(MapOrigin(0, 0, 0), name="free-fall", fuel_rate_l_h=0.9)
        self.assertIsNone(r["relief"])
        self.assertEqual(r["fuel_estimate"]["source"], "user")
        np.testing.assert_allclose(r["fuel_estimate"]["total_ml"], [1, 1])
        with self.assertRaises(ValueError):
            simulate_demo(fuel_rate_l_h=-1)

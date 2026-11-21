import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from diam20k.flight_audit import fuel_inventory, lab_drive, work_balance
from diam20k.instruments import downward_range
from diam20k.inertial_lab import reconstruct_recording
from diam20k.map_flight import simulate_demo
from diam20k.obstacle_map import Box, ObstacleMap
from diam20k.relief import TerrainGrid


class SurfaceTests(unittest.TestCase):
    def test_crossing_between_high_endpoints_and_tangent(self):
        grid = TerrainGrid(0, 0, 10, [[0, 0], [10, 10], [0, 0]])
        start, end = np.array([0, 5, -6]), np.array([20, 5, -6])
        self.assertAlmostEqual(grid.first_hit(start, end), .3)
        self.assertAlmostEqual(grid.first_hit(end, start), .3)
        self.assertAlmostEqual(grid.clearance_bound(start, end, (0, 0, 0)), -4)
        self.assertAlmostEqual(grid.first_hit([0, 5, -10], [20, 5, -10]), .5)

    def test_nonplanar_cell_hits_diagonal_and_overhanging_hull(self):
        grid = TerrainGrid(0, 0, 10, [[0, 8], [8, 0]])
        self.assertAlmostEqual(grid.height(5, 5), 8)
        self.assertAlmostEqual(grid.height(10, 10), 0)
        self.assertAlmostEqual(grid.first_hit([1, 1, -4], [9, 9, -4]), 1.5/8)
        a = np.array([2., 2., -5.])
        self.assertGreater(grid.clearance_bound(a, a, (0, 0, 0)), 0)
        self.assertLess(grid.clearance_bound(a, a, (1, 1, .5)), 0)
        with self.assertRaises(ValueError):
            grid.height(11, 3)

    def test_lidar_uses_nearest_of_roof_and_ground(self):
        grid = TerrainGrid(-50, -50, 100, [[2, 2], [2, 2]])
        world = ObstacleMap(Box((-50, -50, -50), (50, 50, 0)), (Box((-2,-2,-7), (2,2,0)),))
        for east, expected in [(0, 13), (10, 18)]:
            self.assertEqual(downward_range(world, np.array([0., east, -20.]), np.eye(3), grid), expected)

    def test_real_relief_rejects_step_and_keeps_last_valid_sample(self):
        replay = simulate_demo(surface="utah", hull_m=(2,4,1))
        self.assertIn("рельеф", replay["status"])
        self.assertGreater(replay["summary"]["first_rejected_s"], replay["samples"][-1]["time"])
        self.assertGreater(replay["summary"]["min_agl_m"], 3)
        self.assertNotEqual(replay["samples"][-1]["ground_height_m"], 0)

    def test_eight_recorded_passes_on_uneven_ground(self):
        replay = simulate_demo(surface="lab", hull_m=(2,4,1))
        self.assertEqual(replay["status"], "Маршрут завершён")
        self.assertEqual([p["direction"] for p in replay["passes"]],
                         ["Восточнее", "Западнее", "Сверху"]*2 + ["Восточнее","Западнее"])
        self.assertTrue(all(p["face_gap_m"] > 3 for p in replay["passes"]))
        rows = replay["samples"]
        self.assertTrue(any(r["ground_height_m"] > 5 for r in rows))
        self.assertEqual(rows[0]["agl_m"], rows[0]["sensors"]["range_down_m"])


class AuditTests(unittest.TestCase):
    def test_work_and_energy_match_at_irregular_times(self):
        samples = [{"time": t, "external_power_w": 2*t, "mechanical_energy_j": 7+t*t} for t in [0,.25,2,4]]
        result = work_balance(samples)
        self.assertEqual(result["max_residual_j"], 0)
        self.assertEqual(result["external_work_j"], 16)

    def test_load_table_integrates_internal_crossings_and_overload(self):
        config = {"base_shaft_w": 0, "transmission_efficiency": 1, "auxiliary_w": 12,
                  "shaft_w": [0, 50, 100], "flow_l_h": [0, 1, 3]}
        with TemporaryDirectory() as folder:
            path = Path(folder, "drive.json")
            path.write_text(json.dumps(config))
            rows = [{"time": 0., "external_power_w": -50.}, {"time": 4., "external_power_w": 150.}]
            result = lab_drive(path, rows)
            self.assertAlmostEqual(result["unserved_wh"], 25/3600)
            self.assertAlmostEqual(result["auxiliary_wh"], 48/3600)
            self.assertAlmostEqual(rows[-1]["fuel_used_l"], 5.5/3600)
            self.assertEqual(rows[-1]["fuel_rate_l_h"], 3)

    def test_inventory_reports_shortfall_without_negative_reserve(self):
        rows = [{"time": 0}, {"time": 3600}]
        report = fuel_inventory(.5, rows, {"fuel_used_l": None}, {"rate_l_h": [.3,.6]})
        np.testing.assert_allclose(report["remaining_l"], [0, .2])
        np.testing.assert_allclose(report["shortfall_l"], [0, .1])
        with self.assertRaises(ValueError):
            fuel_inventory(.5, rows, {"fuel_used_l": None}, None)

    def test_outage_disables_only_gnss_and_imu_export_reproduces_display(self):
        replay = simulate_demo(name="free-fall", gnss_outage=(1., 3.))
        series = replay["sensor_series"]
        self.assertEqual(replay["summary"]["gnss_received"], 11)
        self.assertEqual(replay["summary"]["gnss_expected"], 21)
        by_time = {round(r["time"], 2): r for r in replay["samples"]}
        for t in (1, 1.55, 2.99):
            self.assertIsNone(by_time[t]["sensors"]["gnss_deg_deg_m"])
            self.assertAlmostEqual(by_time[t]["fix_age_s"], t-.8)
        self.assertAlmostEqual(by_time[3]["fix_age_s"], 0)
        self.assertIsNotNone(by_time[3]["sensors"]["gnss_deg_deg_m"])
        self.assertEqual(by_time[2]["sensors"]["accel_m_s2"], series["accel_m_s2"][400][1])
        self.assertAlmostEqual(by_time[.15]["sensors"]["gnss_deg_deg_m"][2], by_time[0]["geo"][2])
        self.assertNotEqual(by_time[.15]["geo"][2], by_time[0]["geo"][2])

    def test_outage_at_start_has_no_previous_fix(self):
        replay = simulate_demo(name="free-fall", gnss_outage=(0., 4.))
        self.assertIsNone(replay["samples"][0]["fix_age_s"])
        self.assertIsNone(replay["samples"][0]["ins_increment_error_m"])
        self.assertEqual(replay["summary"]["gnss_received"], 1)

    def test_postflight_reconstruction_uses_both_fixes_without_extrapolation(self):
        t = np.arange(7, dtype=float)
        offsets = np.column_stack([t*t, t, 0*t])
        corrected = reconstruct_recording(t, offsets, [1., 5.], [[10., 21., 3.], [18., 25., 3.]])
        self.assertTrue(np.isnan(corrected[[0, 6]]).all())
        np.testing.assert_allclose(corrected[[1, 5]], [[10, 21, 3], [18, 25, 3]])
        # В середине поправка по N равна среднему двух невязок, (9−7)/2.
        np.testing.assert_allclose(corrected[3], [10, 23, 3])
        self.assertTrue(np.isnan(reconstruct_recording(t, offsets, [], [])).all())

    def test_invalid_intervals(self):
        for bounds in [(0, 5), (2, 1), (-1, 1), (1, float("nan"))]:
            with self.assertRaises(ValueError):
                simulate_demo(name="free-fall", gnss_outage=bounds)

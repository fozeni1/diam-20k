import unittest
from unittest.mock import patch

import numpy as np

from diam20k.course import plan_course
from diam20k.geography import MapOrigin
from diam20k.map_flight import flight_summary, simulate_demo
from diam20k.obstacle_map import Box, DEMO_GOAL_NED, DEMO_MAP, DEMO_START_NED, ObstacleMap


class MapFlightTests(unittest.TestCase):
    def test_local_georeferencing_and_altitude_across_antimeridian(self):
        origin = MapOrigin(55.75, 179.9999, 150)
        ned = (800, 200, -20)
        latitude, longitude, height = origin.to_geodetic(ned)
        self.assertGreater(latitude, origin.latitude_deg)
        self.assertLess(longitude, -179)
        self.assertEqual(height, 170)
        np.testing.assert_allclose(origin.to_ned(latitude, longitude, height), ned, atol=1e-8)

    def test_eight_obstacles_in_wind_and_calm(self):
        windy = simulate_demo()
        calm = simulate_demo(wind=False)
        safe = DEMO_MAP.with_margin(3)
        for replay in (windy, calm):
            self.assertEqual(replay["status"], "Маршрут завершён")
            np.testing.assert_allclose(replay["fuel_estimate"]["total_ml"], [200/3, 100])
            self.assertIsNotNone(replay["relief"])
            self.assertEqual(replay["summary"]["passed"], 8)
            self.assertEqual(replay["summary"]["total"], 8)
            self.assertEqual(replay["summary"]["duration_s"], 200)
            self.assertGreater(replay["summary"]["distance_m"], 4000)
            self.assertEqual(len(replay["blockers"]), 8)
            rows = replay["samples"]
            self.assertEqual(rows[-1]["time"], 200)
            np.testing.assert_allclose(rows[-1]["ned"], DEMO_GOAL_NED, atol=0.01)
            for a, b in zip(rows, rows[1:]):
                self.assertTrue(safe.is_segment_free(a["ned"], b["ned"]))
            passes = []
            for i in replay["blockers"]:
                box = DEMO_MAP.obstacles[i]
                self.assertTrue(box.intersects_segment(DEMO_START_NED, DEMO_GOAL_NED))
                inside = [r["ned"] for r in rows
                          if box.minimum_ned[0] <= r["ned"][0] <= box.maximum_ned[0]]
                self.assertTrue(inside)
                if all(p[1] > box.maximum_ned[1] for p in inside):
                    passes.append("east")
                elif all(p[1] < box.minimum_ned[1] for p in inside):
                    passes.append("west")
                elif all(box.minimum_ned[1] <= p[1] <= box.maximum_ned[1]
                         and p[2] < box.minimum_ned[2] for p in inside):
                    passes.append("above")
                else:
                    self.fail(f"Не удалось определить проход объекта {i}")
            self.assertEqual(passes, ["east", "west", "above", "east",
                                      "west", "above", "east", "west"])
        self.assertGreater(windy["samples"][800]["ned"][1] - calm["samples"][800]["ned"][1], 20)
        self.assertTrue(all(r["wind_east"] == 0 for r in calm["samples"]))

    def test_course_changes_when_obstacle_moves(self):
        for low, high, sign in [(-280, 55, 1), (-55, 280, -1)]:
            world = ObstacleMap(DEMO_MAP.bounds, (Box((350, low, -155), (440, high, 0)),))
            route = plan_course(world, DEMO_START_NED, DEMO_GOAL_NED)
            offset, _, _ = route.motion(400, 20)
            self.assertGreater(sign * offset[0], 55)

    def test_blocked_course_reports_failure(self):
        wall = Box((200, -600, -180), (210, 600, 0))
        with self.assertRaisesRegex(ValueError, "Не найден свободный проход"):
            plan_course(ObstacleMap(DEMO_MAP.bounds, (wall,)), DEMO_START_NED, DEMO_GOAL_NED)

    def test_unexpected_wall_stops_replay(self):
        empty = ObstacleMap(DEMO_MAP.bounds)
        route = plan_course(empty, DEMO_START_NED, DEMO_GOAL_NED)
        wall = Box((10.4, -5, -60), (10.6, 5, 0))
        with patch("diam20k.map_flight.plan_course", return_value=route), \
                patch("diam20k.map_flight.DEMO_MAP", ObstacleMap(empty.bounds, (wall,))):
            replay = simulate_demo()
        self.assertIn("Полёт прерван", replay["status"])
        self.assertAlmostEqual(replay["samples"][-1]["ned"][0], 7.0)
        summary = replay["summary"]
        self.assertEqual((summary["passed"], summary["total"]), (0, 1))
        self.assertAlmostEqual(summary["duration_s"], 0.35)
        self.assertAlmostEqual(summary["distance_m"], 7.0)
        self.assertAlmostEqual(summary["closest_sample_m"], 3.4)

    def test_single_frame_and_empty_map(self):
        summary = flight_summary([{"time": 0, "ned": DEMO_START_NED}],
                                 ObstacleMap(DEMO_MAP.bounds), [])
        self.assertEqual(summary["distance_m"], 0)
        self.assertEqual(summary["duration_s"], 0)
        self.assertIsNone(summary["closest_sample_m"])

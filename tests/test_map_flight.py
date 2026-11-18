import unittest
from unittest.mock import patch

import numpy as np

from diam20k.geography import MapOrigin
from diam20k.map_flight import simulate_demo
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

    def test_demo_climbs_over_obstacle_and_reaches_goal(self):
        self.assertFalse(DEMO_MAP.is_segment_free(DEMO_START_NED, DEMO_GOAL_NED))
        samples = simulate_demo()["samples"]
        start, finish = samples[0], samples[-1]
        self.assertEqual(start["time"], 0)
        self.assertEqual(finish["time"], 40)
        np.testing.assert_allclose(start["ned"], DEMO_START_NED, atol=1e-8)
        np.testing.assert_allclose(finish["ned"], DEMO_GOAL_NED, atol=1e-6)
        self.assertAlmostEqual(min(row["ned"][2] for row in samples), -80, places=6)
        self.assertLess(samples[200]["attitude"][2][0], -0.2)
        self.assertGreater(samples[600]["attitude"][2][0], 0.2)
        for start, finish in zip(samples, samples[1:]):
            self.assertTrue(DEMO_MAP.is_segment_free(start["ned"], finish["ned"]))

    def test_collision_keeps_last_safe_sample(self):
        # At 20 m/s the first simulated step spans this thin obstacle.
        wall = Box((0.4, -5, -60), (0.6, 5, 0))
        with patch("diam20k.map_flight.DEMO_MAP", ObstacleMap(DEMO_MAP.bounds, (wall,))):
            replay = simulate_demo()
        self.assertIn("Полёт прерван", replay["status"])
        self.assertEqual(len(replay["samples"]), 1)

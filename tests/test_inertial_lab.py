import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from diam20k.instruments import compare_recording, downward_range
import numpy as np

from diam20k.inertial_lab import bmi088_samples, displacement
from diam20k.scenarios import GRAVITY_M_S2 as G
from diam20k.obstacle_map import Box, ObstacleMap
from diam20k.map_flight import simulate_demo
from diam20k.scenario_report import build_report
from diam20k.scenarios import hover


class LabTests(unittest.TestCase):
    def test_rotating_imu_preserves_horizontal_acceleration(self):
        t = np.arange(201) / 200
        rate = np.pi / 2
        gyro = np.tile([0, 0, rate], (len(t), 1))
        force = np.column_stack([np.cos(rate*t), -np.sin(rate*t), np.full_like(t, -G)])
        trajectory = displacement(t, gyro, force, np.eye(3), np.zeros(3))
        np.testing.assert_allclose(trajectory[:, 0], t*t/2, atol=1e-12)
        np.testing.assert_allclose(trajectory[:, 1:], 0, atol=1e-12)

    def test_accelerometer_bias_has_quadratic_drift(self):
        t = np.arange(401) / 200
        gyro = np.zeros((len(t), 3))
        force = np.tile([0.1, 0, -G], (len(t), 1))
        np.testing.assert_allclose(displacement(t, gyro, force, np.eye(3), np.zeros(3))[-1],
                                   [0.2, 0, 0], atol=1e-12)

    def test_quantization_and_saturation(self):
        x = np.full((10, 3), 10000.)
        g, a = bmi088_samples(x, x, np.random.default_rng(2))
        np.testing.assert_allclose(g, np.deg2rad(250) * 32767/32768)
        np.testing.assert_allclose(a, 6*G * 32767/32768)

    def test_report_repeats_and_exposes_both_estimates(self):
        scenario = hover(duration_s=0.5)
        a = build_report(scenario, inertial=True)
        self.assertEqual(a, build_report(scenario, inertial=True))
        self.assertLess(a['tiles'][-2]['value'], 1e-12)
        self.assertGreater(a['tiles'][-1]['value'], 0)

    def test_wing_clearance_changes_outcome(self):
        world = ObstacleMap(Box((-10,-10,-180), (10,10,0)),
                            (Box((-5,4,-75.1), (5,6,-75)),))
        self.assertIn('заверш', simulate_demo(name='climb', world=world)['status'])
        self.assertIn('прерван', simulate_demo(name='climb', world=world, hull_m=(2,4,1))['status'])
        # При повороте на 90 градусов длинная сторона оказывается вдоль севера.
        start, end = (0,0,-70), (0,0,-80)
        self.assertTrue(world.with_margin(3).envelope_free(start, end, (2,0.8,0.5)))

    def test_reference_error_on_irregular_observation_times(self):
        samples = [{"time": t, "ned": [0, 0, -2*t], "velocity": [1, 0, -2]} for t in [0, 2]]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "record.csv"
            header = "time_s,height_above_map_m,vn_m_s,ve_m_s,vd_m_s\n"
            path.write_text(header + "0.5,2,1,3,2\n1.5,4,1,3,2\n")
            comparison = compare_recording(path, samples)
            self.assertAlmostEqual(comparison["height_rmse_m"], 1)
            self.assertAlmostEqual(comparison["velocity_vector_rmse_m_s"], 5)
            path.write_text(header + "0,0,0,0,0\n3,0,0,0,0\n")
            with self.assertRaises(ValueError):
                compare_recording(path, samples)

    def test_lidar_returns_first_surface_or_no_echo(self):
        world = ObstacleMap(Box((-100,-100,-100), (100,100,0)), (Box((-4,-4,-10),(4,4,0)),))
        self.assertEqual(downward_range(world, np.array([0.,0.,-30.]), np.eye(3)), 20)
        self.assertEqual(downward_range(world, np.array([8.,0.,-30.]), np.eye(3)), 30)
        self.assertIsNone(downward_range(world, np.array([8.,0.,-41.]), np.eye(3)))
        tilted = np.array([[.8,0,.6],[0,1,0],[-.6,0,.8]])
        self.assertEqual(downward_range(world, np.array([8.,0.,-24.]), tilted), 30)

    def test_map_inertial_replay_uses_recorded_streams(self):
        replay = simulate_demo(name='free-fall', inertial_check=True)
        self.assertEqual(replay['samples'][0]['ins_error_m'], 0)
        self.assertGreater(replay['summary']['ins_final_error_m'], 0)
        self.assertEqual(len(replay['sensor_series']['range_down_m']), 201)

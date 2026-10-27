"""Analytical checks on synthetic bodies, NOT DIAM-20K calibration."""

import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from diam20k import RigidBody, air_data, point_load, rhs, rotation, step_rk4
from diam20k.parameters import load_body


def initial_state():
    x = np.zeros(13)
    x[6] = 1
    return x


class MechanicsTests(unittest.TestCase):
    def setUp(self):
        self.body = RigidBody(2, np.diag([2., 3., 4.]))

    def test_free_fall_matches_analytical_trajectory(self):
        x = initial_state()
        for i in range(100):
            x = step_rk4(x, i*.01, .01, self.body, lambda t, s: (np.zeros(3), np.zeros(3)))
        assert_allclose(x[:6], [0, 0, 9.80665/2, 0, 0, 9.80665], atol=1e-12)

    def test_hover_equilibrium(self):
        derivative = rhs(initial_state(), self.body, [0, 0, -2*9.80665], [0, 0, 0])
        assert_allclose(derivative, 0, atol=1e-12)

    def test_rotation_and_quaternion_sign(self):
        q = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        r = rotation(q)
        assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-14)
        assert_allclose(r.T @ r, np.eye(3), atol=1e-14)
        assert_allclose(rotation(-np.array(q)), r)

    def test_tilted_free_fall_acceleration_is_down_in_ned(self):
        x = initial_state()
        x[6:10] = [np.sqrt(.5), np.sqrt(.5), 0, 0]
        derivative = rhs(x, self.body, [0, 0, 0], [0, 0, 0])
        assert_allclose(rotation(x[6:10]) @ derivative[3:6], [0, 0, 9.80665], atol=1e-12)

    def test_off_center_lift_sign(self):
        force, moment = point_load([2, 3, 0], [0, 0, -10], [0, 0, 1])
        assert_allclose(force, [0, 0, -10])
        assert_allclose(moment, [-30, 20, 1])

    def test_rotation_matches_constant_rate(self):
        x = initial_state()
        x[10] = 1
        for i in range(100):
            x = step_rk4(x, i*.01, .01, self.body, lambda t, s: ([0, 0, 0], [0, 0, 0]), gravity_m_s2=0)
        assert_allclose(x[6:10], [np.cos(.5), np.sin(.5), 0, 0], atol=1e-10)

    def test_torque_free_conservation_with_full_inertia(self):
        body = RigidBody(2, [[2, .1, .2], [.1, 3, .15], [.2, .15, 4]])
        x = initial_state()
        x[3:6] = [2, -1, 3]
        x[10:13] = [.3, .5, .7]
        momentum = body.inertia_kg_m2 @ x[10:13]
        energy = .5 * x[10:13] @ momentum
        velocity_ned = x[3:6].copy()
        for i in range(200):
            x = step_rk4(x, i*.005, .005, body, lambda t, s: ([0, 0, 0], [0, 0, 0]), gravity_m_s2=0)
        rot = rotation(x[6:10])
        assert_allclose(rot @ (body.inertia_kg_m2 @ x[10:13]), momentum, atol=1e-9)
        assert_allclose(.5*x[10:13] @ body.inertia_kg_m2 @ x[10:13], energy, atol=1e-9)
        assert_allclose(rot @ x[3:6], velocity_ned, atol=1e-9)
        assert_allclose(np.linalg.norm(x[6:10]), 1, atol=1e-14)

    def test_wind_and_zero_airspeed(self):
        x = initial_state()
        x[3] = 20
        result = air_data(x, [5, 0, 0], 1.2)
        self.assertEqual(result['speed_m_s'], 15)
        self.assertEqual(result['dynamic_pressure_pa'], 135)
        result = air_data(x, [20, 0, 0], 1.2)
        self.assertIsNone(result['alpha_rad'])
        self.assertIsNone(result['beta_rad'])

    def test_invalid_parameters_rejected(self):
        for mass, inertia in [(0, np.eye(3)), (2, np.diag([-1, 2, 3])),
                              (2, np.diag([1, 1, 5])), (2, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])]:
            with self.assertRaises(ValueError):
                RigidBody(mass, inertia)
        with self.assertRaises(ValueError):
            rotation([0, 0, 0, 0])

    def test_missing_aircraft_parameters_not_invented(self):
        path = Path(__file__).resolve().parents[1] / 'configs' / 'diam20k.json'
        with self.assertRaisesRegex(ValueError, 'mass_kg, inertia_cg_kg_m2'):
            load_body(path)


if __name__ == '__main__':
    unittest.main()

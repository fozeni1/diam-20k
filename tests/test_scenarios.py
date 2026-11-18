"""Сценарии на проверочном теле: сравнение с точными решениями, НЕ калибровка ДИАМ-20К."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from diam20k.scenarios import (GRAVITY_M_S2, SCENARIOS, TEST_BODY, TEST_ROTORS_BODY_M, Scenario,
                               climb, deviation, free_fall, hover, hover_thrusts, level_state, main,
                               rotor_loads, run, wind)

WEIGHT_N = TEST_BODY.mass_kg * GRAVITY_M_S2
# Те же винты при центре масс, смещённом на 0,2 м вперёд и на 0,1 м вправо.
SHIFTED_ROTORS = ((0.3, 0.6, 0.1), (0.3, -0.8, 0.1), (-0.7, 0.6, 0.1), (-0.7, -0.8, 0.1))


def no_loads(time_s, state):
    return np.zeros(3), np.zeros(3)


class ScenarioTests(unittest.TestCase):
    def test_front_right_rotor_lifts_nose_and_right_wing(self):
        force, moment = rotor_loads([10, 0, 0, 0])
        assert_allclose(force, [0, 0, -10])
        assert_allclose(moment, [-7, 5, 0])

    def test_hover_thrusts_balance_weight_and_moment(self):
        assert_allclose(hover_thrusts(TEST_BODY), WEIGHT_N / 4, rtol=1e-12)
        thrusts = hover_thrusts(TEST_BODY, SHIFTED_ROTORS)
        # Больше нагружены винты, которые ближе к центру масс: передние и правые.
        self.assertTrue(thrusts[0] > thrusts[1] > thrusts[2] > thrusts[3] > 0)
        force, moment = rotor_loads(thrusts, SHIFTED_ROTORS)
        assert_allclose(force, [0, 0, -WEIGHT_N], atol=1e-12)
        assert_allclose(moment, 0, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "отрицательной тяги"):
            hover_thrusts(TEST_BODY, [(1, 0.7, 0), (1, -0.7, 0), (2, 0.7, 0), (2, -0.7, 0)])
        with self.assertRaisesRegex(ValueError, "Равновесие невозможно"):
            hover_thrusts(TEST_BODY, [(1, 1, 0)] * 4)

    def test_hover_keeps_initial_state(self):
        for rotors in (TEST_ROTORS_BODY_M, SHIFTED_ROTORS):
            scenario = hover(rotors_body_m=rotors, duration_s=10)
            time_s, states = run(scenario)
            self.assertEqual(time_s[-1], 10)
            assert_allclose(states, np.tile(level_state(50), (len(time_s), 1)), atol=1e-9)
            self.assertLess(deviation(scenario, time_s, states), 1e-9)

    def test_free_fall_matches_formula(self):
        scenario = free_fall(duration_s=5)
        time_s, states = run(scenario)
        expected = level_state(1000)
        expected[2] += GRAVITY_M_S2 * 25 / 2
        expected[5] = GRAVITY_M_S2 * 5
        assert_allclose(states[-1], expected, atol=1e-9)
        self.assertLess(deviation(scenario, time_s, states), 1e-9)
        with self.assertRaisesRegex(ValueError, "ниже земли"):
            free_fall(altitude_m=100)

    def test_climb_follows_thrust_commands(self):
        scenario = climb(dt_s=0.02)
        self.assertEqual(scenario.duration_s, 30)
        # Висение, полка разгона, подъём с постоянной скоростью, полка торможения.
        for time_s, extra_m_s2 in [(0, 0), (7, 1), (15, 0), (23, -1), (30, 0)]:
            force, moment = scenario.loads(time_s, scenario.initial_state)
            expected_n = TEST_BODY.mass_kg * (GRAVITY_M_S2 + extra_m_s2)
            assert_allclose(force, [0, 0, -expected_n], atol=1e-12)
            assert_allclose(moment, 0, atol=1e-12)
        time_s, states = run(scenario)
        altitude_m = -states[:, 2]
        assert_allclose(states[-1], level_state(110), atol=1e-8)
        assert_allclose(states[time_s == 15, 5], -4, atol=1e-8)
        self.assertGreater(np.diff(altitude_m).min(), -1e-9)
        self.assertLess(altitude_m.max(), 110 + 1e-8)
        self.assertLess(deviation(scenario, time_s, states), 1e-8)

    def test_climb_without_plateau_and_cruise(self):
        scenario = climb(height_gain_m=2, climb_rate_m_s=1, dt_s=0.02)
        self.assertEqual(scenario.duration_s, 14)
        time_s, states = run(scenario)
        assert_allclose(states[-1], level_state(52), atol=1e-8)
        self.assertLess(deviation(scenario, time_s, states), 1e-8)

    def test_wind_drift_matches_formula(self):
        scenario = wind(duration_s=10)
        time_s, states = run(scenario)
        self.assertLess(deviation(scenario, time_s, states), 1e-8)
        east_speed = states[:, 4]
        self.assertGreater(np.diff(east_speed).min(), 0)
        self.assertTrue(4 < east_speed[-1] < 5)
        self.assertGreater(states[-1, 1], 20)
        assert_allclose(states[:, [0, 2]], [[0, -50]] * len(time_s), atol=1e-9)

        scenario = wind(wind_ned_m_s=(-3, 5, 0), duration_s=5)
        time_s, states = run(scenario)
        self.assertLess(deviation(scenario, time_s, states), 1e-8)
        self.assertLess(states[-1, 0], -1)
        self.assertGreater(states[-1, 1], 1)

        calm = wind(wind_ned_m_s=(0, 0, 0), duration_s=2)
        time_s, states = run(calm)
        assert_allclose(states, np.tile(level_state(50), (len(time_s), 1)), atol=1e-9)

    def test_runs_repeat_on_uniform_grid(self):
        scenario = free_fall(duration_s=1, dt_s=0.3)
        time_s, states = run(scenario)
        assert_allclose(np.diff(time_s), 0.25, atol=1e-15)
        self.assertEqual((len(time_s), time_s[0], time_s[-1]), (5, 0, 1))
        assert_array_equal(states[0], scenario.initial_state)
        repeat_time_s, repeat_states = run(free_fall(duration_s=1, dt_s=0.3))
        assert_array_equal(repeat_time_s, time_s)
        assert_array_equal(repeat_states, states)

    def test_listed_scenarios_have_exact_solutions(self):
        self.assertEqual(list(SCENARIOS), ["free-fall", "hover", "climb", "wind"])
        for name, make in SCENARIOS.items():
            scenario = make()
            self.assertEqual(scenario.name, name)
            assert_array_equal(scenario.exact(np.array([0.0]))[0], scenario.initial_state)

    def test_custom_scenario_without_formula(self):
        state = level_state(10)
        state[6] = 2
        scenario = Scenario("drop", "Падение без формулы", TEST_BODY, state, no_loads, 0.1, 0.05)
        self.assertEqual(scenario.initial_state[6], 1)
        time_s, states = run(scenario)
        self.assertIsNone(deviation(scenario, time_s, states))
        self.assertAlmostEqual(states[-1, 5], GRAVITY_M_S2 * 0.1)

    def test_invalid_parameters_rejected(self):
        for make in [
            lambda: Scenario("x", "x", TEST_BODY, np.zeros(12), no_loads, 1, 0.01),
            lambda: Scenario("x", "x", TEST_BODY, np.zeros(13), no_loads, 1, 0.01),
            lambda: Scenario("x", "x", TEST_BODY, level_state(), no_loads, -1, 0.01),
            lambda: hover(dt_s=0),
            lambda: hover(duration_s=float("nan")),
            lambda: climb(acceleration_m_s2=10),
            lambda: climb(height_gain_m=10),
            lambda: climb(ramp_s=5),
            lambda: rotor_loads([1, 1, 1, -1]),
            lambda: rotor_loads([1, 1, 1]),
            lambda: rotor_loads([1, 1], [(0, 1), (0, -1)]),
            lambda: wind(drag_area_m2=(0.2, -0.4, 0.6)),
            lambda: wind(wind_ned_m_s=(0, 5)),
            lambda: wind(density_kg_m3=0),
        ]:
            with self.assertRaises(ValueError):
                make()

    def test_command_line_reports_deviation(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main(["free-fall"])
        self.assertIn("не параметры ДИАМ-20К", output.getvalue())
        self.assertRegex(output.getvalue(),
                         r"free-fall: Свободное падение\. Время 10 с, высота 509\.668 м")
        self.assertNotIn("hover", output.getvalue())
        with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
            main(["nope"])
        self.assertIn("неизвестный сценарий: nope", errors.getvalue())


if __name__ == "__main__":
    unittest.main()

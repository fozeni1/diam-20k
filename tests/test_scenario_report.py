"""Отчёт с графиками: проверяются данные, которые получает страница, а не её внешний вид."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from diam20k.scenario_report import (DEVIATION_FLOOR, MAX_POINTS, PLACEHOLDER, build_report,
                                     euler_deg, main, render, sample_indices)
from diam20k.scenarios import GRAVITY_M_S2, TEST_BODY, Scenario, climb, free_fall, level_state, wind


def series(report, chart_title, label):
    """Ряд графика по названию графика и подписи ряда: тесты не зависят от порядка графиков."""
    chart = next(chart for chart in report["charts"] if chart["title"] == chart_title)
    return next(item["values"] for item in chart["series"] if item["label"] == label)


class ScenarioReportTests(unittest.TestCase):
    def test_euler_angles_follow_axis_conventions(self):
        half = np.radians(15)
        assert_allclose(euler_deg([1, 0, 0, 0]), [0, 0, 0], atol=1e-12)
        # Положительный крен опускает правое крыло, тангаж поднимает нос, курс уводит нос к востоку.
        assert_allclose(euler_deg([np.cos(half), np.sin(half), 0, 0]), [30, 0, 0], atol=1e-12)
        assert_allclose(euler_deg([np.cos(half), 0, np.sin(half), 0]), [0, 30, 0], atol=1e-12)
        assert_allclose(euler_deg([np.cos(half), 0, 0, np.sin(half)]), [0, 0, 30], atol=1e-12)

    def test_sampling_keeps_first_and_last_points(self):
        self.assertEqual(list(sample_indices(5)), [0, 1, 2, 3, 4])
        self.assertEqual(list(sample_indices(11, limit=4)), [0, 3, 6, 9, 10])
        picked = sample_indices(3001)
        self.assertLessEqual(len(picked), MAX_POINTS + 1)
        self.assertEqual((picked[0], picked[-1]), (0, 3000))

    def test_free_fall_series_match_formula(self):
        report = build_report(free_fall(duration_s=2))
        time_s = np.array(report["time"])
        self.assertEqual((time_s[0], time_s[-1]), (0, 2))
        for chart in report["charts"]:
            for item in chart["series"]:
                self.assertEqual(len(item["values"]), len(time_s))
        # Ряды округлены до семи значащих цифр, поэтому допуск относительный.
        assert_allclose(series(report, "Высота", "высота"), 1000 - GRAVITY_M_S2 * time_s**2 / 2,
                        rtol=1e-6)
        assert_allclose(series(report, "Скорость", "вверх"), -GRAVITY_M_S2 * time_s, rtol=1e-6)
        assert_allclose(series(report, "Внешние силы в осях корпуса", "вверх"), 0, atol=1e-12)
        gap = series(report, "Отклонение от точного решения", "наибольшее по 13 компонентам")
        self.assertGreaterEqual(min(gap), DEVIATION_FLOOR)
        self.assertLess(max(gap), 1e-9)
        tiles = {tile["label"]: tile["value"] for tile in report["tiles"]}
        self.assertAlmostEqual(tiles["Высота в конце"], 1000 - GRAVITY_M_S2 * 2)
        self.assertAlmostEqual(tiles["Скорость в конце"], GRAVITY_M_S2 * 2)
        self.assertLess(tiles["Наибольшее отклонение"], 1e-9)

    def test_force_and_drift_signs(self):
        report = build_report(wind(duration_s=2))
        # Воздух движется на восток: сопротивление толкает аппарат вправо и сносит на восток.
        self.assertAlmostEqual(series(report, "Внешние силы в осях корпуса", "вправо")[0], 6.125)
        self.assertGreater(series(report, "Смещение от точки старта", "на восток")[-1], 1)
        assert_allclose(series(report, "Ориентация", "крен"), 0, atol=1e-9)

        report = build_report(climb(dt_s=0.02))
        self.assertLessEqual(len(report["time"]), MAX_POINTS + 1)
        thrust_n = series(report, "Внешние силы в осях корпуса", "вверх")
        self.assertAlmostEqual(max(thrust_n), TEST_BODY.mass_kg * (GRAVITY_M_S2 + 1), places=5)
        self.assertAlmostEqual(series(report, "Высота", "высота")[-1], 110, places=5)

    def test_scenario_without_formula_has_no_deviation_chart(self):
        scenario = Scenario("drop", "Падение без формулы", TEST_BODY, level_state(10),
                            lambda time_s, state: (np.zeros(3), np.zeros(3)), 0.1, 0.05)
        report = build_report(scenario)
        self.assertEqual(report["description"], "")
        self.assertNotIn("Отклонение от точного решения", [c["title"] for c in report["charts"]])
        self.assertNotIn("Наибольшее отклонение", [tile["label"] for tile in report["tiles"]])

    def test_page_embeds_valid_data(self):
        html = render([free_fall(duration_s=1), wind(duration_s=1)])
        self.assertNotIn(PLACEHOLDER, html)
        start = html.index("const report = ") + len("const report = ")
        data = json.loads(html[start:html.index(";\n", start)])
        self.assertEqual([item["name"] for item in data["scenarios"]], ["free-fall", "wind"])
        # Данные не должны закрывать тег script: в странице он один.
        self.assertEqual(html.count("</script>"), 1)

    def test_command_line_writes_report(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "report.html"
            output = io.StringIO()
            with redirect_stdout(output):
                main(["free-fall", "--output", str(target)])
            self.assertEqual(output.getvalue().strip(), str(target.resolve()))
            self.assertIn("Свободное падение", target.read_text(encoding="utf-8"))
        with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
            main(["nope"])
        self.assertIn("неизвестный сценарий: nope", errors.getvalue())


if __name__ == "__main__":
    unittest.main()

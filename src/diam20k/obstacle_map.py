"""Синтетическая карта для учебного симулятора DIAM-20K.

Координаты (N, E, D) в метрах: север, восток, вниз. Земля на D = 0.

Сохраните файл в src/diam20k/obstacle_map.py. Для проверки шага модели:

    from diam20k.obstacle_map import DEMO_MAP

    if not DEMO_MAP.is_segment_free(state[:3], next_state[:3]):
        # Между состояниями отрезок пересёк препятствие или границу карты.
        ...
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite


PointNED = tuple[float, float, float]


def _point_ned(position: Iterable[float]) -> PointNED:
    try:
        north, east, down = map(float, position)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Нужны три конечные координаты (N, E, D)") from exc

    if not all(map(isfinite, (north, east, down))):
        raise ValueError("Нужны три конечные координаты (N, E, D)")
    return north, east, down


@dataclass(frozen=True)
class Box:
    """Параллелепипед с гранями вдоль осей NED."""

    minimum_ned: PointNED
    maximum_ned: PointNED

    def __post_init__(self) -> None:
        minimum = _point_ned(self.minimum_ned)
        maximum = _point_ned(self.maximum_ned)
        if any(low >= high for low, high in zip(minimum, maximum)):
            raise ValueError("Минимум должен быть меньше максимума по каждой оси")
        object.__setattr__(self, "minimum_ned", minimum)
        object.__setattr__(self, "maximum_ned", maximum)

    def _intersects_segment(self, start: PointNED, end: PointNED) -> bool:
        # Для каждой оси сужаем интервал параметра t отрезка start + t * (end - start).
        first, last = 0.0, 1.0
        for origin, target, low, high in zip(start, end, self.minimum_ned, self.maximum_ned):
            direction = target - origin
            if direction == 0:
                if origin < low or origin > high:
                    return False
                continue

            enter = (low - origin) / direction
            exit_ = (high - origin) / direction
            first = max(first, min(enter, exit_))
            last = min(last, max(enter, exit_))
            if first > last:
                return False
        return True


@dataclass(frozen=True)
class ObstacleMap:
    """Ограниченная область с неподвижными препятствиями."""

    bounds: Box
    obstacles: tuple[Box, ...] = ()

    def _inside_bounds(self, position: PointNED) -> bool:
        return all(
            low < coordinate < high
            for coordinate, low, high in zip(
                position, self.bounds.minimum_ned, self.bounds.maximum_ned
            )
        )

    def is_position_free(self, position_ned: Iterable[float]) -> bool:
        """Проверяет положение центра аппарата. Касание грани считается столкновением."""
        position = _point_ned(position_ned)
        return self.is_segment_free(position, position)

    def is_segment_free(
        self, start_ned: Iterable[float], end_ned: Iterable[float]
    ) -> bool:
        """Проверяет прямой отрезок между положениями центра аппарата."""
        start = _point_ned(start_ned)
        end = _point_ned(end_ned)
        if not (self._inside_bounds(start) and self._inside_bounds(end)):
            return False
        return not any(box._intersects_segment(start, end) for box in self.obstacles)


DEMO_START_NED: PointNED = (0.0, 0.0, -20.0)
DEMO_GOAL_NED: PointNED = (800.0, 0.0, -20.0)

DEMO_MAP = ObstacleMap(
    bounds=Box((-100.0, -250.0, -120.0), (900.0, 450.0, 0.0)),
    obstacles=(
        Box((180.0, -45.0, -36.0), (250.0, 60.0, 0.0)),
        Box((380.0, 110.0, -54.0), (460.0, 200.0, 0.0)),
        Box((540.0, -145.0, -28.0), (630.0, -55.0, 0.0)),
        Box((670.0, 65.0, -45.0), (740.0, 155.0, 0.0)),
    ),
)

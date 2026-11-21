"""Геометрия учебного полигона в метрах NED. Земля находится на D = 0."""

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

    def expanded(self, margins: PointNED) -> "Box":
        return Box(tuple(v - d for v, d in zip(self.minimum_ned, margins)),
                   tuple(v + d for v, d in zip(self.maximum_ned, margins)))

    def intersects_segment(self, start: PointNED, end: PointNED) -> bool:
        return self.entry_fraction(start, end) is not None

    def entry_fraction(self, start: PointNED, end: PointNED) -> float | None:
        # Для каждой оси сужаем интервал параметра t отрезка start + t * (end - start).
        first, last = 0.0, 1.0
        for origin, target, low, high in zip(start, end, self.minimum_ned, self.maximum_ned):
            direction = target - origin
            if direction == 0:
                if origin < low or origin > high:
                    return None
                continue

            enter = (low - origin) / direction
            exit_ = (high - origin) / direction
            first = max(first, min(enter, exit_))
            last = min(last, max(enter, exit_))
            if first > last:
                return None
        return first


@dataclass(frozen=True)
class ObstacleMap:
    """Ограниченная область с неподвижными препятствиями."""

    bounds: Box
    obstacles: tuple[Box, ...] = ()

    def with_margin(self, metres: float) -> "ObstacleMap":
        """Расширяет препятствия и сдвигает границы полигона внутрь."""
        if not isfinite(metres) or metres < 0:
            raise ValueError("Запас должен быть конечным и неотрицательным")

        return ObstacleMap(self.bounds.expanded((-metres,) * 3),
                           tuple(box.expanded((metres,) * 3) for box in self.obstacles))

    def envelope_free(self, start, end, extents):
        """Свободен ли отрезок с заданными полуразмерами оболочки в NED."""
        for p in (start, end):
            for axis, half in enumerate(extents):
                if (p[axis] - half <= self.bounds.minimum_ned[axis] or
                        p[axis] + half >= self.bounds.maximum_ned[axis]):
                    return False
        return not any(box.expanded(extents).intersects_segment(start, end)
                       for box in self.obstacles)

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
        return not any(box.intersects_segment(start, end) for box in self.obstacles)


DEMO_START_NED: PointNED = (0.0, 0.0, -20.0)
DEMO_GOAL_NED: PointNED = (4000.0, 0.0, -20.0)

DEMO_MAP = ObstacleMap(
    bounds=Box((-200, -600, -180), (4200, 600, 0)),
    obstacles=(
        Box((350, -280, -155), (440, 55, 0)),
        Box((820, -55, -155), (910, 320, 0)),
        Box((1290, -260, -42), (1380, 260, 0)),
        Box((1760, -320, -155), (1850, 65, 0)),
        Box((2230, -70, -150), (2320, 300, 0)),
        Box((2700, -290, -52), (2790, 280, 0)),
        Box((3170, -310, -155), (3260, 60, 0)),
        Box((3640, -65, -160), (3730, 320, 0)),
        Box((370, 340, -48), (460, 430, 0)),
        Box((800, -400, -56), (890, -310, 0)),
        Box((1300, 355, -34), (1370, 420, 0)),
        Box((1745, 350, -72), (1825, 450, 0)),
        Box((2240, -415, -69), (2330, -330, 0)),
        Box((2690, 380, -38), (2770, 470, 0)),
        Box((3180, 330, -45), (3270, 430, 0)),
        Box((3650, -390, -60), (3730, -295, 0)),
    ),
)

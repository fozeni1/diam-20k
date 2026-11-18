"""Поперечные проходы для учебного маршрута с продвижением на север."""

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from .obstacle_map import ObstacleMap, PointNED


@dataclass
class Course:
    north: list[float]
    offsets: np.ndarray

    def motion(self, north: float, speed: float):
        """Положение, скорость и ускорение по осям E и D."""
        i = min(max(bisect_right(self.north, north) - 1, 0), len(self.north) - 2)
        length = self.north[i + 1] - self.north[i]
        u = min(1.0, max(0.0, (north - self.north[i]) / length))
        rate = speed / length
        delta = self.offsets[i + 1] - self.offsets[i]
        # На обоих концах перехода поперечная скорость и ускорение равны нулю.
        position = self.offsets[i] + delta * u**3 * (10 - 15*u + 6*u*u)
        velocity = delta * (30*u*u*(1-u)**2 * rate)
        acceleration = delta * (60*u*(1-u)*(1-2*u) * rate**2)
        return position, velocity, acceleration


def plan_course(world: ObstacleMap, start: PointNED, goal: PointNED) -> Course:
    if goal[0] <= start[0]:
        raise ValueError("Финиш полигона должен находиться севернее старта")
    safe = world.with_margin(3.0)
    if not all(safe.is_position_free(p) for p in (start, goal)):
        raise ValueError("Старт или финиш за пределами свободного пространства")

    # Объекты с перекрывающимися северными координатами проходим одной группой.
    groups = []
    for box in sorted(safe.obstacles, key=lambda b: b.minimum_ned[0]):
        begin = max(start[0], box.minimum_ned[0] - 6)
        end = min(goal[0], box.maximum_ned[0] + 6)
        if begin >= end:
            continue
        if groups and begin <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2].append(box)
        else:
            groups.append([begin, end, [box]])

    knots = [start]

    def append(point):
        if point != knots[-1]:
            knots.append(point)

    for begin, end, boxes in groups:
        east = {start[1], goal[1]}
        down = {start[2], goal[2]}
        for box in boxes:
            east.update((box.minimum_ned[1] - 58, box.maximum_ned[1] + 58))
            down.add(box.minimum_ned[2] - 12)
        choices = []
        for e in east:
            for d in down:
                if begin == start[0] and (e, d) != start[1:]:
                    continue
                if end == goal[0] and (e, d) != goal[1:]:
                    continue
                if safe.is_segment_free((begin, e, d), (end, e, d)):
                    choices.append(((e-start[1])**2 + 4*(d-start[2])**2, e, d))
        if not choices:
            raise ValueError(f"Не найден свободный проход между N={begin:g} и N={end:g} м")
        _, east_pass, down_pass = min(choices)
        append((begin, east_pass, down_pass))
        append((end, east_pass, down_pass))
    append(goal)
    return Course([p[0] for p in knots], np.array([p[1:] for p in knots]))

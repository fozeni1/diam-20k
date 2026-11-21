"""Сетка высот полигона, пересечение с поверхностью и подложка участка Юты."""

import json

from base64 import b64encode
from importlib.resources import files
from math import atan, ceil, degrees, floor, pi, sinh

import numpy as np


def terrain_layer(origin, bounds):
    # Границы исходного тайла Terrain Tiles в Web Mercator.
    zoom, x, y = 12, 789, 1587
    count = 2**zoom
    west, east = [col * 360 / count - 180 for col in (x, x + 1)]
    north, south = [degrees(atan(sinh(pi * (1 - 2*row/count)))) for row in (y, y+1)]
    if not (south < origin.latitude_deg < north and west < origin.longitude_deg < east):
        return None
    nw = origin.to_ned(north, west, origin.ellipsoid_height_m)
    se = origin.to_ned(south, east, origin.ellipsoid_height_m)
    if not (se[0] <= bounds.minimum_ned[0] and nw[0] >= bounds.maximum_ned[0]
            and nw[1] <= bounds.minimum_ned[1] and se[1] >= bounds.maximum_ned[1]):
        return None
    raster = files("diam20k").joinpath("utah-relief.jpg").read_bytes()
    return {"nw": nw[:2], "se": se[:2], "image": "data:image/jpeg;base64," +
            b64encode(raster).decode("ascii")}


class TerrainGrid:
    """Высоты над уровнем старта. Ячейку делит диагональ между (1,0) и (0,1)."""

    def __init__(self, north0, east0, step, heights):
        self.north0, self.east0, self.step = north0, east0, step
        self.heights = np.asarray(heights, dtype=float)
        if (self.heights.ndim != 2 or min(self.heights.shape) < 2 or step <= 0
                or not np.isfinite([north0, east0, step]).all() or not np.isfinite(self.heights).all()):
            raise ValueError("Неверный размер, шаг или высоты сетки рельефа")
        self.rows, self.cols = self.heights.shape
        dn = abs(np.diff(self.heights, axis=0)) / step
        de = abs(np.diff(self.heights, axis=1)) / step
        self.slope_n = np.maximum(dn[:, :-1], dn[:, 1:])
        self.slope_e = np.maximum(de[:-1, :], de[1:, :])

    def _coordinates(self, north, east):
        return (north-self.north0)/self.step, (east-self.east0)/self.step

    def height(self, north, east):
        u, v = self._coordinates(north, east)
        if not (0 <= u <= self.rows-1 and 0 <= v <= self.cols-1):
            raise ValueError("Координаты за пределами сетки высот")
        i, j = min(int(u), self.rows-2), min(int(v), self.cols-2)
        a, b = u-i, v-j
        z = self.heights
        if a+b <= 1:
            return z[i,j]*(1-a-b) + z[i+1,j]*a + z[i,j+1]*b
        return z[i+1,j+1]*(a+b-1) + z[i+1,j]*(1-b) + z[i,j+1]*(1-a)

    def _profile(self, start, end):
        u, v = self._coordinates(*start[:2])
        x, y = self._coordinates(*end[:2])
        cuts = {0., 1.}
        # Целые u и v дают границы ячеек, целая сумма u+v — их диагонали.
        # Между соседними пересечениями луч идёт над одной плоской гранью.
        for a, b in ((u,x), (v,y), (u+v,x+y)):
            for k in range(floor(min(a,b))+1, ceil(max(a,b))):
                cuts.add((k-a)/(b-a))
        delta = np.subtract(end, start)
        for t in sorted(cuts):
            n, e, d = start + t*delta
            yield t, -d-self.height(n,e)

    def first_hit(self, start, end):
        previous_t = previous_gap = 0.
        for t, gap in self._profile(start, end):
            if gap <= 0:
                if t == 0:
                    return 0.
                return previous_t + (t-previous_t)*previous_gap/(previous_gap-gap)
            previous_t, previous_gap = t, gap
        return None

    def clearance_bound(self, start, end, extents):
        """Нижняя оценка вертикального зазора до огибающей между отсчётами."""
        dn, de, dd = extents
        u0,v0 = self._coordinates(min(start[0],end[0])-dn, min(start[1],end[1])-de)
        u1,v1 = self._coordinates(max(start[0],end[0])+dn, max(start[1],end[1])+de)
        if u0 < 0 or v0 < 0 or u1 > self.rows-1 or v1 > self.cols-1:
            raise ValueError("Оболочка выходит за пределы сетки высот")
        cells = np.s_[min(int(u0),self.rows-2):int(u1)+1, min(int(v0),self.cols-2):int(v1)+1]
        allowance = dd + dn*self.slope_n[cells].max() + de*self.slope_e[cells].max()
        return min(gap for _, gap in self._profile(start,end)) - allowance


def utah_surface(origin, bounds):
    if (origin.latitude_deg, origin.longitude_deg, origin.ellipsoid_height_m) != (37.45, -110.61, 1300.):
        raise ValueError("Сетка Юты привязана к стандартному началу карты")
    grid = json.loads(files("diam20k").joinpath("utah-grid.json").read_text(encoding="utf-8"))
    surface = TerrainGrid(**grid)
    for n,e,_ in (bounds.minimum_ned, bounds.maximum_ned):
        for delta in (-40, 40):
            surface.height(n+delta, e+delta)
    return surface, grid


def lab_surface(bounds):
    """Невысокие гряды учебного полигона. Координаты и высоты в метрах."""
    step = 50
    n0, e0 = [step*floor(x/step)-100 for x in bounds.minimum_ned[:2]]
    n1, e1 = [step*ceil(x/step)+100 for x in bounds.maximum_ned[:2]]
    north = np.arange(n0, n1+1, step)[:, None]
    east = np.arange(e0, e1+1, step)[None, :]
    heights = np.sin(north/430)**2 * (6+3*np.cos(east/180)) + 2*np.sin(east/160)**2
    grid = {"north0": n0, "east0": e0, "step": step, "heights": heights.round(3).tolist()}
    return TerrainGrid(**grid), grid

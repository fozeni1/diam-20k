"""Local NED georeferencing for a small map on WGS 84."""

from dataclasses import dataclass, field
from math import cos, degrees, isfinite, radians, sin, sqrt

from .obstacle_map import PointNED, _point_ned

_SEMI_MAJOR_M = 6_378_137.0
_ECCENTRICITY_SQUARED = 6.69437999014e-3


@dataclass(frozen=True)
class MapOrigin:
    """Geodetic origin in degrees and meters above the WGS 84 ellipsoid."""

    latitude_deg: float
    longitude_deg: float
    ellipsoid_height_m: float
    _north_radius_m: float = field(init=False, repr=False)
    _east_radius_m: float = field(init=False, repr=False)

    def __post_init__(self):
        _point_ned((self.latitude_deg, self.longitude_deg, self.ellipsoid_height_m))
        if not -90 < self.latitude_deg < 90:
            raise ValueError("Широта начала должна быть строго между −90 и 90°")
        if not -180 <= self.longitude_deg <= 180:
            raise ValueError("Долгота начала должна быть между −180 и 180°")

        latitude = radians(self.latitude_deg)
        curvature = sqrt(1 - _ECCENTRICITY_SQUARED * sin(latitude)**2)
        meridian = _SEMI_MAJOR_M * (1 - _ECCENTRICITY_SQUARED) / curvature**3
        prime_vertical = _SEMI_MAJOR_M / curvature
        north_radius = meridian + self.ellipsoid_height_m
        east_radius = (prime_vertical + self.ellipsoid_height_m) * cos(latitude)
        if not all(isfinite(radius) and radius > 0 for radius in (north_radius, east_radius)):
            raise ValueError("Высота начала несовместима с локальной системой NED")
        object.__setattr__(self, "_north_radius_m", north_radius)
        object.__setattr__(self, "_east_radius_m", east_radius)

    def to_geodetic(self, position_ned: PointNED) -> tuple[float, float, float]:
        north, east, down = _point_ned(position_ned)
        latitude = self.latitude_deg + degrees(north / self._north_radius_m)
        longitude_delta = degrees(east / self._east_radius_m)
        if not -90 < latitude < 90 or not -180 <= longitude_delta < 180:
            raise ValueError("Точка вне области локальной привязки")
        longitude = (self.longitude_deg + longitude_delta + 180) % 360 - 180
        return latitude, longitude, self.ellipsoid_height_m - down

    def to_ned(self, latitude_deg: float, longitude_deg: float, height_m: float) -> PointNED:
        latitude, longitude, height = _point_ned((latitude_deg, longitude_deg, height_m))
        if not -90 < latitude < 90 or not -180 <= longitude <= 180:
            raise ValueError("Географические координаты вне допустимого диапазона")
        longitude_delta = (longitude - self.longitude_deg + 180) % 360 - 180
        return (
            radians(latitude - self.latitude_deg) * self._north_radius_m,
            radians(longitude_delta) * self._east_radius_m,
            self.ellipsoid_height_m - height,
        )

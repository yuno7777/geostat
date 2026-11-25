"""Local scene overlay and WGS84 measurement, independent of raster and network I/O."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from pyproj import Geod, Transformer
from shapely import (
    STRtree,
    get_coordinates,
    intersection,
    orient_polygons,
    segmentize,
    transform,
    union_all,
)
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.affinity import translate

GEOD = Geod(ellps="WGS84")
WGS84 = "EPSG:4326"
BBox = tuple[float, float, float, float]
Coordinates = NDArray[np.float64]


def _parts(geom: BaseGeometry) -> Iterator[BaseGeometry]:
    if geom.is_empty:
        return
    if isinstance(geom, (GeometryCollection, MultiPolygon)) or geom.geom_type.startswith("Multi"):
        for part in geom.geoms:
            yield from _parts(part)
    else:
        yield geom


def _coordinates(geom: BaseGeometry) -> Coordinates:
    coords: Coordinates = get_coordinates(geom)
    if not np.isfinite(coords).all() or (np.abs(coords[:, 1]) > 90).any():
        raise ValueError("WGS84 geometry contains nonfinite coordinates or latitude outside [-90, 90]")
    return coords


def wgs84_length_m(geom: BaseGeometry) -> float:
    """Measure lines and polygon boundaries along WGS84 geodesics."""
    _coordinates(geom)
    lengths = []
    for part in _parts(geom):
        if isinstance(part, Polygon):
            lengths.append(wgs84_length_m(part.boundary))
        elif part.geom_type in {"LineString", "LinearRing"}:
            coords = get_coordinates(part)
            lengths.append(float(GEOD.line_length(coords[:, 0], coords[:, 1])))
    return math.fsum(lengths)


def wgs84_area_m2(geom: BaseGeometry) -> float:
    """Measure polygon components independently, subtracting holes regardless of winding."""
    _coordinates(geom)
    areas = []
    for part in _parts(geom):
        if isinstance(part, Polygon):
            # GEOS determines winding in XY. Unwrap local antimeridian rings first.
            coords = get_coordinates(part)
            local = _around(part, float(coords[0, 0]))
            # A ring encircling a pole may have zero area in XY. Geod can still
            # measure it, so measure each ring independently using its absolute area.
            polygon = local if local.is_valid else part
            shell, _ = GEOD.geometry_area_perimeter(Polygon(polygon.exterior))
            holes = math.fsum(
                abs(GEOD.geometry_area_perimeter(Polygon(ring))[0])
                for ring in polygon.interiors
            )
            areas.append(max(0.0, abs(shell) - holes))
    return math.fsum(areas)


def _around(geom: BaseGeometry, center: float) -> BaseGeometry:
    def shift(coords: Coordinates) -> Coordinates:
        delta = coords[:, 0] - center
        outside = (delta < -180) | (delta >= 180)
        coords[outside, 0] -= 360 * np.floor((delta[outside] + 180) / 360)
        return coords

    result: BaseGeometry = transform(geom, shift)
    return result


@lru_cache(maxsize=32)
def _to_wgs84(crs: str) -> Transformer:
    return Transformer.from_crs(crs, WGS84, always_xy=True)


def bounds_wgs84(bounds: BBox, crs: str) -> BBox:
    """Return west, south, east, north. East may be less than west at the dateline."""
    west, south, east, north = _to_wgs84(crs).transform_bounds(*bounds, densify_pts=21, errcheck=True)
    if not all(math.isfinite(v) for v in (west, south, east, north)) or not -90 <= south < north <= 90:
        raise ValueError("raster bounds do not define a finite WGS84 extent")
    if east - west >= 360:
        return -180.0, south, 180.0, north
    # Already-normalized values retain all their original precision.
    if not -180 <= west <= 180:
        west = (west + 180) % 360 - 180
    if not -180 <= east <= 180:
        east = (east + 180) % 360 - 180
    return west, south, east, north


def polygons_wgs84(geom: BaseGeometry, crs: str) -> Iterator[BaseGeometry]:
    """Transform raster polygons and cut any antimeridian crossings for GeoJSON."""
    converter = _to_wgs84(crs)
    if not converter.source_crs.equals(converter.target_crs, ignore_axis_order=True):
        minx, miny, maxx, maxy = geom.bounds
        geom = segmentize(geom, max(maxx - minx, maxy - miny) / 32)

        def project(coords: Coordinates) -> Coordinates:
            lon, lat = converter.transform(coords[:, 0], coords[:, 1], errcheck=True)
            return np.column_stack((lon, lat))

        geom = transform(geom, project)
    for part in _parts(geom):
        if not isinstance(part, Polygon):
            continue
        coords = _coordinates(part)
        center = float((coords[0, 0] + 180) % 360 - 180)
        local = _around(part, center)
        minx, _, maxx, _ = local.bounds
        if not local.is_valid or maxx - minx >= 180:
            raise ValueError("flood polygon is invalid or too wide for a local scene overlay")
        for strip in range(math.floor((minx + 180) / 360), math.floor((maxx + 180) / 360) + 1):
            offset = 360 * strip
            clipped = local if strip == 0 and minx >= -180 and maxx <= 180 else local.intersection(
                box(-180 + offset, -90, 180 + offset, 90)
            )
            for piece in _parts(clipped):
                if isinstance(piece, Polygon):
                    yield orient_polygons(translate(piece, xoff=-offset))


class FloodIndex:
    """Index flood polygons in a single local longitude frame.

    Overlay uses straight segments in locally unwrapped longitude/latitude.
    This model supports local scenes, including dateline crossings.
    """

    def __init__(self, geometries: Sequence[BaseGeometry | None]) -> None:
        parts = [p for geom in geometries if geom is not None for p in _parts(geom)]
        if any(not isinstance(p, Polygon) for p in parts):
            raise ValueError("flood geometries must be polygons")
        self._center = float(_coordinates(parts[0])[0, 0]) if parts else 0.0
        for part in parts:
            _coordinates(part)
        local = [_around(p, self._center) for p in parts]
        if any(not p.is_valid or p.bounds[2] - p.bounds[0] >= 180 for p in local):
            raise ValueError("flood geometry is invalid or outside a local longitude frame")
        if local and max(p.bounds[2] for p in local) - min(p.bounds[0] for p in local) >= 180:
            raise ValueError("flood scene spans 180 degrees of longitude or more")
        self._tree = STRtree(local)

    def clip(self, geom: BaseGeometry) -> BaseGeometry | None:
        if not len(self._tree) or geom.is_empty:
            return None
        _coordinates(geom)
        local = _around(geom, self._center)
        if local.bounds[2] - local.bounds[0] >= 180:
            # The asset crosses the opposite seam, outside the local scene.
            return None
        if not local.is_valid:
            raise ValueError("asset geometry is invalid")
        matches = self._tree.query(local, predicate="intersects")
        if not matches.size:
            return None
        candidates = self._tree.geometries.take(matches)
        try:
            clipped: BaseGeometry = intersection(local, candidates[0]) if matches.size == 1 else union_all(
                intersection(candidates, local)
            )
            return clipped
        except GEOSException as exc:
            raise ValueError("could not intersect asset with flood polygons") from exc

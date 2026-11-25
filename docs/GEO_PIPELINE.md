# GeoStat geospatial pipeline

## Raster grids and masks

GeoStat requires a declared CRS on both input rasters. Training and reference
comparison require the same CRS, dimensions and affine grid. The transform
comparison defaults to a (10^{-9}) absolute tolerance in native units.
Ingest does not resample misaligned rasters.

Label rasters use 0 for land, 1 for water and −1 for invalid pixels. GDAL masks
and nodata metadata are honored on both inputs. A SAR pixel is available if at
least one band is finite. The model fills unavailable bands with zero after
clipping dB to [−30, 5] and scaling to [0, 1]. The loss and evaluation mask
exclude pixels without usable SAR or a valid label.

## Tiling and scene inference

The manifest uses a fixed, non-overlapping grid with equal tile size and
stride. Tiles inherit their scene's event split. A stable ID combines the scene
ID and tile row and column. Training and inference pad border windows with
zero-valued SAR and mark the padded pixels invalid. Inference crops back to
the source dimensions. Non-overlapping tiles can show boundary artifacts
where the model has reduced context.

## Outputs and exposure

| Artifact | CRS | Format and validity |
|---|---|---|
| Flood probability | Source CRS | float32 COG, NaN nodata, values in [0, 1] |
| Flood mask | Source CRS | uint8 COG, 255 nodata, 0 land, 1 water |
| Flood and reference vectors | WGS84 lon/lat | GeoJSON polygon features, split at the antimeridian |
| Infrastructure exposure | WGS84 ellipsoid | JSON summary of measured lengths, areas, counts and OSM attribution |

Rasterio polygonizes wet pixels using four-way connectivity and the raster
affine. The export repairs invalid polygonal topology where possible,
transforms projected coordinates to WGS84 using longitude-first order and
splits at the antimeridian. COG pixels stay on their original grid.

The exposure stage indexes flood polygons in a local longitude frame, clips
against an asset's intersecting polygons, and measures the clipped geometry
using WGS84 geodesics. Clipping follows straight segments in that local
longitude/latitude frame. It supports ordinary local scenes on both sides of
the dateline. Pole-enclosing and broad, global polygon overlays are outside the
local-scene model. High-latitude length and area measurement uses the ellipsoid.

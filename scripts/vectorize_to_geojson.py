"""
scripts/vectorize_to_geojson.py

Converts binary building masks to GeoJSON polygons.
Reads CRS from GeoTIFF tags (no GDAL required) and reprojects UTM -> WGS84.

Usage:
    # single tile
    python -m scripts.vectorize_to_geojson \
        --mask predictions/austin1_mask.png \
        --source data/inria/train/images/austin1.tif \
        --out vectors/austin1.geojson

    # batch
    python -m scripts.vectorize_to_geojson \
        --mask_dir submission/mitb5 \
        --source_dir data/inria/test/images \
        --out_dir vectors
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.postprocess import extract_building_polygons


def read_geotransform(tif_path):
    try:
        import tifffile
    except ImportError:
        return None, None
    try:
        with tifffile.TiffFile(tif_path) as tif:
            tags = tif.pages[0].tags
            scale    = tags.get(33550)
            tiepoint = tags.get(33922)
            transform = tags.get(34264)

            crs = None
            try:
                geo = tif.geotiff_metadata or {}
                crs = geo.get("ProjectedCSTypeGeoKey") or geo.get("GeographicTypeGeoKey")
            except Exception:
                pass

            if scale is not None and tiepoint is not None:
                sx, sy = scale.value[0], scale.value[1]
                i, j   = tiepoint.value[0], tiepoint.value[1]
                X, Y   = tiepoint.value[3], tiepoint.value[4]
                def fn(col, row, X0=X, Y0=Y, i=i, j=j, sx=sx, sy=sy):
                    return (X0 + (col - i) * sx, Y0 - (row - j) * sy)
                return fn, crs

            if transform is not None:
                m = list(transform.value)
                a, b, d = m[0], m[1], m[3]
                e, f, h = m[4], m[5], m[7]
                def fn(col, row, a=a, b=b, d=d, e=e, f=f, h=h):
                    return (a * col + b * row + d, e * col + f * row + h)
                return fn, crs
    except Exception:
        pass
    return None, None


def make_reprojector(src_epsg):
    if src_epsg is None:
        return None
    try:
        from pyproj import Transformer
        t = Transformer.from_crs(src_epsg, 4326, always_xy=True)
        return lambda x, y: t.transform(x, y)
    except Exception:
        return None


def shoelace(ring):
    n = len(ring)
    s = sum(ring[i][0] * ring[(i+1)%n][1] - ring[(i+1)%n][0] * ring[i][1]
            for i in range(n))
    return abs(s) / 2.0


def build_geojson(polygons, transform_fn, crs_epsg, source_name):
    reproject = make_reprojector(crs_epsg) if transform_fn else None
    wgs84 = reproject is not None
    features = []

    for idx, (poly, area_px) in enumerate(polygons, 1):
        if transform_fn:
            utm = [transform_fn(int(c), int(r)) for c, r in poly]
            if wgs84:
                ring = [list(reproject(x, y)) for x, y in utm]
                area_val, area_unit = round(shoelace(utm), 1), "m2"
                crs_tag = "WGS84"
            else:
                ring = [[round(x, 2), round(y, 2)] for x, y in utm]
                area_val, area_unit = round(shoelace(utm), 1), "m2"
                crs_tag = f"UTM EPSG:{crs_epsg}"
        else:
            ring = [[int(c), int(r)] for c, r in poly]
            area_val, area_unit, crs_tag = float(area_px), "px", "pixel"

        if ring[0] != ring[-1]:
            ring.append(ring[0])

        features.append({
            "type": "Feature",
            "properties": {
                "id": idx,
                "area": area_val,
                "area_unit": area_unit,
                "crs": crs_tag,
                "source": source_name,
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    return {"type": "FeatureCollection", "features": features}, wgs84


def process_one(mask_path, source_path, out_path, min_area, simplify):
    mask = np.array(Image.open(mask_path).convert("L"))
    transform_fn, crs = None, None
    if source_path and os.path.isfile(source_path):
        transform_fn, crs = read_geotransform(source_path)

    polygons = extract_building_polygons(mask, min_area=min_area, simplify_frac=simplify)
    fc, wgs84 = build_geojson(polygons, transform_fn, crs, Path(mask_path).name)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)

    crs_info = "WGS84" if wgs84 else (f"UTM EPSG:{crs}" if transform_fn else "pixels")
    print(f"  {Path(mask_path).name}: {len(polygons)} buildings [{crs_info}] -> {out_path}")
    return len(polygons), wgs84


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask",       type=str)
    ap.add_argument("--source",     type=str)
    ap.add_argument("--out",        type=str)
    ap.add_argument("--mask_dir",   type=str)
    ap.add_argument("--source_dir", type=str)
    ap.add_argument("--out_dir",    type=str, default="vectors")
    ap.add_argument("--min_area",   type=int,   default=40)
    ap.add_argument("--simplify",   type=float, default=0.01)
    args = ap.parse_args()

    if args.mask:
        out = args.out or (os.path.splitext(args.mask)[0] + ".geojson")
        process_one(args.mask, args.source, out, args.min_area, args.simplify)
        return

    if args.mask_dir:
        masks = sorted(Path(args.mask_dir).glob("*.tif"))
        total = 0
        print(f"vectorizing {len(masks)} masks...")
        for m in masks:
            src = os.path.join(args.source_dir, m.name) if args.source_dir else None
            out = os.path.join(args.out_dir, m.stem + ".geojson")
            nb, _ = process_one(str(m), src, out, args.min_area, args.simplify)
            total += nb
        print(f"done: {total} buildings -> {args.out_dir}/")
        return

    ap.error("specify --mask or --mask_dir")


if __name__ == "__main__":
    main()

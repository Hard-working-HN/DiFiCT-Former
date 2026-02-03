# -*- coding: utf-8 -*-

import os
import sys
from typing import Iterable, Tuple

import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd


SRC_ROOT = r"F:\Your\Rasters\Root"
CUTLINE_SHP = r"F:\Article2\SSP可持续发展路径\地图\2023年县级.shp"
OUT_ROOT = r"F:\Your\Rasters\Result"
ADD_SUFFIX = ""
OVERWRITE = False
DST_NODATA_DEFAULT = 0


def iter_tifs(root: str) -> Iterable[str]:
    exts = (".tif", ".tiff", ".TIF", ".TIFF")
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(exts):
                yield os.path.join(r, fn)


def make_out_path(in_path: str) -> str:
    rel_dir = os.path.relpath(os.path.dirname(in_path), SRC_ROOT)
    out_dir = os.path.join(OUT_ROOT, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(in_path)
    if ADD_SUFFIX:
        name, ext = os.path.splitext(base)
        base = f"{name}{ADD_SUFFIX}{ext}"
    return os.path.join(out_dir, base)


def load_shapes_in_crs(shp_path: str, dst_crs) -> Tuple[Iterable, str]:
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise RuntimeError("The shapefile has no features.")
    if gdf.crs is None:
        raise RuntimeError("The shapefile has no CRS (.prj). Please add CRS and try again.")
    gdf = gdf.to_crs(dst_crs)
    return gdf.geometry.values, str(gdf.crs)


def clip_one(src_tif: str, shp_path: str, out_tif: str):
    with rasterio.open(src_tif) as src:
        nodata = src.nodata if (src.nodata is not None) else DST_NODATA_DEFAULT

        shapes, _ = load_shapes_in_crs(shp_path, src.crs)

        out_arr, out_transform = rio_mask(
            src,
            shapes=shapes,
            crop=True,
            all_touched=True,
            nodata=nodata,
        )

        profile = src.profile.copy()
        profile.update(
            {
                "height": out_arr.shape[1],
                "width": out_arr.shape[2],
                "transform": out_transform,
                "nodata": nodata,
                "driver": "GTiff",
            }
        )

        for k in ["compress", "tiled", "blockxsize", "blockysize", "predictor", "sparse_ok", "bigtiff"]:
            profile.pop(k, None)

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(out_arr)


def main():
    if not os.path.isdir(SRC_ROOT):
        print(f"[ERR] Source folder does not exist: {SRC_ROOT}")
        sys.exit(1)
    if not os.path.isfile(CUTLINE_SHP):
        print(f"[ERR] Shapefile does not exist: {CUTLINE_SHP}")
        sys.exit(1)

    os.makedirs(OUT_ROOT, exist_ok=True)

    count_all = 0
    count_done = 0
    count_skip = 0
    count_fail = 0

    for tif in iter_tifs(SRC_ROOT):
        count_all += 1
        out_path = make_out_path(tif)

        if (not OVERWRITE) and os.path.exists(out_path):
            print(f"[SKIP] Already exists: {out_path}")
            count_skip += 1
            continue

        try:
            print(f"[CLIP] {tif}\n   -> {out_path}")
            clip_one(tif, CUTLINE_SHP, out_path)
            count_done += 1
        except Exception as e:
            print(f"[FAIL] {tif}\nReason: {e}")
            count_fail += 1

    print("\n====== Summary ======")
    print(f"Found: {count_all}")
    print(f"Clipped: {count_done}")
    print(f"Skipped (exists): {count_skip}")
    print(f"Failed: {count_fail}")


if __name__ == "__main__":
    main()

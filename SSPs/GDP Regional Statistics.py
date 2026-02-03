# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from rasterio import open as rio_open
from rasterstats import zonal_stats


SHP_PATH = r"F:\Article2\SSP_RCP\MAP\County2023.shp"
TIF_ROOT = r"F:\Article2\SSP_RCP\China_Clip\GDP"
OUT_CSV = r"F:\Article2\SSP_RCP\Result\GDP_sum_by_code.csv"
CODE_FIELD = "code"
ALL_TOUCHED = False
STRIP_EXT = False
FILL_IF_NONE = 0.0
FIX_INVALID_GEOM = True
FORCE_NODATA_ZERO = True
MAKE_ZONES_DISJOINT = False


def list_tifs(root_dir):
    tifs = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".tif", ".tiff")):
                tifs.append(os.path.join(root, f))
    tifs.sort()
    return tifs


def safe_fix_geometry(gdf):
    if not FIX_INVALID_GEOM:
        return gdf
    fixed = gdf.copy()
    fixed["__is_valid__"] = fixed.geometry.is_valid
    if (~fixed["__is_valid__"]).any():
        print("[Geom] Invalid geometries detected. Trying buffer(0) to fix them...")
        fixed.loc[~fixed["__is_valid__"], fixed.geometry.name] = (
            fixed.loc[~fixed["__is_valid__"], fixed.geometry.name].buffer(0)
        )
    fixed = fixed.drop(columns="__is_valid__", errors="ignore")
    fixed = fixed[~fixed.geometry.is_empty & fixed.geometry.notna()].copy()
    return fixed


def dissolve_by_code_keep_attrs(gdf, code_field, original_cols_order):
    attrs = gdf.drop(columns=gdf.geometry.name).copy()
    if code_field not in attrs.columns:
        raise ValueError(f"Field not found in SHP: {code_field}")
    attrs_first = attrs.groupby(code_field, as_index=False).first()
    gdf_geom = gdf[[code_field, gdf.geometry.name]].dissolve(by=code_field).reset_index()
    merged = gdf_geom.merge(attrs_first, on=code_field, how="left")
    front_cols = [c for c in original_cols_order if c != gdf.geometry.name]
    merged = merged[[*front_cols, merged.geometry.name]]
    return merged


def ensure_polygon(gdf):
    mask = gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"[Geom] Dropped {dropped} non-polygon features (points/lines).")
    return gdf[mask].copy()


def normalize_code_series(s):
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def pick_nodata(src):
    if FORCE_NODATA_ZERO:
        return 0.0
    if src.nodata is not None:
        return float(src.nodata)
    try:
        nd = src.nodatavals[0]
        if nd is not None:
            return float(nd)
    except Exception:
        pass
    return 0.0


def make_disjoint_by_area(gdf, code_field):
    gdf = gdf.copy()
    gdf["__area__"] = gdf.geometry.area
    gdf = gdf.sort_values("__area__", ascending=False).drop(columns="__area__", errors="ignore")

    taken = None
    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if taken is not None:
            geom = geom.difference(taken)
        if geom.is_empty:
            continue
        nr = row.copy()
        nr.geometry = geom
        rows.append(nr)
        taken = geom if taken is None else unary_union([taken, geom])

    out = gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)
    return out


def incremental_write_column(csv_path, base_df, code_field, col_name, values):
    col_name = str(col_name)
    add_df = pd.DataFrame({code_field: base_df[code_field].values, col_name: values})
    add_df[code_field] = normalize_code_series(add_df[code_field])

    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        base_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[CSV] Initialized with SHP attributes -> {csv_path}")

    df_now = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={code_field: "string"})
    if code_field not in df_now.columns:
        raise RuntimeError(f"CSV missing key field '{code_field}', cannot align and append.")
    if col_name in df_now.columns:
        print(f"[CSV] Column already exists: {col_name}. Skipping.")
        return

    merged = df_now.merge(add_df, on=code_field, how="left")
    merged = merged[list(df_now.columns) + [col_name]]
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[CSV] Appended column: {col_name}")


def main():
    t0 = time.time()
    print("=== Job started (ensure each pixel counted at most once) ===")
    print(f"[Config] SHP: {SHP_PATH}")
    print(f"[Config] TIF_ROOT: {TIF_ROOT}")
    print(f"[Config] OUT_CSV: {OUT_CSV}")
    print(f"[Config] MAKE_ZONES_DISJOINT={MAKE_ZONES_DISJOINT}, ALL_TOUCHED={ALL_TOUCHED}")
    print("-" * 60)

    gdf = gpd.read_file(SHP_PATH)
    if gdf.empty:
        raise RuntimeError("Shapefile is empty.")
    if CODE_FIELD not in gdf.columns:
        raise ValueError(f"Field not found in SHP: {CODE_FIELD}")
    original_cols_order = list(gdf.columns)
    print(f"[SHP] Features: {len(gdf)} | CRS: {gdf.crs}")

    gdf = safe_fix_geometry(gdf)
    gdf = ensure_polygon(gdf)
    gdf[CODE_FIELD] = normalize_code_series(gdf[CODE_FIELD])

    if gdf.duplicated(CODE_FIELD).any():
        dup_n = int(gdf[CODE_FIELD].duplicated().sum())
        print(f"[SHP] Found {dup_n} duplicated '{CODE_FIELD}' values. Dissolving by '{CODE_FIELD}'...")
        gdf = dissolve_by_code_keep_attrs(gdf, CODE_FIELD, original_cols_order)
        gdf[CODE_FIELD] = normalize_code_series(gdf[CODE_FIELD])
        print(f"[SHP] Dissolve done. Rows: {len(gdf)}")
    else:
        front_cols = [c for c in original_cols_order if c != gdf.geometry.name]
        gdf = gdf[[*front_cols, gdf.geometry.name]]

    if MAKE_ZONES_DISJOINT:
        print("[SHP] Making polygons disjoint (largest-first; subtract overlaps)...")
        before = len(gdf)
        gdf = make_disjoint_by_area(gdf, CODE_FIELD)
        after = len(gdf)
        print(f"[SHP] Disjoint done. Before={before}, After={after}")

    base_df = gdf.drop(columns=gdf.geometry.name).copy()
    base_df[CODE_FIELD] = normalize_code_series(base_df[CODE_FIELD])

    if not os.path.exists(OUT_CSV):
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        base_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print(f"[CSV] Initialized with SHP attributes -> {OUT_CSV}")
    else:
        print(f"[CSV] Existing file found; will append new columns -> {OUT_CSV}")

    tif_paths = list_tifs(TIF_ROOT)
    if not tif_paths:
        raise RuntimeError("No TIF files found under TIF_ROOT.")
    print(f"[TIF] Found {len(tif_paths)} rasters.")
    print("-" * 60)

    gdf_cache = {}

    for idx, tif in enumerate(tif_paths, 1):
        t_start = time.time()
        try:
            with rio_open(tif) as src:
                tif_crs = src.crs
                bounds = src.bounds
                transform = src.transform
                res_x = float(transform.a)
                res_y = float(-transform.e) if transform.e < 0 else float(transform.e)
                nodata_use = pick_nodata(src)

                print(f"[{idx}/{len(tif_paths)}] {tif}")
                print(f"  - CRS: {tif_crs}")
                print(f"  - Resolution: {res_x} x {res_y} | Bounds: {bounds}")
                print(f"  - Using nodata={nodata_use} (ALL_TOUCHED={ALL_TOUCHED})")

                key = str(tif_crs) if tif_crs is not None else "NONE"
                if key in gdf_cache:
                    gdf_use = gdf_cache[key]
                    print("  - Reusing reprojected polygons.")
                else:
                    base = gdf if (gdf.crs is not None) else gdf.set_crs("EPSG:4326")
                    gdf_use = base if (tif_crs is None) else base.to_crs(tif_crs)
                    if tif_crs is not None:
                        print("  - Reprojected polygons to raster CRS.")
                    gdf_cache[key] = gdf_use

                raster_bbox = box(*bounds)
                n_inter = int(gdf_use.intersects(raster_bbox).sum())
                print(f"  - Intersecting zones: {n_inter} / {len(gdf_use)}")

                zs = zonal_stats(
                    vectors=gdf_use,
                    raster=tif,
                    stats=["sum", "count"],
                    nodata=nodata_use,
                    all_touched=ALL_TOUCHED,
                    raster_out=False,
                    geojson_out=False,
                    band=1,
                )

                sums = []
                none_count = 0
                for z in zs:
                    val = z.get("sum", None)
                    cnt = z.get("count", 0)
                    if val is None or (isinstance(val, float) and np.isnan(val)) or cnt == 0:
                        sums.append(float(FILL_IF_NONE))
                        none_count += 1
                    else:
                        sums.append(float(val))

                col_name = Path(tif).name if not STRIP_EXT else Path(tif).stem

                incremental_write_column(
                    csv_path=OUT_CSV,
                    base_df=base_df,
                    code_field=CODE_FIELD,
                    col_name=col_name,
                    values=sums,
                )

                dt = time.time() - t_start
                print(f"  - Done: {col_name} | filled-empty: {none_count} | time: {dt:.2f}s")

        except Exception as e:
            print(f"[ERR] Failed: {tif}\n      Reason: {e}")

        print("-" * 60)

    print(f"=== Finished -> {OUT_CSV} | Total time {time.time() - t0:.2f}s ===")


if __name__ == "__main__":
    main()

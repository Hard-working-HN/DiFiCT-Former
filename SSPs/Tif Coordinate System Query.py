# -*- coding: utf-8 -*-

import sys
from pathlib import Path


def find_first_tif(folder: Path):
    cands = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))
    if not cands:
        cands = sorted(list(folder.rglob("*.tif")) + list(folder.rglob("*.tiff")))
    return cands[0] if cands else None


def print_with_rasterio(tif_path: Path):
    import rasterio

    with rasterio.open(tif_path) as ds:
        crs = ds.crs
        print(f"[File] {tif_path}")
        print(f"[Driver] {ds.driver} | Size={ds.width}x{ds.height} | Bands={ds.count}")

        if crs:
            epsg = crs.to_epsg()
            print(f"[CRS] {crs.to_string()}")
            print(f"[EPSG] {epsg if epsg is not None else '-'}")
            if crs.is_geographic:
                crs_type = "Geographic (lon/lat)"
            elif crs.is_projected:
                crs_type = "Projected"
            else:
                crs_type = "Unknown"
            print(f"[Type] {crs_type}")

            try:
                units = getattr(crs, "linear_units", None) or getattr(crs, "unit_name", None)
            except Exception:
                units = None
            if not units and crs.is_geographic:
                units = "degree"
            print(f"[Units] {units if units else '-'}")

            try:
                axis = getattr(crs, "axis_info", None)
                if axis:
                    axis_str = "; ".join([f"{a.name} ({a.direction})" for a in axis])
                    print(f"[Axis] {axis_str}")
            except Exception:
                pass
        else:
            print("[CRS] Not found (this raster may have no CRS).")

        print(f"[Resolution] {ds.res[0]} x {ds.res[1]} (same units as CRS)")
        print(f"[Transform] {ds.transform}")

        b = ds.bounds
        print(f"[Bounds] left={b.left}, bottom={b.bottom}, right={b.right}, top={b.top}")


def print_with_gdal(tif_path: Path):
    from osgeo import gdal, osr

    ds = gdal.Open(str(tif_path))
    if ds is None:
        print(f"[GDAL] Cannot open: {tif_path}")
        return

    print(f"[File] {tif_path}")
    print(f"[Driver] {ds.GetDriver().ShortName} | Size={ds.RasterXSize}x{ds.RasterYSize} | Bands={ds.RasterCount}")

    gt = ds.GetGeoTransform(can_return_null=True)
    if gt:
        x_min = gt[0]
        px_w = gt[1]
        y_max = gt[3]
        px_h = gt[5]

        print(f"[Resolution] {px_w} x {abs(px_h)}")
        x_max = x_min + ds.RasterXSize * px_w
        y_min = y_max + ds.RasterYSize * px_h
        print(f"[Bounds] left={x_min}, bottom={y_min}, right={x_max}, top={y_max}")
        print(f"[Transform] {gt}")
    else:
        print("[GeoTransform] -")

    wkt = ds.GetProjection()
    if wkt:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        try:
            epsg = srs.GetAttrValue("AUTHORITY", 1)
        except Exception:
            epsg = None

        wkt_preview = wkt[:200] + ("..." if len(wkt) > 200 else "")
        print(f"[CRS WKT] {wkt_preview}")
        print(f"[EPSG] {epsg if epsg else '-'}")
        if srs.IsGeographic():
            srs_type = "Geographic (lon/lat)"
        elif srs.IsProjected():
            srs_type = "Projected"
        else:
            srs_type = "Unknown"
        print(f"[Type] {srs_type}")
    else:
        print("[CRS] Not found (this raster may have no CRS).")


def main():
    if len(sys.argv) < 2:
        print('Please provide a folder path, e.g.: python get_first_tif_crs.py "F:/data/rasters"')
        sys.exit(1)

    folder = Path(sys.argv[1]).expanduser()
    if not folder.exists() or not folder.is_dir():
        print(f"[ERR] Invalid folder: {folder}")
        sys.exit(1)

    tif = find_first_tif(folder)
    if not tif:
        print("[INFO] No .tif/.tiff files found in this folder (or its subfolders).")
        sys.exit(0)

    try:
        import rasterio  # noqa: F401

        print_with_rasterio(tif)
    except Exception as e:
        print(f"[rasterio] Not available or failed ({e}). Falling back to GDAL.")
        try:
            print_with_gdal(tif)
        except Exception as e2:
            print(f"[GDAL] Failed: {e2}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv.append(r"F:\Article2\SSP可持续发展路径\China_Clip\Land_Use\SSP1-RCP26\2020")
    main()

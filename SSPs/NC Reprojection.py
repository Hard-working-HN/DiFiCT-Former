# -*- coding: utf-8 -*-

from pathlib import Path
import os
import sys
import warnings
import numpy as np


IN_ROOT = Path(r"F:\Article2\SSP_RCP\Resampling_Fixed\Pr\conservative\SSP126\IPSL").resolve()
OUT_ROOT = Path(r"F:\Article2\SSP\Reprojection").resolve()

KEEP_SHAPE = True
RESAMPLING_METHOD = "nearest"
INCLUDE_VARS = None
SKIP_IF_EXISTS = True
VAR_ENCODING = dict(zlib=True, complevel=1, shuffle=True)


def boost_conda_dll_search():
    if os.name != "nt":
        return
    prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
    for sub in ("Library\\bin", "DLLs", "lib", "bin"):
        p = os.path.join(prefix, sub)
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
    prepend = [os.path.join(prefix, s) for s in ("Library\\bin", "DLLs", "lib", "bin")]
    original = os.environ.get("PATH", "")
    os.environ["PATH"] = ";".join([*prepend, original])


boost_conda_dll_search()

warnings.filterwarnings("ignore", ".*dataset contains invalid geotransform.*")
warnings.filterwarnings("ignore", ".*will be ignored by CF conventions.*")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_nc_files(root: Path):
    return sorted(root.rglob("*.nc"))


def build_out_path(in_path: Path) -> Path:
    rel = in_path.relative_to(IN_ROOT)
    return OUT_ROOT / rel.parent / f"{in_path.stem}_wgs84.nc"


def open_dataset_robust(nc_path: Path):
    import xarray as xr

    last_err = None
    for engine in ("netcdf4", "h5netcdf"):
        try:
            return xr.open_dataset(nc_path, engine=engine, chunks="auto", mask_and_scale=True), engine
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot open with xarray: {nc_path}. Last error: {last_err}")


def guess_spatial_dims(ds):
    cand = [
        ("lat", "lon", True),
        ("latitude", "longitude", True),
        ("y", "x", False),
        ("Y", "X", False),
    ]
    for y, x, is_geo in cand:
        if y in ds.dims and x in ds.dims:
            return y, x, is_geo
    if "lat" in ds and "lon" in ds and ds["lat"].ndim == 1 and ds["lon"].ndim == 1:
        return "lat", "lon", True
    if "y" in ds and "x" in ds and ds["y"].ndim == 1 and ds["x"].ndim == 1:
        return "y", "x", False
    return None, None, None


def select_data_vars(ds, y_dim, x_dim):
    names = []
    for k, da in ds.data_vars.items():
        if INCLUDE_VARS and k not in INCLUDE_VARS:
            continue
        if y_dim in da.dims and x_dim in da.dims:
            names.append(k)
    return names


def resampling_enum(name):
    from rasterio.enums import Resampling

    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "mode": Resampling.mode,
        "max": Resampling.max,
        "min": Resampling.min,
        "med": Resampling.med,
        "q1": Resampling.q1,
        "q3": Resampling.q3,
    }
    return mapping[name.lower()]


def write_crs_if_missing_latlon(da, epsg="EPSG:4326"):
    try:
        if getattr(da.rio, "crs", None) is None:
            da = da.rio.write_crs(epsg, inplace=False)
    except Exception:
        pass
    return da


def try_infer_and_write_crs_from_grid_mapping(ds, da):
    if da.rio.crs is not None:
        return da

    gm_name = da.attrs.get("grid_mapping")
    if gm_name and gm_name in ds.variables:
        gm = ds[gm_name]
        wkt = gm.attrs.get("spatial_ref") or gm.attrs.get("crs_wkt")
        if wkt:
            try:
                return da.rio.write_crs(wkt, inplace=False)
            except Exception:
                pass

    try:
        if ds.rio.crs is not None:
            return da.rio.write_crs(ds.rio.crs, inplace=False)
    except Exception:
        pass

    for v in ds.variables.values():
        try:
            wkt = getattr(v, "attrs", {}).get("spatial_ref") or getattr(v, "attrs", {}).get("crs_wkt")
            if wkt:
                try:
                    return da.rio.write_crs(wkt, inplace=False)
                except Exception:
                    pass
        except Exception:
            pass

    return da


def reproject_only_xarray(ds, y_dim, x_dim):
    import xarray as xr
    import rioxarray  # noqa: F401

    rs = resampling_enum(RESAMPLING_METHOD)

    var_names = select_data_vars(ds, y_dim, x_dim)
    if not var_names:
        raise RuntimeError("No data variables found containing spatial dims.")

    out_vars = {}
    for name in var_names:
        da = ds[name].rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=False)
        is_latlon = (y_dim.lower() in ("lat", "latitude")) and (x_dim.lower() in ("lon", "longitude"))

        if is_latlon:
            da = write_crs_if_missing_latlon(da, "EPSG:4326")
            out_vars[name] = da
            print(f"    [OK] {name}: lat/lon -> only write/normalize WGS84")
            continue

        da = try_infer_and_write_crs_from_grid_mapping(ds, da)
        if da.rio.crs is None:
            raise RuntimeError(f"{name}: missing source CRS (grid_mapping/spatial_ref), cannot reproject.")

        kwargs = dict(dst_crs="EPSG:4326", resampling=rs)
        if KEEP_SHAPE:
            kwargs["shape"] = (da.sizes[y_dim], da.sizes[x_dim])

        da_wgs = da.rio.reproject(**kwargs)
        out_vars[name] = da_wgs
        print(f"    [OK] {name}: x/y -> WGS84 (shape preserved)")

    ds_out = xr.Dataset(out_vars)

    rename = {}
    if y_dim.lower() not in ("lat", "latitude"):
        rename[y_dim] = "lat"
    if x_dim.lower() not in ("lon", "longitude"):
        rename[x_dim] = "lon"
    if rename:
        ds_out = ds_out.rename(rename)

    try:
        ds_out = ds_out.rio.write_crs("EPSG:4326", inplace=False)
        ds_out.rio.write_coordinate_system(inplace=True)
    except Exception:
        pass

    for k in list(ds_out.data_vars):
        dims = list(ds_out[k].dims)
        order = []
        for d in ("time", "lat", "lon"):
            if d in dims:
                order.append(d)
        order += [d for d in dims if d not in order]
        if order != dims:
            ds_out[k] = ds_out[k].transpose(*order)

    return ds_out


def rewrap_h5py_only(nc_path: Path):
    import h5py
    import xarray as xr
    import rioxarray  # noqa: F401

    with h5py.File(str(nc_path), "r") as f:
        lat = f.get("lat") or f.get("latitude")
        lon = f.get("lon") or f.get("longitude")
        if lat is None or lon is None:
            raise RuntimeError("h5py fallback: no 1D lat/lon found.")

        latv = np.array(lat[...])
        lonv = np.array(lon[...])

        def walk(g, prefix=""):
            for k in g.keys():
                obj = g[k]
                p = f"{prefix}/{k}" if prefix else k
                if isinstance(obj, h5py.Dataset):
                    yield p, obj
                elif isinstance(obj, h5py.Group):
                    yield from walk(obj, p)

        best = None
        best_size = -1
        for path, ds_ in walk(f):
            base = Path(path).name.lower()
            if base in {"lat", "latitude", "lon", "longitude", "x", "y", "time"}:
                continue
            if ds_.ndim < 2:
                continue
            size = int(np.prod(ds_.shape))
            if size > best_size:
                best = (path, ds_)
                best_size = size

        if best is None:
            raise RuntimeError("h5py fallback: no 2D/3D data variable found.")

        var_path, var_ds = best
        arr = np.array(var_ds[...])

        coords = {"lat": latv, "lon": lonv}
        if arr.ndim == 3:
            time = f.get("time")
            coords["time"] = np.array(time[...]) if time is not None else np.arange(arr.shape[0], dtype=np.float64)
            dims = ["time", "lat", "lon"]
        elif arr.ndim == 2:
            dims = ["lat", "lon"]
        else:
            raise RuntimeError("h5py fallback supports only 2D or 3D variables.")

        da = xr.DataArray(arr, dims=dims, coords=coords, name=Path(var_path).name)
        ds_out = da.to_dataset()

        try:
            ds_out = ds_out.rio.write_crs("EPSG:4326", inplace=False)
            ds_out.rio.write_coordinate_system(inplace=True)
        except Exception:
            pass

        return ds_out


def save_netcdf(ds, out_path: Path):
    ensure_dir(out_path.parent)

    encoding = {k: VAR_ENCODING.copy() for k in ds.data_vars}
    for c in ("time", "lat", "lon"):
        if c in ds:
            encoding[c] = {"zlib": False}

    last_err = None
    for engine in ("netcdf4", "h5netcdf"):
        try:
            ds.to_netcdf(out_path, engine=engine, format="NETCDF4", encoding=encoding)
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to save: {out_path}. Last error: {last_err}")


def process_one(nc_path: Path):
    out_path = build_out_path(nc_path)
    if SKIP_IF_EXISTS and out_path.exists():
        print(f"  -> SKIP (exists): {out_path}")
        return

    print(f"[IN] {nc_path}")

    xarray_err = None
    try:
        ds, _eng = open_dataset_robust(nc_path)
        y_dim, x_dim, _is_geo = guess_spatial_dims(ds)
        if y_dim is None or x_dim is None:
            raise RuntimeError("Cannot recognize spatial dims (expected lat/lon or y/x).")
        ds_out = reproject_only_xarray(ds, y_dim, x_dim)
        save_netcdf(ds_out, out_path)
        ds.close()
        print(f"  -> WROTE: {out_path}")
        return
    except Exception as e:
        xarray_err = e
        print(f"  [xarray failed, trying h5py fallback] Reason: {e}")

    try:
        ds_out = rewrap_h5py_only(nc_path)
        save_netcdf(ds_out, out_path)
        print(f"  -> WROTE (h5py fallback): {out_path}")
        return
    except Exception as e2:
        raise RuntimeError(
            f"Both methods failed for: {nc_path}\n"
            f"  - xarray error: {xarray_err}\n"
            f"  - h5py error: {e2}"
        )


def main():
    print(f"Input root:  {IN_ROOT}")
    print(f"Output root: {OUT_ROOT}")
    print("Action: reproject to EPSG:4326 only; keep shape; lat/lon -> write WGS84 only.")

    files = list_nc_files(IN_ROOT)
    if not files:
        print("No .nc files found.")
        return

    ok = 0
    fail = 0

    for p in files:
        try:
            process_one(p)
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {p}\n    Reason: {e}")
            fail += 1

    print(f"\n=== Done === OK: {ok} | FAIL: {fail}")


if __name__ == "__main__":
    main()

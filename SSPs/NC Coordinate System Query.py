# -*- coding: utf-8 -*-

import sys
import os
import argparse
from pathlib import Path
import numpy as np


def win_longpath(p: Path) -> str:
    s = str(p)
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        try:
            if not p.is_absolute():
                p = p.resolve()
            s = "\\\\?\\" + str(p)
        except Exception:
            s = "\\\\?\\" + s
    return s


def find_nc_candidates(folder: Path, min_size_mb: float = 1.0):
    files = sorted(list(folder.glob("*.nc")))
    if not files:
        files = sorted(list(folder.rglob("*.nc")))
    min_bytes = int(min_size_mb * 1024 * 1024)
    files = [f for f in files if f.exists() and f.stat().st_size >= min_bytes]
    return files


def median_step(a):
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or a.size < 2:
        return np.nan
    return float(np.median(np.diff(a)))


def centers_to_edges_1d(c):
    c = np.asarray(c, dtype=float)
    if c.ndim != 1 or c.size < 2:
        if c.size == 1:
            return np.array([c[0] - 0.5, c[0] + 0.5], dtype=float)
        return np.array([0.0, 0.0], dtype=float)
    step = median_step(c)
    e = np.empty(c.size + 1, dtype=float)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - 0.5 * step
    e[-1] = c[-1] + 0.5 * step
    return e


def human_units_for_latlon(lat_units, lon_units):
    lat_u = (str(lat_units or "")).lower()
    lon_u = (str(lon_units or "")).lower()
    return "degree" if ("degree" in lat_u or "degree" in lon_u) else (lat_units or lon_units or "-")


def summarize(lon, lat, units_hint=None):
    dx = abs(median_step(lon))
    dy = abs(median_step(lat))
    ex = centers_to_edges_1d(lon)
    ey = centers_to_edges_1d(lat)
    left, right = min(ex[0], ex[-1]), max(ex[0], ex[-1])
    bottom, top = min(ey[0], ey[-1]), max(ey[0], ey[-1])
    return {
        "grid_type": "Geographic (lon/lat)",
        "units": units_hint or "degree",
        "resolution": (dx, dy),
        "bounds": (left, bottom, right, top),
    }


def summarize_xy(x, y, units_hint=None):
    dx = abs(median_step(x))
    dy = abs(median_step(y))
    ex = centers_to_edges_1d(x)
    ey = centers_to_edges_1d(y)
    left, right = min(ex[0], ex[-1]), max(ex[0], ex[-1])
    bottom, top = min(ey[0], ey[-1]), max(ey[0], ey[-1])
    return {
        "grid_type": "Projected",
        "units": units_hint or "-",
        "resolution": (dx, dy),
        "bounds": (left, bottom, right, top),
    }


def try_open_netcdf4(path: Path):
    from netCDF4 import Dataset

    ds = Dataset(win_longpath(path), "r")
    return ds, "netCDF4"


def try_open_h5netcdf(path: Path):
    try:
        import hdf5plugin  # noqa: F401
    except Exception:
        pass
    from h5netcdf import legacyapi as NC

    ds = NC.Dataset(win_longpath(path), "r")
    return ds, "h5netcdf"


def try_open_h5py(path: Path):
    import h5py

    f = h5py.File(win_longpath(path), "r")
    return f, "h5py"


def parse_via_netcdf(ds):
    info = {
        "backend": "netCDF4",
        "var": None,
        "shape": None,
        "crs_text": None,
        "epsg": None,
        "grid_type": "Unknown",
        "units": "-",
        "resolution": (None, None),
        "bounds": (None, None, None, None),
        "notes": [],
    }

    def var_by_names(names):
        for n in names:
            if n in ds.variables:
                return ds.variables[n]
        return None

    def var_by_std(std):
        for v in ds.variables.values():
            if getattr(v, "standard_name", "").lower() == std:
                return v
        return None

    lat = var_by_names(["lat", "latitude"]) or var_by_std("latitude")
    lon = var_by_names(["lon", "longitude"]) or var_by_std("longitude")
    y = var_by_names(["y"]) or var_by_std("projection_y_coordinate")
    x = var_by_names(["x"]) or var_by_std("projection_x_coordinate")

    best = None
    best_size = -1
    for name, v in ds.variables.items():
        stdn = getattr(v, "standard_name", "").lower()
        if name in ("lat", "latitude", "lon", "longitude", "x", "y", "time") or stdn in (
            "latitude",
            "longitude",
            "projection_x_coordinate",
            "projection_y_coordinate",
            "time",
        ):
            continue
        if getattr(v, "ndim", 0) < 2:
            continue
        dims = getattr(v, "dimensions", ())
        if not any(d in dims for d in ("lat", "latitude", "y")):
            continue
        if not any(d in dims for d in ("lon", "longitude", "x")):
            continue
        size = int(np.prod(getattr(v, "shape", ())))
        if size > best_size:
            best = (name, v)
            best_size = size

    if best:
        info["var"] = best[0]
        v = best[1]
        info["shape"] = tuple(getattr(v, "shape", ()))
    else:
        v = None

    crs_text = None
    epsg = None
    gm_name = getattr(v, "grid_mapping", None) if v is not None else None
    gm = ds.variables.get(gm_name) if gm_name else None

    try:
        from pyproj import CRS
    except Exception:
        CRS = None

    if gm is not None:
        wkt = getattr(gm, "spatial_ref", None) or getattr(gm, "crs_wkt", None)
        if wkt:
            crs_text = wkt
            if CRS:
                try:
                    epsg = CRS.from_wkt(wkt).to_epsg()
                except Exception:
                    pass
        elif CRS:
            try:
                cf = {k: getattr(gm, k) for k in gm.ncattrs()}
                crs_obj = CRS.from_cf(cf)
                crs_text = crs_obj.to_string()
                epsg = crs_obj.to_epsg()
            except Exception:
                pass

    if lat is not None and lon is not None and getattr(lat, "ndim", 0) == 1 and getattr(lon, "ndim", 0) == 1:
        info.update(
            summarize(
                lon[:],
                lat[:],
                units_hint=human_units_for_latlon(getattr(lat, "units", None), getattr(lon, "units", None)),
            )
        )
    elif x is not None and y is not None and getattr(x, "ndim", 0) == 1 and getattr(y, "ndim", 0) == 1:
        info.update(summarize_xy(x[:], y[:], units_hint=getattr(x, "units", None) or getattr(y, "units", None) or "-"))
    else:
        info["notes"].append("No standard 1D lat/lon or x/y grid recognized")

    info["crs_text"] = crs_text
    info["epsg"] = epsg
    return info


def parse_via_h5py(f):
    info = {
        "backend": "h5py",
        "var": None,
        "shape": None,
        "crs_text": None,
        "epsg": None,
        "grid_type": "Unknown",
        "units": "-",
        "resolution": (None, None),
        "bounds": (None, None, None, None),
        "notes": [],
    }

    def ds_or_none(name_opts):
        for n in name_opts:
            if n in f:
                return f[n]
        for name in f:
            obj = f[name]
            try:
                import h5py

                if isinstance(obj, h5py.Group):
                    for n in name_opts:
                        if n in obj:
                            return obj[n]
            except Exception:
                pass
        return None

    lat = ds_or_none(["lat", "latitude"])
    lon = ds_or_none(["lon", "longitude"])
    y = ds_or_none(["y"])
    x = ds_or_none(["x"])

    coord_names = {"lat", "latitude", "lon", "longitude", "x", "y", "time"}

    def walk_vars(g, prefix=""):
        import h5py

        for k in g:
            obj = g[k]
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(obj, h5py.Dataset):
                yield path, obj
            elif isinstance(obj, h5py.Group):
                yield from walk_vars(obj, path)

    best = None
    best_size = -1
    for path, dset in walk_vars(f):
        base = Path(path).name.lower()
        if base in coord_names:
            continue
        if dset.ndim < 2:
            continue
        size = int(np.prod(dset.shape))
        if size > best_size:
            best = (path, dset)
            best_size = size

    if best:
        info["var"] = best[0]
        info["shape"] = tuple(best[1].shape)

    if lon is not None and lat is not None and lon.ndim == 1 and lat.ndim == 1:
        lat_units = lat.attrs.get("units", None)
        lon_units = lon.attrs.get("units", None)
        info.update(summarize(lon[...], lat[...], units_hint=human_units_for_latlon(lat_units, lon_units)))
    elif x is not None and y is not None and x.ndim == 1 and y.ndim == 1:
        units = x.attrs.get("units", None) or y.attrs.get("units", None) or "-"
        info.update(summarize_xy(x[...], y[...], units_hint=units))
    else:
        info["notes"].append("In h5py mode, no 1D lon/lat or x/y grid recognized")

    root_attrs = dict(f.attrs.items())
    for k in ("spatial_ref", "crs_wkt", "crs", "projection"):
        if k in root_attrs:
            try:
                info["crs_text"] = root_attrs[k].decode() if isinstance(root_attrs[k], (bytes, bytearray)) else str(root_attrs[k])
            except Exception:
                info["crs_text"] = str(root_attrs[k])
            break

    return info


def pretty(info, file):
    print(f"[File] {file}")
    print(f"[Backend] {info['backend']}")
    print(f"[Var] {info['var']} | Shape={info['shape']}")
    print(f"[CRS] {info['crs_text'] or '-'}")
    print(f"[EPSG] {info['epsg'] if info['epsg'] is not None else '-'}")
    print(f"[Type] {info['grid_type']}")
    print(f"[Units] {info['units']}")
    dx, dy = info["resolution"]
    if dx is not None:
        print(f"[Resolution] {dx} x {dy}")
    else:
        print("[Resolution] -")
    l, b, r, t = info["bounds"]
    if l is not None:
        print(f"[Bounds] left={l}, bottom={b}, right={r}, top={t}")
    else:
        print("[Bounds] -")
    for n in info["notes"]:
        print(f"[Note] {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder to scan")
    ap.add_argument("--min-size", type=float, default=1.0, help="Minimum file size (MB)")
    args = ap.parse_args()

    folder = Path(args.folder)
    cands = find_nc_candidates(folder, args.min_size)
    if not cands:
        print("No .nc files found.")
        return

    for p in cands:
        try:
            from netCDF4 import Dataset as _  # noqa: F401

            ds, _backend = try_open_netcdf4(p)
            info = parse_via_netcdf(ds)
            ds.close()
            pretty(info, p)
            return
        except Exception:
            pass

        try:
            ds, _backend = try_open_h5netcdf(p)
            info = parse_via_netcdf(ds)
            ds.close()
            info["backend"] = "h5netcdf"
            pretty(info, p)
            return
        except Exception:
            pass

        try:
            f, _backend = try_open_h5py(p)
            info = parse_via_h5py(f)
            f.close()
            pretty(info, p)
            return
        except Exception as e:
            print(f"[FAIL] Cannot open: {p} -> {e}")
            continue

    print("All candidate .nc files failed to open.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append(r"F:\Article2\SSP\Result\Pr\SSP126\IPSL")
    main()

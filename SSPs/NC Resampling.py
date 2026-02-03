from pathlib import Path
import os
import sys
import numpy as np

IN_ROOT = Path(r"F:\Article2\SSP_RCP\Resampling").resolve()
OUT_ROOT = Path(r"G:\Article2\SSP_RCP\Resampling_Fixed").resolve()
SKIP_IF_EXISTS = True


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
    os.environ["PATH"] = (
        ";".join([os.path.join(prefix, s) for s in ("Library\\bin", "DLLs", "lib", "bin")])
        + ";"
        + os.environ.get("PATH", "")
    )


boost_conda_dll_search()

import h5py
from netCDF4 import Dataset


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_lat_lon_time_var(fi: h5py.File):
    def find_path(names):
        targets = {n.lower() for n in names}
        found = None

        def walk(g, prefix=""):
            nonlocal found
            for k, v in g.items():
                if found is not None:
                    return
                path = f"{prefix}/{k}" if prefix else k
                if isinstance(v, h5py.Dataset) and k.lower() in targets:
                    found = path
                    return
                if isinstance(v, h5py.Group):
                    walk(v, path)

        walk(fi)
        return found

    lat_p = find_path(("lat", "latitude"))
    lon_p = find_path(("lon", "longitude"))
    time_p = find_path(("time",))

    if not (lat_p and lon_p and time_p):
        raise RuntimeError("lat/lon/time not found")

    return (lat_p, lon_p, time_p), (fi[lat_p][...], fi[lon_p][...], fi[time_p][...])


def find_main_3d_var(fi: h5py.File, ntime, nlat, nlon):
    cands = []

    def walk(g, prefix=""):
        for k, v in g.items():
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(v, h5py.Dataset) and v.ndim == 3 and v.shape == (ntime, nlat, nlon):
                cands.append((int(np.prod(v.shape)), path))
            elif isinstance(v, h5py.Group):
                walk(v, path)

    walk(fi)

    if not cands:
        raise RuntimeError("No main 3D variable found with shape (time, lat, lon)")

    cands.sort(key=lambda x: x[0], reverse=True)
    var_path = cands[0][1]
    return var_path, fi[var_path]


def _decode_attr(v):
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "ignore")
    if isinstance(v, np.ndarray) and v.dtype.kind in ("S", "O"):
        try:
            return np.array([_decode_attr(x) for x in v.tolist()])
        except Exception:
            return v
    return v


def convert_one(src_nc: Path, dst_nc: Path):
    if SKIP_IF_EXISTS and dst_nc.exists():
        print(f"  -> SKIP (exists): {dst_nc}")
        return

    with h5py.File(str(src_nc), "r") as fi:
        (lat_p, lon_p, time_p), (lat, lon, time_vals) = find_lat_lon_time_var(fi)
        var_path, var = find_main_3d_var(fi, len(time_vals), len(lat), len(lon))
        var_name = Path(var_path).name

        var_attrs = {k: _decode_attr(v) for k, v in dict(var.attrs.items()).items()}
        fill_value = var_attrs.get("_FillValue", np.float32(-9.96921e36))

        global_attrs = {k: _decode_attr(v) for k, v in dict(fi.attrs.items()).items()}
        time_attrs = {k: _decode_attr(v) for k, v in dict(fi[time_p].attrs.items()).items()}

        ensure_dir(dst_nc.parent)

        with Dataset(str(dst_nc), "w", format="NETCDF4") as ds:
            ds.createDimension("time", len(time_vals))
            ds.createDimension("lat", len(lat))
            ds.createDimension("lon", len(lon))

            vtime = ds.createVariable("time", "f8", ("time",))
            vlat = ds.createVariable("lat", "f4", ("lat",))
            vlon = ds.createVariable("lon", "f4", ("lon",))

            vtime[:] = np.asarray(time_vals, dtype="f8")
            vlat[:] = np.asarray(lat, dtype="f4")
            vlon[:] = np.asarray(lon, dtype="f4")

            vlat.units = "degrees_north"
            vlon.units = "degrees_east"

            for k, v in (time_attrs or {}).items():
                try:
                    setattr(vtime, k, v)
                except Exception:
                    pass

            chunks = (1, int(min(4096, len(lat))), int(min(8192, len(lon))))
            vout = ds.createVariable(
                var_name,
                "f4",
                ("time", "lat", "lon"),
                zlib=True,
                complevel=1,
                shuffle=True,
                chunksizes=chunks,
                fill_value=np.float32(fill_value),
            )

            for k, v in (var_attrs or {}).items():
                if k in ("_FillValue", "missing_value"):
                    continue
                try:
                    setattr(vout, k, v)
                except Exception:
                    pass

            for k, v in (global_attrs or {}).items():
                try:
                    setattr(ds, k, v)
                except Exception:
                    pass

            fv = fill_value
            fv_is_nan = isinstance(fv, float) and np.isnan(fv)

            for t in range(len(time_vals)):
                day = var[t, :, :].astype(np.float32)
                if fv is not None and not fv_is_nan:
                    day = np.where(day == fv, np.float32(np.nan), day)
                vout[t, :, :] = day

    print(f"  -> WROTE: {dst_nc}")


def main():
    print(f"Input root:  {IN_ROOT}")
    print(f"Output root: {OUT_ROOT}")

    files = sorted(IN_ROOT.rglob("*.nc"))
    if not files:
        print("No .nc files found.")
        return

    ok = 0
    fail = 0

    for f in files:
        try:
            rel = f.relative_to(IN_ROOT)
            dst = OUT_ROOT / rel.parent / f"{f.stem}_fixed.nc"
            print(f"[CONVERT] {f}")
            convert_one(f, dst)
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {f}\n    Reason: {e}")
            fail += 1

    print(f"\n=== Done === OK: {ok} | FAIL: {fail}")


if __name__ == "__main__":
    main()

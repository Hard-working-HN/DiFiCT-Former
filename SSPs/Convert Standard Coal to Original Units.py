from pathlib import Path
import pandas as pd

INPUT_ROOT = r"F:\Article2\SSP_RCP\Result\新情景设定\能源细分及标准煤转换为原能源\能源细分\2013-2100小类能源汇总结果"
OUTPUT_ROOT = r"F:\Article2\SSP_RCP\Result\新情景设定\能源细分及标准煤转换为原能源\标准煤转换为原始能源"

DIV_COEF = {
    "原煤 (万吨标准煤) RawCoal (104tons)_Energy": 0.7143,
    "洗精煤 (万吨标准煤) Cleaned Coal (104tons)_Energy": 0.9,
    "其他洗煤 (万吨标准煤) Other Washed Coal (104tons)_Energy": 0.3333,
    "煤制品 (万吨标准煤) Briquettes (104tons)_Energy": 0.5286,
    "煤矸石 (万吨标准煤) Gangue (104tons)_Energy": 0.2857,
    "焦炭 (万吨标准煤) Coke (104tons)_Energy": 0.9714,
    "焦炉煤气 (万吨标准煤) CokeOven Gas (108cu.m)_Energy": 5.928,
    "高炉煤气 (万吨标准煤) BlastFurnace Gas (108cu.m)_Energy": 1.286,
    "转炉煤气 (万吨标准煤) Converter Gas (108cu.m)_Energy": 2.714,
    "其他煤气 (万吨标准煤) OtherGas (108cu.m)_Energy": 6.243,
    "其他焦化产品 (万吨标准煤) Other Coking Products (104tons)_Energy": 1.3,
    "原油 (万吨标准煤) CrudeOil (104tons)_Energy": 1.4286,
    "汽油 (万吨标准煤) Gasoline (104tons)_Energy": 1.4714,
    "煤油 (万吨标准煤) Kerosene (104tons)_Energy": 1.4714,
    "柴油 (万吨标准煤) DieselOil (104tons)_Energy": 1.4571,
    "燃料油 (万吨标准煤) FuelOil (104tons)_Energy": 1.4286,
    "石脑油 (万吨标准煤) Naphtha (104tons)_Energy": 1.5,
    "润滑油 (万吨标准煤) Lubricants (104tons)_Energy": 1.4143,
    "石蜡 (万吨标准煤) Paraffin Waxes (104tons)_Energy": 1.3648,
    "溶剂油 (万吨标准煤) WhiteSpirit (104tons)_Energy": 1.4672,
    "石油沥青 (万吨标准煤) Bitumen Asphalt (104tons)_Energy": 1.3307,
    "石油焦 (万吨标准煤) Petroleum Coke (104tons)_Energy": 1.0918,
    "液化石油气 (万吨标准煤) LPG (104tons)_Energy": 1.7143,
    "炼厂干气 (万吨标准煤) Refinery Gas (104tons)_Energy": 1.5714,
    "其他石油制品 (万吨标准煤) Other Petroleum Products (104tons)_Energy": 1.4,
    "天然气 (万吨标准煤) NaturalGas (108cu.m)_Energy": 12.15,
    "液化天然气 (万吨标准煤) LNG (104tons)_Energy": 1.7572,
    "热力 (万吨标准煤) Heat (1010kJ)_Energy": 0.03412,
    "电力 (万吨标准煤) Electricity (108kW·h)_Energy": 1.229,
}

for k, v in DIV_COEF.items():
    if v == 0:
        raise ValueError(f"Coefficient cannot be 0: {k}")

def read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    seps = [",", "\t"]
    last_err = None

    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(path, encoding=enc, sep=sep)
            except Exception as e:
                last_err = e
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to read: {path}\nLast error: {last_err}")

def convert_one_file(in_csv: Path, out_csv: Path):
    df = read_csv_flexible(in_csv)

    for col, coef in DIV_COEF.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = s / coef

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

def main():
    in_root = Path(INPUT_ROOT)
    out_root = Path(OUTPUT_ROOT)

    if not in_root.exists():
        raise FileNotFoundError(f"Input directory not found: {in_root.resolve()}")

    csv_files = sorted(in_root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {in_root.resolve()}")

    ok, skip = 0, 0
    for f in csv_files:
        rel = f.relative_to(in_root)
        out_path = out_root / rel

        try:
            convert_one_file(f, out_path)
            ok += 1
            print(f"[OK] {rel}")
        except Exception as e:
            skip += 1
            print(f"[SKIP] {rel} -> {e}")

    print(f"\n[DONE] Success: {ok}, Skipped: {skip}")
    print(f"Output directory: {out_root.resolve()}")

if __name__ == "__main__":
    main()

"""
NEDO 拡張アメダス(EA-20)気象データ 読み込みモジュール

NEDO METPV-20 からダウンロードしたCSV/テキストファイルを読み込み、
pvlib が利用できる形式の pandas.DataFrame に変換する。
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 公開API
# ---------------------------------------------------------------------------

def read_ea20(filepath: str, timezone: str = "Asia/Tokyo") -> pd.DataFrame:
    """
    NEDO EA-20 形式の気象データファイルを読み込む。

    Parameters
    ----------
    filepath : str
        CSV/テキストファイルのパス
    timezone : str
        タイムゾーン (デフォルト: "Asia/Tokyo")

    Returns
    -------
    pd.DataFrame
        列: GHI, DNI, DHI [W/m²], Tair [°C], RH [%], WS [m/s]
        インデックス: DatetimeIndex (各時刻の開始時刻, tz付き)
    """
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"気象データファイルが見つかりません: {fp}")

    raw = _read_raw(fp)
    df = _normalize_columns(raw)
    df = _convert_units(df)
    df = _build_index(df, timezone)

    # 日射量の負値をゼロに補正
    for col in ["GHI", "DNI", "DHI"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    _check_required(df)
    return df


def create_sample_ea20(output_path: str = "data/ea20.csv") -> pd.DataFrame:
    """
    サンプル気象データを生成する（実データがない場合のテスト用）。
    東京 (35.69N, 139.69E) の典型的な気象パターンを近似。
    """
    import pvlib

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    times = pd.date_range(
        start="2020-01-01 00:00", periods=8760, freq="h", tz="Asia/Tokyo"
    )
    loc = pvlib.location.Location(35.69, 139.69, tz="Asia/Tokyo", altitude=25)
    solar_pos = loc.get_solarposition(times)

    # 月別平均日射量 [W/m²] (東京・水平面近似)
    monthly_ghi_avg = {
        1: 100, 2: 140, 3: 185, 4: 205, 5: 225, 6: 185,
        7: 195, 8: 225, 9: 170, 10: 155, 11: 115, 12: 95,
    }

    rng = np.random.default_rng(42)
    ghi_list, dni_list, dhi_list, temp_list = [], [], [], []

    monthly_temp_avg = {
        1: 5, 2: 6, 3: 9, 4: 15, 5: 20, 6: 24,
        7: 28, 8: 30, 9: 26, 10: 20, 11: 14, 12: 8,
    }

    for i, t in enumerate(times):
        elev = solar_pos["elevation"].iloc[i]
        if elev <= 0:
            ghi_list.append(0.0)
            dni_list.append(0.0)
            dhi_list.append(0.0)
        else:
            g_avg = monthly_ghi_avg[t.month]
            elev_rad = np.radians(elev)
            ghi = max(0.0, g_avg * np.sin(elev_rad) * 1.5 + rng.normal(0, 40))
            kt = rng.uniform(0.4, 0.7)  # clearness index
            dhi = max(0.0, ghi * (1 - kt))
            cos_z = max(0.01, np.cos(np.radians(solar_pos["apparent_zenith"].iloc[i])))
            dni = max(0.0, (ghi - dhi) / cos_z)
            ghi_list.append(ghi)
            dni_list.append(dni)
            dhi_list.append(dhi)

        t_avg = monthly_temp_avg[t.month]
        temp = t_avg + 4 * np.sin(np.pi * (t.hour - 6) / 12) + rng.normal(0, 1.5)
        temp_list.append(temp)

    df_out = pd.DataFrame(
        {
            "month": [t.month for t in times],
            "day": [t.day for t in times],
            "hour": [t.hour for t in times],
            "GHI": np.round(ghi_list, 1),
            "DNI": np.round(dni_list, 1),
            "DHI": np.round(dhi_list, 1),
            "Tair": np.round(temp_list, 1),
            "RH": np.clip(60 + rng.normal(0, 10, len(times)), 20, 100),
            "WS": np.clip(rng.exponential(2.5, len(times)), 0, 20),
        }
    )
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  [sample] サンプル気象データを生成しました: {output_path}")
    return df_out


# ---------------------------------------------------------------------------
# 内部処理
# ---------------------------------------------------------------------------

def _read_raw(fp: Path) -> pd.DataFrame:
    """複数エンコーディング・区切り文字に対応したCSV読み込み"""
    for enc in ("shift_jis", "utf-8", "utf-8-sig", "cp932"):
        try:
            with open(fp, encoding=enc, errors="replace") as f:
                head = [f.readline() for _ in range(40)]
            skip = _detect_skip_rows(head)
            sep = "\t" if "\t" in head[skip] else ","
            df = pd.read_csv(fp, skiprows=skip, encoding=enc,
                             sep=sep, on_bad_lines="skip")
            if len(df) > 100:  # 最低限の行数があるか確認
                return df
        except Exception:
            continue
    raise ValueError(f"ファイルを読み込めませんでした: {fp}")


def _detect_skip_rows(lines: list) -> int:
    """数値列が多い最初の行をデータ開始行と判定"""
    for i, line in enumerate(lines):
        parts = re.split(r"[\t,]", line.strip())
        n_numeric = sum(1 for p in parts if re.match(r"^-?\d+\.?\d*$", p.strip()))
        if n_numeric >= 4:
            return max(0, i - 1)  # 1行前をヘッダーとして使う
    return 0


def _normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """列名を統一された英語名に変換する"""
    mapping = {}
    for col in raw.columns:
        c = str(col).lower().strip()
        if re.search(r"ghi|全天|global_h|gh(?!i)", c):
            mapping[col] = "GHI"
        elif re.search(r"dni|直達|direct_n|dn(?!i)", c):
            mapping[col] = "DNI"
        elif re.search(r"dhi|散乱|diffuse|dh(?!i)", c):
            mapping[col] = "DHI"
        elif re.search(r"tair|temp|気温|外気|ta$|temperature", c):
            mapping[col] = "Tair"
        elif re.search(r"\brh\b|湿度|humidity", c):
            mapping[col] = "RH"
        elif re.search(r"\bws\b|風速|wind_?speed|wv", c):
            mapping[col] = "WS"
        elif re.search(r"^月$|^month$", c):
            mapping[col] = "month"
        elif re.search(r"^日$|^day$", c):
            mapping[col] = "day"
        elif re.search(r"^時$|^hour$", c):
            mapping[col] = "hour"

    return raw.rename(columns=mapping)


def _convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """MJ/m²/h → W/m² 変換（必要な場合のみ）"""
    for col in ["GHI", "DNI", "DHI"]:
        if col not in df.columns:
            continue
        col_max = df[col].replace(0, np.nan).dropna()
        if len(col_max) > 0 and col_max.max() < 10:
            # MJ/m²/h とみなして変換 (1 MJ/m²/h = 277.78 W/m²)
            df[col] = df[col] * 277.78
            print(f"  [unit] {col}: MJ/m²/h → W/m² に変換しました")
    return df


def _build_index(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """DatetimeIndex を構築する"""
    if {"month", "day", "hour"}.issubset(df.columns):
        # NEDO形式: hour は 0-23 または 1-24
        df = df.copy()
        year = 2020  # 閏年を避けるため2020固定
        ts_list = []
        for _, row in df.iterrows():
            m, d, h = int(row["month"]), int(row["day"]), int(row["hour"])
            if h == 24:
                h = 0
                # 簡易処理: 月末/日末の繰り上げは無視（24時は23時として扱う）
            ts_list.append(pd.Timestamp(year=year, month=m, day=d, hour=h))

        df.index = pd.DatetimeIndex(ts_list, tz=timezone)
    else:
        # 行番号から8760時間のインデックスを生成
        start = pd.Timestamp("2020-01-01 00:00:00", tz=timezone)
        df.index = pd.date_range(start=start, periods=len(df), freq="h")

    return df


def _check_required(df: pd.DataFrame):
    required = {"GHI", "DNI", "DHI", "Tair"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"必要な列が見つかりません: {missing}\n"
            "data/README.md を参照して列名を確認してください。"
        )

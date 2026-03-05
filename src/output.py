"""
結果出力モジュール

シミュレーション結果を CSV / Excel に保存する。
Excel は以下のシートを含む:
  - 時刻別データ  : 8760時間の詳細
  - 月別集計      : 月ごとの合計
  - 日別最大需要  : 日毎のピーク
  - サマリー      : 年間 KPI
"""

import io
from pathlib import Path
from typing import Dict

import pandas as pd


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(result: pd.DataFrame, path: str) -> None:
    """時刻別データを CSV に保存する"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "GHI", "Tair",
        "poa_global", "solar_elevation",
        "cell_temp", "E_gen_kWh",
        "E_hvac_kWh", "E_hw_kWh", "E_base_kWh", "E_total_kWh",
        "charge_kw", "discharge_kw",
        "grid_export_kw", "grid_import_kw",
        "soc_after", "E_self_kWh",
    ]
    cols = [c for c in cols if c in result.columns]
    result[cols].to_csv(path, encoding="utf-8-sig", float_format="%.4f")
    print(f"  CSV 保存: {path}")


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def save_excel(
    result: pd.DataFrame,
    summary: Dict,
    path: str,
) -> None:
    """
    結果を Excel (xlsx) に保存する。
    シート: 時刻別データ / 月別集計 / サマリー
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Excel は timezone-aware な DatetimeIndex を受け付けないため除去
    result_xl = result.copy()
    if hasattr(result_xl.index, "tz") and result_xl.index.tz is not None:
        result_xl.index = result_xl.index.tz_localize(None)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:

        # ---- シート1: 時刻別データ ----
        hourly_cols = [
            "GHI", "Tair", "poa_global",
            "E_gen_kWh", "E_total_kWh",
            "E_hvac_kWh", "E_hw_kWh", "E_base_kWh",
            "charge_kw", "discharge_kw",
            "grid_export_kw", "grid_import_kw",
            "soc_after", "E_self_kWh",
        ]
        hourly_cols = [c for c in hourly_cols if c in result_xl.columns]
        result_xl[hourly_cols].to_excel(writer, sheet_name="時刻別データ")

        # ---- シート2: 月別集計 ----
        sum_cols = [
            "E_gen_kWh", "E_total_kWh",
            "E_hvac_kWh", "E_hw_kWh",
            "grid_export_kw", "grid_import_kw",
            "E_self_kWh",
        ]
        sum_cols = [c for c in sum_cols if c in result_xl.columns]
        monthly = result_xl[sum_cols].resample("ME").sum()
        monthly.index = monthly.index.strftime("%Y-%m")
        monthly.index.name = "月"
        col_rename = {
            "E_gen_kWh":       "発電量[kWh]",
            "E_total_kWh":     "消費量[kWh]",
            "E_hvac_kWh":      "空調[kWh]",
            "E_hw_kWh":        "給湯[kWh]",
            "grid_export_kw":  "売電[kWh]",
            "grid_import_kw":  "買電[kWh]",
            "E_self_kWh":      "自家消費[kWh]",
        }
        monthly = monthly.rename(columns=col_rename)
        monthly.to_excel(writer, sheet_name="月別集計")

        # ---- シート3: サマリー ----
        summary_df = pd.DataFrame(
            list(summary.items()), columns=["項目", "値"]
        )
        summary_df.to_excel(writer, sheet_name="サマリー", index=False)

    print(f"  Excel 保存: {path}")


def to_excel_bytes(result: pd.DataFrame, summary: Dict) -> bytes:
    """結果を Excel バイト列として返す (Streamlit ダウンロードボタン用)"""
    buf = io.BytesIO()

    result_xl = result.copy()
    if hasattr(result_xl.index, "tz") and result_xl.index.tz is not None:
        result_xl.index = result_xl.index.tz_localize(None)

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        hourly_cols = [
            "GHI", "Tair", "poa_global",
            "E_gen_kWh", "E_total_kWh",
            "E_hvac_kWh", "E_hw_kWh", "E_base_kWh",
            "charge_kw", "discharge_kw",
            "grid_export_kw", "grid_import_kw",
            "soc_after", "E_self_kWh",
        ]
        hourly_cols = [c for c in hourly_cols if c in result_xl.columns]
        result_xl[hourly_cols].to_excel(writer, sheet_name="時刻別データ")

        sum_cols = [
            "E_gen_kWh", "E_total_kWh",
            "E_hvac_kWh", "E_hw_kWh",
            "grid_export_kw", "grid_import_kw",
            "E_self_kWh",
        ]
        sum_cols = [c for c in sum_cols if c in result_xl.columns]
        monthly = result_xl[sum_cols].resample("ME").sum()
        monthly.index = monthly.index.strftime("%Y-%m")
        monthly.index.name = "月"
        monthly = monthly.rename(columns={
            "E_gen_kWh":       "発電量[kWh]",
            "E_total_kWh":     "消費量[kWh]",
            "E_hvac_kWh":      "空調[kWh]",
            "E_hw_kWh":        "給湯[kWh]",
            "grid_export_kw":  "売電[kWh]",
            "grid_import_kw":  "買電[kWh]",
            "E_self_kWh":      "自家消費[kWh]",
        })
        monthly.to_excel(writer, sheet_name="月別集計")

        summary_df = pd.DataFrame(list(summary.items()), columns=["項目", "値"])
        summary_df.to_excel(writer, sheet_name="サマリー", index=False)

    return buf.getvalue()

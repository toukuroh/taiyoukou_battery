"""
太陽光発電量計算モジュール

傾斜面日射量とパネル仕様から AC 出力電力を計算する。

計算フロー:
  傾斜面日射量 (poa_global)
    → セル温度補正 (NOCT法)
    → DC 出力 (温度係数・システムロス適用)
    → AC 出力 (パワコン効率適用)
"""

import numpy as np
import pandas as pd


def calc_pv_output(
    irradiance_df: pd.DataFrame,
    panel_area: float,
    panel_efficiency: float,
    pcs_efficiency: float,
    temp_coefficient: float,
    noct: float,
    system_loss: float,
) -> pd.DataFrame:
    """
    傾斜面日射量から発電量を計算する。

    Parameters
    ----------
    irradiance_df : pd.DataFrame
        solar_calc.calc_tilted_irradiance の出力
        (poa_global, Tair, WS が必要)
    panel_area : float
        パネル面積合計 [m²]
    panel_efficiency : float
        パネル変換効率 (STC: 25°C, 1000W/m²) [-]
    pcs_efficiency : float
        パワコン効率 [-]
    temp_coefficient : float
        出力温度係数 [/°C]  例: -0.0041 (-0.41%/°C)
    noct : float
        NOCT [°C] (公称動作セル温度)  例: 45
    system_loss : float
        配線・汚れ・ミスマッチ等のロス [-]  例: 0.05 (5%)

    Returns
    -------
    pd.DataFrame
        追加列:
        - cell_temp  : セル温度 [°C]
        - eta_temp   : 温度補正後の変換効率 [-]
        - P_dc_kW    : DC 出力 [kW]
        - P_ac_kW    : AC 出力 [kW]
        - E_gen_kWh  : 1時間の発電量 [kWh]
    """
    df = irradiance_df.copy()

    gpoa = df["poa_global"]
    tair = df.get("Tair", pd.Series(25.0, index=df.index))
    ws   = df.get("WS", pd.Series(1.0, index=df.index))

    # ---- セル温度 (NOCT法) ----
    # Tc = Tair + (NOCT - 20) / 800 * Gpoa
    # 風速補正: 風が強いと冷却効果あり（簡易）
    wind_factor = 1.0 - 0.02 * ws.clip(upper=10)  # 風速10m/s以上は同等
    df["cell_temp"] = (
        tair + (noct - 20.0) / 800.0 * gpoa * wind_factor
    )

    # ---- 温度補正後の変換効率 ----
    df["eta_temp"] = panel_efficiency * (
        1.0 + temp_coefficient * (df["cell_temp"] - 25.0)
    )
    df["eta_temp"] = df["eta_temp"].clip(lower=0.0)  # 負値防止

    # ---- DC 出力 [kW] ----
    # P_dc = Gpoa [W/m²] × A [m²] × η_temp × (1 - loss) / 1000
    df["P_dc_kW"] = (
        gpoa * panel_area * df["eta_temp"] * (1.0 - system_loss) / 1000.0
    ).clip(lower=0.0)

    # ---- AC 出力 [kW] ----
    df["P_ac_kW"] = df["P_dc_kW"] * pcs_efficiency

    # ---- 1時間発電量 [kWh] ----
    # 1時間刻みなので P_ac_kW × 1h = kWh
    df["E_gen_kWh"] = df["P_ac_kW"]

    return df

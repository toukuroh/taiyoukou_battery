"""
建物負荷プロファイル計算モジュール

各時刻の電力消費量を以下の要素から計算する:
  1. 空調負荷 (HVAC): UA値・ηAC・COP から算出
  2. 給湯負荷 (HP給湯機): 深夜に集中運転
  3. ベース負荷 (家電・照明・待機): 時刻別スケジュール
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def calc_load_profile(
    weather_df: pd.DataFrame,
    # 建物スペック
    floor_area: float,
    envelope_area: float,
    ua_value: float,
    eta_ac: float,
    # 空調設定
    heating_setpoint: float,
    cooling_setpoint: float,
    cop_heating: float,
    cop_cooling: float,
    heating_months: List[int],
    cooling_months: List[int],
    hvac_start_h: int,
    hvac_end_h: int,
    # 在室・内部発熱
    occupants: int,
    internal_gain_per_person_w: float,
    # ベース負荷 (時刻別 [W])
    base_load_w: Dict[int, float],
    # 給湯
    hw_daily_kwh: float,
    hw_hour: int,
    hw_op_hours: int,
    hw_cop: float,
) -> pd.DataFrame:
    """
    各時刻の電力消費量を計算する。

    Parameters
    ----------
    weather_df : pd.DataFrame
        GHI, Tair を含む気象 DataFrame (pv_generation の出力でも可)
    floor_area : float
        床面積 [m²]
    envelope_area : float
        外皮面積 [m²]  (外壁+屋根+床の合計面積)
    ua_value : float
        UA値 [W/m²K]  (熱損失係数)
    eta_ac : float
        冷房期日射熱取得率 ηAC [%]  (省エネ計算書の値)
    heating_setpoint : float
        暖房室内設定温度 [°C]
    cooling_setpoint : float
        冷房室内設定温度 [°C]
    cop_heating : float
        暖房 COP (APF 近似)
    cop_cooling : float
        冷房 COP (APF 近似)
    heating_months / cooling_months : list[int]
        暖房・冷房運転月
    hvac_start_h / hvac_end_h : int
        空調稼働時刻範囲 (start <= hour < end)
    occupants : int
        居住者数 [人]
    internal_gain_per_person_w : float
        在室時の人体発熱量 [W/人]
    base_load_w : dict[int, float]
        時刻別ベース負荷 {時刻: [W]}
    hw_daily_kwh : float
        1日あたり給湯電力量 [kWh/日]  (HP給湯機の入力電力)
    hw_hour : int
        HP給湯機の深夜運転開始時刻
    hw_op_hours : int
        HP給湯機の運転時間 [h]
    hw_cop : float
        HP給湯機の COP

    Returns
    -------
    pd.DataFrame
        追加列:
        - Q_heat_kWh  : 必要暖房熱量 [kWh]
        - Q_cool_kWh  : 必要冷房熱量 [kWh]
        - E_hvac_kWh  : 空調消費電力量 [kWh]
        - E_hw_kWh    : 給湯消費電力量 [kWh]
        - E_base_kWh  : ベース負荷消費電力量 [kWh]
        - E_total_kWh : 合計消費電力量 [kWh]
    """
    df = weather_df.copy()
    n = len(df)

    tair_arr = df["Tair"].to_numpy(dtype=float)
    ghi_arr  = df["GHI"].to_numpy(dtype=float) if "GHI" in df.columns else np.zeros(n)

    months_arr = df.index.month.to_numpy()
    hours_arr  = df.index.hour.to_numpy()

    # 空調稼働時間マスク
    hvac_mask = (hours_arr >= hvac_start_h) & (hours_arr < hvac_end_h)

    # --- 暖房負荷 (ベクトル化) ---
    # 貫流 + 換気 (簡易: 貫流の 130%)
    heating_mask = hvac_mask & np.isin(months_arr, heating_months) & (tair_arr < heating_setpoint)
    q_trans_heat = ua_value * envelope_area * np.maximum(0.0, heating_setpoint - tair_arr)
    Q_heat = np.where(heating_mask, q_trans_heat * 1.30, 0.0)

    # --- 冷房負荷 (ベクトル化) ---
    cooling_mask = hvac_mask & np.isin(months_arr, cooling_months) & (tair_arr > cooling_setpoint)
    q_trans_cool = ua_value * envelope_area * np.maximum(0.0, tair_arr - cooling_setpoint)
    q_solar_arr  = (eta_ac / 100.0) * ghi_arr * floor_area * 0.05
    # 内部発熱: 昼間(8〜17時)は居住者 1/3、それ以外は全員
    n_home_arr   = np.where((hours_arr >= 8) & (hours_arr <= 17),
                             max(1, occupants // 3), occupants)
    q_internal_arr = n_home_arr * internal_gain_per_person_w
    Q_cool = np.where(cooling_mask,
                       np.maximum(0.0, q_trans_cool + q_solar_arr + q_internal_arr),
                       0.0)

    # W → kWh (1時間刻み)
    df["Q_heat_kWh"] = Q_heat / 1000.0
    df["Q_cool_kWh"] = Q_cool / 1000.0
    df["E_hvac_kWh"] = (
        df["Q_heat_kWh"] / cop_heating
        + df["Q_cool_kWh"] / cop_cooling
    )

    # --- 給湯負荷 ---
    # HP給湯機を深夜に集中運転
    hw_power_kw = (hw_daily_kwh / hw_cop) / hw_op_hours  # [kW]
    df["E_hw_kWh"] = 0.0
    for offset in range(hw_op_hours):
        h_target = (hw_hour + offset) % 24
        mask = df.index.hour == h_target
        df.loc[mask, "E_hw_kWh"] = hw_power_kw  # × 1h = kWh

    # --- ベース負荷 ---
    df["E_base_kWh"] = df.index.hour.map(
        lambda h: base_load_w.get(h, 300) / 1000.0  # W → kWh
    ).values

    # --- 合計 ---
    df["E_total_kWh"] = (
        df["E_hvac_kWh"] + df["E_hw_kWh"] + df["E_base_kWh"]
    )

    return df

"""
メインシミュレーションモジュール

config.yaml の設定に基づき、各モジュールを順番に呼び出して
8760時間分の時刻別エネルギー収支を計算する。
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from .battery import BatteryConfig, BatterySimulator
from .load_profile import calc_load_profile
from .nedo_reader import create_sample_ea20, read_ea20
from .product_loader import load_battery, load_panel, load_pcs
from .pv_generation import calc_pv_output
from .solar_calc import calc_tilted_irradiance


def run_simulation(
    config_path: str = "config.yaml",
    ea20_path: Optional[str] = None,
    products_dir: str = "products",
    cfg_override: Optional[dict] = None,
) -> pd.DataFrame:
    """
    シミュレーション全体を実行する。

    Parameters
    ----------
    config_path : str
        設定 YAML ファイルのパス (cfg_override が None の場合に使用)
    ea20_path : str, optional
        EA-20 CSV のパス。None の場合は config.yaml の値を使用。
    products_dir : str
        products/ ディレクトリのパス
    cfg_override : dict, optional
        設定辞書。指定した場合は config_path を読まずにこの値を使用。
        Streamlit などから直接設定を渡す際に使用する。

    Returns
    -------
    pd.DataFrame
        8760行の時刻別シミュレーション結果
    """
    print("=" * 60)
    print(" 太陽光パネル・蓄電池 シミュレーション")
    print("=" * 60)

    cfg = cfg_override if cfg_override is not None else _load_config(config_path)
    loc  = cfg["location"]
    pv_c = cfg["pv_system"]
    obs  = cfg.get("obstruction", {})
    bt_c = cfg["battery"]
    bldg = cfg["building"]
    hvac = bldg["hvac"]
    hw   = bldg["hot_water"]

    # ================================================================
    # 製品仕様の解決
    # ================================================================

    # ---- パネル仕様 ----
    panel_spec = load_panel(pv_c["panel_model"], products_dir)
    panel_count = pv_c["panel_count"]

    # config.yaml に手動上書きがあれば優先する
    panel_eff  = pv_c.get("panel_efficiency",  panel_spec["efficiency"])
    panel_tc   = pv_c.get("temp_coefficient",  panel_spec["temp_coefficient"])
    panel_noct = pv_c.get("noct",              panel_spec["noct"])
    panel_area = panel_spec["area_m2"] * panel_count  # 総面積 [m2]

    # ---- PCS 仕様 ----
    pcs_spec = load_pcs(pv_c["pcs_model"], products_dir)
    pcs_eff  = pcs_spec["efficiency"]

    # DC/AC 比のチェック (任意)
    dc_peak_kw = panel_spec["power_w"] * panel_count / 1000
    dcac_ratio = dc_peak_kw / pcs_spec["rated_kw"]

    # ---- 蓄電池仕様 ----
    batt_spec = load_battery(bt_c["model"], bt_c.get("count", 1), products_dir)
    batt_soc_min = bt_c.get("soc_min", batt_spec["soc_min"])
    batt_soc_max = bt_c.get("soc_max", batt_spec["soc_max"])

    # ---- 構成サマリー表示 ----
    print(f"\n[構成]")
    print(f"  パネル : {panel_spec['model']}")
    print(f"          {panel_spec['power_w']}W x {panel_count}枚"
          f" = {dc_peak_kw:.1f}kW  (面積 {panel_area:.1f}m2)")
    print(f"          効率{panel_eff*100:.1f}%  温度係数{panel_tc*100:.3f}%/K")
    print(f"  PCS   : {pcs_spec['model']}  {pcs_spec['rated_kw']}kW  "
          f"効率{pcs_eff*100:.1f}%")
    print(f"          DC/AC比 = {dcac_ratio:.2f}"
          + (" ← 注意: 1.3超でクリッピング増加" if dcac_ratio > 1.3 else ""))
    print(f"  蓄電池: {batt_spec['model']}")
    print(f"          {batt_spec['capacity_kwh']}kWh  "
          f"最大{batt_spec['max_discharge_kw']}kW  "
          f"往復効率{batt_spec['charge_efficiency']*batt_spec['discharge_efficiency']*100:.1f}%")

    # ================================================================
    # 1. 気象データ読み込み
    # ================================================================
    if ea20_path is None:
        ea20_path = cfg.get("data", {}).get("ea20_path", "data/ea20.csv")

    ea20_path = Path(ea20_path)
    if not ea20_path.exists():
        print(f"\n[注意] EA-20データが見つかりません: {ea20_path}")
        print("       テスト用のサンプルデータを自動生成します。\n")
        create_sample_ea20(str(ea20_path))

    print(f"\n[1/4] 気象データ読み込み中... ({ea20_path.name})")
    weather = read_ea20(str(ea20_path), timezone=loc["timezone"])
    print(f"      -> {len(weather)} 時間分のデータを読み込みました")

    # ================================================================
    # 2. 傾斜面日射量計算
    # ================================================================
    print("\n[2/4] 傾斜面日射量を計算中...")
    horizon = None
    if obs.get("enabled", False):
        horizon = {float(k): float(v) for k, v in obs["horizon_angles"].items()}
        print(f"      -> 包囲角考慮あり ({len(horizon)} 方位点)")

    irr_df = calc_tilted_irradiance(
        weather_df=weather,
        latitude=loc["latitude"],
        longitude=loc["longitude"],
        altitude=loc["altitude"],
        tilt=pv_c["tilt"],
        azimuth=pv_c["azimuth"],
        horizon_angles=horizon,
    )
    poa_annual = irr_df["poa_global"].sum() / 1000
    print(f"      -> 年間傾斜面日射量: {poa_annual:.0f} kWh/m2")

    # ================================================================
    # 3. 発電量計算
    # ================================================================
    print("\n[3/4] 発電量を計算中...")
    pv_df = calc_pv_output(
        irradiance_df=irr_df,
        panel_area=panel_area,
        panel_efficiency=panel_eff,
        pcs_efficiency=pcs_eff,
        temp_coefficient=panel_tc,
        noct=panel_noct,
        system_loss=pv_c["system_loss"],
    )
    annual_gen = pv_df["E_gen_kWh"].sum()
    print(f"      -> 年間発電量: {annual_gen:.0f} kWh")
    print(f"         (参考: 設備利用率 {annual_gen/dc_peak_kw/8760*100:.1f}%)")

    # ================================================================
    # 4. 負荷計算
    # ================================================================
    print("\n[4/4] 負荷プロファイルを計算中...")
    load_df = calc_load_profile(
        weather_df=pv_df,
        floor_area=bldg["floor_area"],
        envelope_area=bldg["envelope_area"],
        ua_value=bldg["ua_value"],
        eta_ac=bldg["eta_ac"],
        heating_setpoint=hvac["heating_setpoint"],
        cooling_setpoint=hvac["cooling_setpoint"],
        cop_heating=hvac["cop_heating"],
        cop_cooling=hvac["cop_cooling"],
        heating_months=hvac["heating_months"],
        cooling_months=hvac["cooling_months"],
        hvac_start_h=hvac["operation_start_h"],
        hvac_end_h=hvac["operation_end_h"],
        occupants=bldg["occupants"],
        internal_gain_per_person_w=bldg["internal_gain_per_person_w"],
        base_load_w={int(k): float(v) for k, v in bldg["base_load_w"].items()},
        hw_daily_kwh=hw["daily_kwh"],
        hw_hour=hw["operation_hour"],
        hw_op_hours=hw.get("operation_hours", 3),
        hw_cop=hw["cop"],
    )
    annual_load = load_df["E_total_kWh"].sum()
    print(f"      -> 年間消費電力量: {annual_load:.0f} kWh")

    # ================================================================
    # 5. 蓄電池シミュレーション
    # ================================================================
    print("\n[蓄電池] 充放電シミュレーション中...")
    batt_cfg = BatteryConfig(
        capacity_kwh=batt_spec["capacity_kwh"],
        charge_efficiency=batt_spec["charge_efficiency"],
        discharge_efficiency=batt_spec["discharge_efficiency"],
        max_charge_kw=batt_spec["max_charge_kw"],
        max_discharge_kw=batt_spec["max_discharge_kw"],
        soc_min=batt_soc_min,
        soc_max=batt_soc_max,
        initial_soc=bt_c.get("initial_soc", 0.5),
    )
    simulator = BatterySimulator(batt_cfg)

    # 余剰電力 [kW] = 発電量 [kWh/h] - 消費量 [kWh/h]
    net_power = load_df["E_gen_kWh"] - load_df["E_total_kWh"]
    batt_result = simulator.run(net_power)

    # ================================================================
    # 結果統合
    # ================================================================
    result = load_df.join(batt_result, how="left")
    result["E_self_kWh"] = (
        result["E_gen_kWh"] - result["grid_export_kw"]
    ).clip(lower=0.0)

    # メタ情報を属性として付加 (output モジュールで使用)
    result.attrs["panel_model"]  = panel_spec["model"]
    result.attrs["panel_count"]  = panel_count
    result.attrs["dc_peak_kw"]   = dc_peak_kw
    result.attrs["pcs_model"]    = pcs_spec["model"]
    result.attrs["battery_model"] = batt_spec["model"]
    result.attrs["battery_kwh"]  = batt_spec["capacity_kwh"]

    print("\n  シミュレーション完了")
    return result


def calc_summary(result: pd.DataFrame) -> dict:
    """年間エネルギー収支のサマリーを計算する"""
    gen     = result["E_gen_kWh"].sum()
    load    = result["E_total_kWh"].sum()
    export  = result["grid_export_kw"].sum()
    import_ = result["grid_import_kw"].sum()
    self_   = result["E_self_kWh"].sum()

    self_rate  = self_ / gen  * 100 if gen  > 0 else 0.0
    indep_rate = self_ / load * 100 if load > 0 else 0.0

    attrs = result.attrs  # メタ情報
    summary = {}

    # 構成情報
    if attrs.get("panel_model"):
        summary["パネル"] = f"{attrs['panel_model']} x {attrs['panel_count']}枚"
        summary["PV 定格"] = f"{attrs['dc_peak_kw']:.1f} kW"
    if attrs.get("pcs_model"):
        summary["PCS"] = attrs["pcs_model"]
    if attrs.get("battery_model"):
        summary["蓄電池"] = f"{attrs['battery_model']} ({attrs['battery_kwh']} kWh)"

    # エネルギー収支
    summary.update({
        "年間発電量 [kWh]":     round(gen,     1),
        "年間消費電力量 [kWh]":  round(load,    1),
        "年間売電量 [kWh]":     round(export,  1),
        "年間買電量 [kWh]":     round(import_, 1),
        "年間自家消費量 [kWh]":  round(self_,   1),
        "自家消費率 [%]":       round(self_rate,  1),
        "自給率 [%]":           round(indep_rate, 1),
        "発電 / 消費 比 [-]":   round(gen / load, 3) if load > 0 else 0.0,
    })
    return summary


def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

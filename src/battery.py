"""
蓄電池シミュレーションモジュール

1時間ステップで充放電を繰り返し、SOCを管理する。

エネルギーフロー:
  余剰 (発電 > 消費):
    → 蓄電池充電 (充電効率 η_c 適用)
    → 満充電なら売電

  不足 (発電 < 消費):
    → 蓄電池放電 (放電効率 η_d 適用)
    → SOC下限なら買電
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class BatteryConfig:
    capacity_kwh: float = 16.4       # 蓄電容量 [kWh]
    charge_efficiency: float = 0.95  # 充電効率 η_c [-]
    discharge_efficiency: float = 0.95  # 放電効率 η_d [-]
    max_charge_kw: float = 5.9       # 最大充電電力 [kW]
    max_discharge_kw: float = 5.9    # 最大放電電力 [kW]
    soc_min: float = 0.05            # SOC 下限 [-]
    soc_max: float = 0.95            # SOC 上限 [-]
    initial_soc: float = 0.50        # 初期 SOC [-]


class BatterySimulator:
    """
    蓄電池の 1時間ステップ 充放電シミュレーター

    用語:
      charge_kw   : 系統(またはPV)から蓄電池へ流れる電力 [kW]
      discharge_kw: 蓄電池から負荷へ供給される電力 [kW]
                    (放電効率の損失を引いた、実際に使える電力)
    """

    def __init__(self, config: BatteryConfig):
        self.cfg = config
        self.soc = config.initial_soc

    # ------------------------------------------------------------------
    # 1ステップ計算
    # ------------------------------------------------------------------

    def step(self, net_kw: float) -> dict:
        """
        1時間分の充放電を計算する。

        Parameters
        ----------
        net_kw : float
            余剰電力 [kW]
            正値 → 余剰 (充電方向)
            負値 → 不足 (放電方向)

        Returns
        -------
        dict
            charge_kw     : 充電電力 [kW]
            discharge_kw  : 蓄電池から供給される電力 [kW]
            grid_export_kw: 売電量 [kW]
            grid_import_kw: 買電量 [kW]
            soc_before    : ステップ前 SOC [-]
            soc_after     : ステップ後 SOC [-]
        """
        cfg = self.cfg
        cap = cfg.capacity_kwh
        soc0 = self.soc

        charge_kw = 0.0
        discharge_kw = 0.0
        grid_export_kw = 0.0
        grid_import_kw = 0.0

        if net_kw >= 0.0:
            # ---- 余剰 → 充電 ----
            # 充電可能な最大エネルギー (SOC上限まで)
            headroom_kwh = (cfg.soc_max - self.soc) * cap
            # 実際に充電する電力 (充電電力×効率=蓄積エネルギー)
            charge_kw = min(net_kw, cfg.max_charge_kw, headroom_kwh)
            charge_kw = max(0.0, charge_kw)

            stored_kwh = charge_kw * cfg.charge_efficiency
            self.soc = min(cfg.soc_max, self.soc + stored_kwh / cap)

            # 充電しきれなかった分は売電
            grid_export_kw = max(0.0, net_kw - charge_kw)

        else:
            # ---- 不足 → 放電 ----
            deficit_kw = -net_kw  # 不足量 [kW] (正値)

            # 放電できる上限 (SOC下限まで)
            # 蓄電池から引き出すエネルギー量 [kWh]
            available_kwh = (self.soc - cfg.soc_min) * cap
            # 放電効率を考慮した、実際に供給できる電力 [kW]
            max_supply_kw = available_kwh * cfg.discharge_efficiency

            # 供給電力 = 不足量と制約の最小値
            discharge_kw = min(deficit_kw, cfg.max_discharge_kw, max_supply_kw)
            discharge_kw = max(0.0, discharge_kw)

            # 蓄電池から引き出すエネルギー (効率で割る)
            drawn_kwh = discharge_kw / cfg.discharge_efficiency
            drawn_kwh = min(drawn_kwh, available_kwh)  # SOC下限の担保
            self.soc = max(cfg.soc_min, self.soc - drawn_kwh / cap)

            # 放電でも不足する分は買電
            grid_import_kw = max(0.0, deficit_kw - discharge_kw)

        return {
            "charge_kw":      round(charge_kw, 4),
            "discharge_kw":   round(discharge_kw, 4),
            "grid_export_kw": round(grid_export_kw, 4),
            "grid_import_kw": round(grid_import_kw, 4),
            "soc_before":     round(soc0, 4),
            "soc_after":      round(self.soc, 4),
        }

    # ------------------------------------------------------------------
    # 全期間一括計算
    # ------------------------------------------------------------------

    def run(self, net_power: pd.Series) -> pd.DataFrame:
        """
        全期間 (8760時間) の充放電シミュレーションを実行する。

        Parameters
        ----------
        net_power : pd.Series
            各時刻の余剰電力 [kW] (発電 - 消費)

        Returns
        -------
        pd.DataFrame
            インデックス: net_power と同じ DatetimeIndex
            列: charge_kw, discharge_kw, grid_export_kw,
                grid_import_kw, soc_before, soc_after
        """
        self.soc = self.cfg.initial_soc  # 初期化

        records = []
        for ts, val in net_power.items():
            r = self.step(float(val))
            r["timestamp"] = ts
            records.append(r)

        return pd.DataFrame(records).set_index("timestamp")

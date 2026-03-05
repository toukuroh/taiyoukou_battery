"""
太陽位置計算・傾斜面日射量計算モジュール

pvlib を使用して:
  1. 太陽位置 (高度角・方位角) を計算
  2. 包囲角 (障害物による遮蔽) を適用
  3. 傾斜面日射量を計算 (Hay-Davies モデル)
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import pvlib


def calc_tilted_irradiance(
    weather_df: pd.DataFrame,
    latitude: float,
    longitude: float,
    altitude: float,
    tilt: float,
    azimuth: float,
    horizon_angles: Optional[Dict[float, float]] = None,
) -> pd.DataFrame:
    """
    傾斜面日射量を計算する。

    Parameters
    ----------
    weather_df : pd.DataFrame
        GHI, DNI, DHI, Tair, WS を含む気象DataFrame
        (DatetimeIndex, tz付き)
    latitude, longitude : float
        緯度・経度 [deg]
    altitude : float
        標高 [m]
    tilt : float
        パネル傾斜角 [deg] (水平=0, 垂直=90)
    azimuth : float
        パネル方位角 [deg] (pvlib基準: 0=北, 90=東, 180=南, 270=西)
    horizon_angles : dict, optional
        包囲角 {方位[deg]: 仰角[deg]}
        例: {0:0, 45:0, 90:5, 135:10, 180:2, 225:10, 270:5, 315:0}

    Returns
    -------
    pd.DataFrame
        追加列:
        - solar_elevation  : 太陽高度角 [deg]
        - solar_azimuth    : 太陽方位角 [deg]
        - obstructed       : 遮蔽フラグ (True=直達日射が遮られる)
        - dni_eff          : 有効直達日射量 [W/m²]
        - poa_global       : 傾斜面全日射量 [W/m²]
        - poa_direct       : 傾斜面直達成分 [W/m²]
        - poa_sky_diffuse  : 傾斜面天空散乱成分 [W/m²]
        - poa_ground       : 傾斜面地面反射成分 [W/m²]
    """
    loc = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        tz=weather_df.index.tz,
        altitude=altitude,
    )

    # 太陽位置計算
    solar_pos = loc.get_solarposition(weather_df.index)

    # 包囲角による遮蔽処理
    dni_eff = weather_df["DNI"].copy().astype(float)
    obstructed = pd.Series(False, index=weather_df.index)

    if horizon_angles:
        obstructed = _make_obstruction_mask(
            solar_pos["elevation"],
            solar_pos["azimuth"],
            horizon_angles,
        )
        dni_eff[obstructed] = 0.0

    # 法線面外部日射量 (Hay-Davies モデルに必要)
    dni_extra = pvlib.irradiance.get_extra_radiation(weather_df.index)

    # 傾斜面日射量 (Hay-Davies モデル)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        dni=dni_eff,
        ghi=weather_df["GHI"],
        dhi=weather_df["DHI"],
        dni_extra=dni_extra,
        model="haydavies",
        albedo=0.2,  # 地面アルベド (草地・アスファルト: 0.2)
    )

    df = weather_df.copy()
    df["solar_elevation"] = solar_pos["elevation"]
    df["solar_azimuth"] = solar_pos["azimuth"]
    df["obstructed"] = obstructed
    df["dni_eff"] = dni_eff
    df["poa_global"] = poa["poa_global"].fillna(0.0).clip(lower=0.0)
    df["poa_direct"] = poa["poa_direct"].fillna(0.0).clip(lower=0.0)
    df["poa_sky_diffuse"] = poa["poa_sky_diffuse"].fillna(0.0).clip(lower=0.0)
    df["poa_ground"] = poa["poa_ground_diffuse"].fillna(0.0).clip(lower=0.0)

    return df


# ---------------------------------------------------------------------------
# 包囲角処理
# ---------------------------------------------------------------------------

def _make_obstruction_mask(
    solar_elevation: pd.Series,
    solar_azimuth: pd.Series,
    horizon_angles: Dict[float, float],
) -> pd.Series:
    """
    各時刻で太陽が包囲角（障害物）の背後にあるか判定する。
    NumPy ベクトル演算で全 8760 時間を一括処理する。

    Returns
    -------
    pd.Series[bool]
        True のとき、その時刻は直達日射が遮蔽される
    """
    az_sorted = np.array(sorted(horizon_angles.keys()), dtype=float)
    elv_sorted = np.array([horizon_angles[a] for a in az_sorted], dtype=float)

    # 360° 折り返しを処理するため、末尾に [0°+360°, elv[0]] を追加
    az_ext  = np.append(az_sorted, az_sorted[0] + 360.0)
    elv_ext = np.append(elv_sorted, elv_sorted[0])

    elev_arr = solar_elevation.to_numpy(dtype=float)
    az_arr   = solar_azimuth.to_numpy(dtype=float) % 360.0

    # 昼間 (太陽高度 > 0) のみ判定
    daytime = elev_arr > 0.0

    # 各時刻の方位に対応する補間区間を一括検索
    idx = np.searchsorted(az_ext, az_arr, side="right")
    idx = np.clip(idx, 1, len(az_ext) - 1)

    az0  = az_ext[idx - 1]
    az1  = az_ext[idx]
    elv0 = elv_ext[idx - 1]
    elv1 = elv_ext[idx]

    denom      = az1 - az0
    t          = np.where(denom != 0.0, (az_arr - az0) / denom, 0.0)
    horizon_elv = elv0 + t * (elv1 - elv0)

    obstructed = daytime & (elev_arr < horizon_elv)
    return pd.Series(obstructed, index=solar_elevation.index)

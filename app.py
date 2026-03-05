"""
太陽光パネル・蓄電池 年間エネルギーシミュレーター
Streamlit Web アプリ

起動方法:
    streamlit run app.py
"""

import contextlib
import io
import sys
import tempfile
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# src を Python パスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.output import to_excel_bytes
from src.product_loader import list_batteries, list_panels, list_pcs
from src.simulation import calc_summary, run_simulation

# ─────────────────────────────────────────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="太陽光・蓄電池 シミュレーター",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("☀️ 太陽光・蓄電池 年間エネルギーシミュレーター")
st.caption("高気密高断熱住宅向け · NEDO EA-20 気象データ · 8760時間シミュレーション")

# ─────────────────────────────────────────────────────────────────────────────
# 製品リスト (起動時にキャッシュ)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def _get_products():
    return {
        "panels":    list_panels(),
        "batteries": list_batteries(),
        "pcs":       list_pcs(),
    }

products = _get_products()

# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────
LOCATION_PRESETS = {
    "東京":     {"lat": 35.69,  "lon": 139.69, "alt": 25},
    "大阪":     {"lat": 34.69,  "lon": 135.50, "alt": 10},
    "名古屋":   {"lat": 35.17,  "lon": 136.91, "alt": 51},
    "札幌":     {"lat": 43.06,  "lon": 141.35, "alt": 17},
    "福岡":     {"lat": 33.60,  "lon": 130.40, "alt": 3},
    "那覇":     {"lat": 26.20,  "lon": 127.68, "alt": 28},
    "手動入力": None,
}

DEFAULT_BASE_LOAD_W = {
    0: 150,  1: 150,  2: 150,  3: 150,  4: 150,  5: 200,
    6: 900,  7: 1100, 8: 700,  9: 400,  10: 350, 11: 400,
    12: 600, 13: 350, 14: 350, 15: 350, 16: 400, 17: 900,
    18: 1300,19: 1200,20: 1000,21: 900, 22: 700, 23: 300,
}

HORIZON_DEFAULTS = {0: 0.0, 45: 0.0, 90: 3.0, 135: 5.0,
                    180: 2.0, 225: 5.0, 270: 3.0, 315: 0.0}

# ─────────────────────────────────────────────────────────────────────────────
# サイドバー
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ シミュレーション設定")

    # ── 地点 ──
    with st.expander("📍 地点", expanded=True):
        location_name = st.selectbox("地点プリセット", list(LOCATION_PRESETS.keys()), index=0)
        if location_name == "手動入力":
            lat = st.number_input("緯度 [°]", value=35.69, format="%.4f")
            lon = st.number_input("経度 [°]", value=139.69, format="%.4f")
            alt = st.number_input("標高 [m]", value=25)
        else:
            p = LOCATION_PRESETS[location_name]
            lat, lon, alt = p["lat"], p["lon"], p["alt"]
            st.caption(f"緯度 {lat}°N / 経度 {lon}°E / 標高 {alt}m")

    # ── 太陽光パネル ──
    with st.expander("☀️ 太陽光パネル", expanded=True):
        default_panel = "Jinko Tiger Neo JKM415N-54HL4-V"
        panel_idx = (products["panels"].index(default_panel)
                     if default_panel in products["panels"] else 0)
        panel_model = st.selectbox("パネル製品", products["panels"], index=panel_idx)
        panel_count = st.number_input("枚数", min_value=1, max_value=100, value=14, step=1)

        default_pcs = "Omron KP-LM-KU-55-JP"
        pcs_idx = (products["pcs"].index(default_pcs)
                   if default_pcs in products["pcs"] else 0)
        pcs_model = st.selectbox("PCS 製品", products["pcs"], index=pcs_idx)

        tilt     = st.slider("傾斜角 [°]", 0, 90, 30)
        azimuth  = st.slider("方位角 [°]  (180=南)", 0, 359, 180)
        system_loss = st.slider("システムロス", 0.00, 0.20, 0.05, 0.01, format="%.2f")

    # ── 蓄電池 ──
    with st.expander("🔋 蓄電池", expanded=True):
        default_batt = "Sharp JH-WB1821"
        batt_idx = (products["batteries"].index(default_batt)
                    if default_batt in products["batteries"] else 0)
        battery_model = st.selectbox("蓄電池製品", products["batteries"], index=batt_idx)
        battery_count = st.number_input("台数", min_value=1, max_value=10, value=1, step=1)
        initial_soc   = st.slider("初期 SOC", 0.0, 1.0, 0.50, 0.05, format="%.2f")

    # ── 建物 ──
    with st.expander("🏠 建物スペック", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            floor_area    = st.number_input("床面積 [m²]", value=120.0, step=5.0)
            ua_value      = st.number_input("UA値 [W/m²K]", value=0.46, step=0.01,
                                             format="%.2f", help="HEAT20 G2≦0.46 / ZEH≦0.60")
            heating_sp    = st.number_input("暖房設定温度 [°C]", value=20.0, step=1.0)
            cop_heating   = st.number_input("暖房 COP", value=4.5, step=0.1, format="%.1f")
        with c2:
            envelope_area = st.number_input("外皮面積 [m²]", value=350.0, step=10.0,
                                             help="外壁+屋根+床の合計面積")
            eta_ac        = st.number_input("ηAC [%]", value=2.8, step=0.1, format="%.1f",
                                             help="冷房期日射熱取得率 (省エネ計算書の値)")
            cooling_sp    = st.number_input("冷房設定温度 [°C]", value=26.0, step=1.0)
            cop_cooling   = st.number_input("冷房 COP", value=5.5, step=0.1, format="%.1f")
        occupants = st.number_input("居住者数", min_value=1, max_value=10, value=3, step=1)

    # ── 給湯 ──
    with st.expander("🚿 給湯 (ヒートポンプ)", expanded=False):
        hw_daily = st.number_input("給湯消費電力量 [kWh/日]", value=3.5,
                                    step=0.1, format="%.1f", help="HP給湯機の消費電力(入力)")
        hw_cop   = st.number_input("給湯 COP", value=3.5, step=0.1, format="%.1f")

    # ── 包囲角 (障害物) ──
    hz = {}
    with st.expander("🌳 包囲角（障害物）", expanded=False):
        obstruction_enabled = st.toggle("障害物による遮蔽を考慮する", value=True)
        if obstruction_enabled:
            st.caption("各方位での障害物の仰角 [°] (0=遮蔽なし)")
            hc1, hc2 = st.columns(2)
            directions = [
                (0,   "北(0°)"),   (45,  "北東(45°)"),
                (90,  "東(90°)"),  (135, "南東(135°)"),
                (180, "南(180°)"), (225, "南西(225°)"),
                (270, "西(270°)"), (315, "北西(315°)"),
            ]
            for i, (deg, label) in enumerate(directions):
                col = hc1 if i % 2 == 0 else hc2
                hz[deg] = col.number_input(
                    label, min_value=0.0, max_value=45.0,
                    value=HORIZON_DEFAULTS.get(deg, 0.0),
                    step=0.5, format="%.1f", key=f"hz_{deg}",
                )

    # ── 気象データ ──
    with st.expander("📊 気象データ (NEDO EA-20)", expanded=False):
        ea20_file = st.file_uploader(
            "EA-20 CSV をアップロード", type=["csv"],
            help="NEDO METPV-20 からダウンロードした拡張アメダスデータ",
        )
        if ea20_file:
            st.success(f"✓ {ea20_file.name}  ({ea20_file.size/1024:.1f} KB)")
        else:
            st.info("未選択: テスト用サンプルデータを自動生成します")

    st.divider()
    run_btn = st.button("▶ シミュレーション実行", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# シミュレーション実行
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    # 設定辞書を組み立て
    obs_cfg = {
        "enabled": obstruction_enabled,
        "horizon_angles": {str(k): float(v) for k, v in hz.items()} if obstruction_enabled else {},
    }

    cfg_dict = {
        "location": {
            "name":      location_name,
            "latitude":  float(lat),
            "longitude": float(lon),
            "altitude":  float(alt),
            "timezone":  "Asia/Tokyo",
        },
        "pv_system": {
            "panel_model":  panel_model,
            "panel_count":  int(panel_count),
            "pcs_model":    pcs_model,
            "tilt":         float(tilt),
            "azimuth":      float(azimuth),
            "system_loss":  float(system_loss),
        },
        "obstruction": obs_cfg,
        "battery": {
            "model":       battery_model,
            "count":       int(battery_count),
            "initial_soc": float(initial_soc),
        },
        "building": {
            "floor_area":                 float(floor_area),
            "envelope_area":              float(envelope_area),
            "ua_value":                   float(ua_value),
            "eta_ac":                     float(eta_ac),
            "occupants":                  int(occupants),
            "internal_gain_per_person_w": 80,
            "hvac": {
                "heating_setpoint": float(heating_sp),
                "cooling_setpoint": float(cooling_sp),
                "cop_heating":      float(cop_heating),
                "cop_cooling":      float(cop_cooling),
                "heating_months":   [11, 12, 1, 2, 3],
                "cooling_months":   [6, 7, 8, 9],
                "operation_start_h": 6,
                "operation_end_h":   23,
            },
            "base_load_w": DEFAULT_BASE_LOAD_W,
            "hot_water": {
                "daily_kwh":       float(hw_daily),
                "operation_hour":  1,
                "operation_hours": 3,
                "cop":             float(hw_cop),
            },
        },
    }

    # EA-20 をテンポラリファイルに保存
    ea20_path = None
    _tmp_path = None
    if ea20_file:
        _tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        _tmp.write(ea20_file.read())
        _tmp.close()
        ea20_path = _tmp.name
        _tmp_path = _tmp.name

    prog = st.progress(0, text="シミュレーション準備中...")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            prog.progress(20, text="気象データ・発電量を計算中...")
            result = run_simulation(cfg_override=cfg_dict, ea20_path=ea20_path)
            prog.progress(90, text="集計中...")
            summary = calc_summary(result)
        prog.progress(100, text="完了!")
        st.session_state["result"]  = result
        st.session_state["summary"] = summary
        prog.empty()
        st.success("✅ シミュレーション完了!")
    except Exception as e:
        prog.empty()
        st.error(f"エラーが発生しました:\n{e}")
        st.stop()
    finally:
        if _tmp_path:
            try:
                os.unlink(_tmp_path)
            except OSError:
                pass

# ─────────────────────────────────────────────────────────────────────────────
# 初期画面 (未実行時)
# ─────────────────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.info("👈 左のサイドバーで設定を入力し、「▶ シミュレーション実行」を押してください。")
    with st.expander("💡 使い方"):
        st.markdown("""
1. **地点** を選択（東京・大阪・名古屋など）
2. **太陽光パネル** の製品と枚数を選択
3. **蓄電池** の製品と台数を選択
4. 必要に応じて建物スペック・給湯・障害物を調整
5. **EA-20気象データ** があればアップロード（なければサンプルデータを自動生成）
6. 「▶ シミュレーション実行」を押す
        """)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 結果表示
# ─────────────────────────────────────────────────────────────────────────────
result  = st.session_state["result"]
summary = st.session_state["summary"]

# ── KPI カード ──
st.subheader("📊 年間エネルギー収支")
kc = st.columns(6)
kc[0].metric("発電量",    f"{summary['年間発電量 [kWh]']:,.0f} kWh")
kc[1].metric("消費量",    f"{summary['年間消費電力量 [kWh]']:,.0f} kWh")
kc[2].metric("自給率",    f"{summary['自給率 [%]']:.1f} %",
             help="自家消費 ÷ 消費量")
kc[3].metric("自家消費率", f"{summary['自家消費率 [%]']:.1f} %",
             help="自家消費 ÷ 発電量")
kc[4].metric("売電量",    f"{summary['年間売電量 [kWh]']:,.0f} kWh")
kc[5].metric("買電量",    f"{summary['年間買電量 [kWh]']:,.0f} kWh")

# ── システム構成 ──
with st.expander("🔧 システム構成"):
    for key in ["パネル", "PV 定格", "PCS", "蓄電池"]:
        if key in summary:
            st.write(f"**{key}**: {summary[key]}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# チャートタブ
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📅 月別収支", "📈 週間詳細", "🔋 SOC ヒートマップ", "⚡ エネルギーフロー"])

# ────────────── Tab1: 月別 ──────────────
with tab1:
    monthly = result.resample("ME").sum(numeric_only=True)
    months_str = [f"{m.month}月" for m in monthly.index]

    fig_m = go.Figure()
    fig_m.add_bar(x=months_str, y=monthly["E_gen_kWh"].round(0),
                  name="発電量 [kWh]", marker_color="#F4A020")
    fig_m.add_bar(x=months_str, y=monthly["E_total_kWh"].round(0),
                  name="消費量 [kWh]", marker_color="#4A90D9")
    fig_m.add_bar(x=months_str, y=monthly["grid_export_kw"].round(0),
                  name="売電 [kWh]", marker_color="#7ED321", opacity=0.75)
    fig_m.add_bar(x=months_str, y=monthly["grid_import_kw"].round(0),
                  name="買電 [kWh]", marker_color="#E74C3C", opacity=0.75)
    fig_m.update_layout(
        barmode="group", title="月別エネルギー収支",
        yaxis_title="電力量 [kWh]", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_m, use_container_width=True)

    # 月別内訳テーブル
    st.dataframe(
        monthly[["E_gen_kWh", "E_total_kWh", "E_hvac_kWh", "E_hw_kWh",
                  "grid_export_kw", "grid_import_kw", "E_self_kWh"]]
        .rename(columns={
            "E_gen_kWh":      "発電[kWh]",
            "E_total_kWh":    "消費[kWh]",
            "E_hvac_kWh":     "空調[kWh]",
            "E_hw_kWh":       "給湯[kWh]",
            "grid_export_kw": "売電[kWh]",
            "grid_import_kw": "買電[kWh]",
            "E_self_kWh":     "自家消費[kWh]",
        }).round(0).astype(int),
        use_container_width=True,
    )

# ────────────── Tab2: 週間詳細 ──────────────
with tab2:
    WEEK_OPTS = {
        "1月 第2週 (冬)":  "2016-01-11",
        "4月 第2週 (春)":  "2016-04-11",
        "7月 第3週 (夏)":  "2016-07-18",
        "10月 第2週 (秋)": "2016-10-10",
    }
    week_label = st.selectbox("表示週を選択", list(WEEK_OPTS.keys()), key="week_sel")
    ws = pd.Timestamp(WEEK_OPTS[week_label], tz=result.index.tz)
    we = ws + pd.Timedelta(days=7)
    wd = result.loc[ws:we]

    fig_w = go.Figure()
    fig_w.add_scatter(
        x=wd.index, y=wd["E_gen_kWh"],
        name="発電量 [kWh]", line_color="#F4A020",
        fill="tozeroy", fillcolor="rgba(244,160,32,0.12)",
    )
    fig_w.add_scatter(
        x=wd.index, y=wd["E_total_kWh"],
        name="消費量 [kWh]", line_color="#4A90D9",
    )
    fig_w.add_scatter(
        x=wd.index, y=wd["soc_after"] * 100,
        name="SOC [%]", line_color="#27AE60", line_dash="dot",
        yaxis="y2",
    )
    fig_w.update_layout(
        title="週間詳細 (発電量 / 消費量 / SOC)",
        yaxis =dict(title="電力量 [kWh]"),
        yaxis2=dict(title="SOC [%]", overlaying="y", side="right", range=[0, 110]),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_w, use_container_width=True)

# ────────────── Tab3: SOC ヒートマップ ──────────────
with tab3:
    soc_tmp = result[["soc_after"]].copy()
    soc_tmp["month"] = soc_tmp.index.month
    soc_tmp["hour"]  = soc_tmp.index.hour
    pivot = soc_tmp.pivot_table(
        values="soc_after", index="month", columns="hour", aggfunc="mean"
    )
    month_labels = ["1月","2月","3月","4月","5月","6月",
                    "7月","8月","9月","10月","11月","12月"]

    fig_h = go.Figure(data=go.Heatmap(
        z=(pivot.values * 100).round(1),
        x=[f"{h}時" for h in range(24)],
        y=month_labels,
        colorscale="RdYlGn",
        zmin=0, zmax=100,
        colorbar=dict(title="SOC [%]"),
    ))
    fig_h.update_layout(
        title="月別・時刻別 平均 SOC [%]",
        xaxis_title="時刻",
        yaxis_title="月",
        height=440,
    )
    st.plotly_chart(fig_h, use_container_width=True)
    st.caption("色が緑=蓄電池が満充電、赤=放電済みを示します。")

# ────────────── Tab4: エネルギーフロー (Sankey) ──────────────
with tab4:
    gen_     = float(summary["年間発電量 [kWh]"])
    load_    = float(summary["年間消費電力量 [kWh]"])
    export_  = float(summary["年間売電量 [kWh]"])
    import_  = float(summary["年間買電量 [kWh]"])
    charge_  = float(result["charge_kw"].sum())
    disc_    = float(result["discharge_kw"].sum())
    direct_  = max(0.01, gen_ - export_ - charge_)

    # Nodes: 0=発電  1=売電  2=蓄電池  3=消費量  4=買電
    fig_s = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20, thickness=22,
            label=["発電", "売電 (系統へ)", "蓄電池", "消費量", "買電 (系統から)"],
            color=["#F4A020", "#7ED321", "#9B59B6", "#4A90D9", "#E74C3C"],
        ),
        link=dict(
            source=[0,         0,         0,       2,       4      ],
            target=[1,         2,         3,       3,       3      ],
            value =[max(0.01, export_),
                    max(0.01, charge_),
                    max(0.01, direct_),
                    max(0.01, disc_),
                    max(0.01, import_)],
            color =["rgba(126,211,33,0.35)",
                    "rgba(155,89,182,0.35)",
                    "rgba(244,160,32,0.35)",
                    "rgba(155,89,182,0.35)",
                    "rgba(231,76,60,0.35)"],
        ),
    )])
    fig_s.update_layout(
        title_text="年間エネルギーフロー [kWh]",
        height=420,
    )
    st.plotly_chart(fig_s, use_container_width=True)

    # 簡易テーブル
    flow_df = pd.DataFrame({
        "フロー": ["発電 → 直接自家消費", "発電 → 蓄電池充電",
                   "蓄電池 → 放電消費",   "発電 → 売電",  "系統 → 買電"],
        "[kWh]":  [round(direct_, 0), round(charge_, 0),
                   round(disc_, 0),   round(export_, 0), round(import_, 0)],
    })
    st.dataframe(flow_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# ダウンロード
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("💾 結果ダウンロード")
dc1, dc2 = st.columns(2)

with dc1:
    csv_cols = [c for c in [
        "GHI", "Tair", "poa_global", "E_gen_kWh",
        "E_total_kWh", "E_hvac_kWh", "E_hw_kWh", "E_base_kWh",
        "charge_kw", "discharge_kw", "grid_export_kw", "grid_import_kw",
        "soc_after", "E_self_kWh",
    ] if c in result.columns]
    csv_bytes = result[csv_cols].to_csv(float_format="%.4f").encode("utf-8-sig")
    st.download_button(
        "📥 CSV ダウンロード", data=csv_bytes,
        file_name="simulation_result.csv", mime="text/csv",
        use_container_width=True,
    )

with dc2:
    try:
        excel_bytes = to_excel_bytes(result, summary)
        st.download_button(
            "📥 Excel ダウンロード", data=excel_bytes,
            file_name="simulation_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Excel エクスポート失敗: {e}")

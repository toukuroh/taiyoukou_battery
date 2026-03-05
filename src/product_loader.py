"""
製品データベース 読み込み・検索モジュール

products/ ディレクトリの YAML ファイルから
太陽光パネル / 蓄電池 / PCS の仕様を取得する。
"""

from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def load_panel(model: str, products_dir: str = "products") -> dict:
    """
    太陽光パネルの仕様を取得する。

    Parameters
    ----------
    model : str
        製品名 (products/solar_panels.yaml の model フィールドと一致)
    products_dir : str
        products/ ディレクトリのパス

    Returns
    -------
    dict
        パネル仕様辞書 (efficiency, area_m2, temp_coefficient, noct, ...)

    Raises
    ------
    ValueError
        製品が見つからない場合 (候補一覧を表示)
    """
    db = _load_yaml(products_dir, "solar_panels.yaml")
    return _find(db["panels"], model, "solar_panels.yaml")


def load_battery(model: str, count: int = 1, products_dir: str = "products") -> dict:
    """
    蓄電池の仕様を取得する。count > 1 の場合は容量をスケールする。

    Parameters
    ----------
    model : str
        製品名 (products/batteries.yaml の model フィールドと一致)
    count : int
        台数。capacity_kwh は count 倍になる。
        max_charge_kw / max_discharge_kw は scalable_power: true のときのみ倍にする。
    products_dir : str
        products/ ディレクトリのパス

    Returns
    -------
    dict
        蓄電池仕様辞書 (capacity_kwh, charge_efficiency, ...)
    """
    db = _load_yaml(products_dir, "batteries.yaml")
    spec = _find(db["batteries"], model, "batteries.yaml")
    spec = dict(spec)  # コピー

    if count > 1:
        spec["capacity_kwh"] = spec["capacity_kwh"] * count
        if spec.get("scalable_power", False):
            spec["max_charge_kw"] = spec["max_charge_kw"] * count
            spec["max_discharge_kw"] = spec["max_discharge_kw"] * count
        # scalable_power: false のとき最大電力は変わらない

    return spec


def load_pcs(model: str, products_dir: str = "products") -> dict:
    """PCS の仕様を取得する"""
    db = _load_yaml(products_dir, "pcs.yaml")
    return _find(db["pcs"], model, "pcs.yaml")


# ---------------------------------------------------------------------------
# 一覧表示
# ---------------------------------------------------------------------------

def list_panels(products_dir: str = "products") -> list:
    """利用可能なパネル製品名の一覧を返す"""
    db = _load_yaml(products_dir, "solar_panels.yaml")
    return [p["model"] for p in db["panels"]]


def list_batteries(products_dir: str = "products") -> list:
    """利用可能な蓄電池製品名の一覧を返す"""
    db = _load_yaml(products_dir, "batteries.yaml")
    return [b["model"] for b in db["batteries"]]


def list_pcs(products_dir: str = "products") -> list:
    """利用可能な PCS 製品名の一覧を返す"""
    db = _load_yaml(products_dir, "pcs.yaml")
    return [p["model"] for p in db["pcs"]]


def print_catalog(products_dir: str = "products") -> None:
    """製品カタログを見やすく表示する"""
    _print_panel_catalog(products_dir)
    _print_battery_catalog(products_dir)
    _print_pcs_catalog(products_dir)


# ---------------------------------------------------------------------------
# 内部処理
# ---------------------------------------------------------------------------

def _load_yaml(products_dir: str, filename: str) -> dict:
    path = Path(products_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"製品DBファイルが見つかりません: {path}\n"
            "products/ ディレクトリを確認してください。"
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find(items: list, model: str, source: str) -> dict:
    """model 名で製品を検索。見つからない場合は候補を表示してエラー。"""
    for item in items:
        if item["model"] == model:
            return item

    # 見つからなかった場合: 候補一覧を表示
    candidates = [it["model"] for it in items]
    candidates_str = "\n  ".join(candidates)
    raise ValueError(
        f"製品 '{model}' が {source} に見つかりません。\n"
        f"利用可能な製品:\n  {candidates_str}\n"
        "\nconfig.yaml の model 名を上記から選んでください。"
    )


def _print_panel_catalog(products_dir: str) -> None:
    db = _load_yaml(products_dir, "solar_panels.yaml")
    print("\n" + "=" * 70)
    print(" 太陽光パネル 製品一覧")
    print("=" * 70)
    print(f"  {'製品名':<40} {'出力':>5}  {'効率':>5}  {'温度係数':>8}  {'参考価格':>10}")
    print("-" * 70)
    for p in db["panels"]:
        tc = f"{p['temp_coefficient']*100:.2f}%/K"
        print(
            f"  {p['model']:<40} {p['power_w']:>4}W"
            f"  {p['efficiency']*100:>4.1f}%"
            f"  {tc:>8}"
            f"  {p['price_yen']:>8,}円"
        )


def _print_battery_catalog(products_dir: str) -> None:
    db = _load_yaml(products_dir, "batteries.yaml")
    print("\n" + "=" * 70)
    print(" 蓄電池 製品一覧")
    print("=" * 70)
    print(f"  {'製品名':<36} {'容量':>6}  {'最大kW':>6}  {'往復効率':>8}  {'参考価格':>10}")
    print("-" * 70)
    for b in db["batteries"]:
        rt = b["charge_efficiency"] * b["discharge_efficiency"]
        print(
            f"  {b['model']:<36} {b['capacity_kwh']:>5.1f}kWh"
            f"  {b['max_discharge_kw']:>4.1f}kW"
            f"  {rt*100:>6.1f}%"
            f"  {b['price_yen']:>8,}円"
        )


def _print_pcs_catalog(products_dir: str) -> None:
    db = _load_yaml(products_dir, "pcs.yaml")
    print("\n" + "=" * 70)
    print(" パワコン (PCS) 製品一覧")
    print("=" * 70)
    print(f"  {'製品名':<36} {'定格':>5}  {'効率':>5}  {'参考価格':>10}")
    print("-" * 70)
    for p in db["pcs"]:
        print(
            f"  {p['model']:<36} {p['rated_kw']:>4.1f}kW"
            f"  {p['efficiency']*100:>4.1f}%"
            f"  {p['price_yen']:>8,}円"
        )

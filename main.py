"""
太陽光パネル・蓄電池 シミュレーション
高気密高断熱住宅向け 年間 8760時間 エネルギー収支計算

使い方:
    python main.py                        # デフォルト設定 (config.yaml)
    python main.py --config my_config.yaml
    python main.py --ea20 data/tokyo_poor.csv  # 寡照年データを直接指定

データ配置:
    data/ea20.csv  ← NEDO METPV-20 からダウンロードしたCSVを置く
                     (なければサンプルデータを自動生成)
"""

import argparse
import io
import sys
from pathlib import Path

# Windows でも日本語を正しく出力する
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(
        description="太陽光・蓄電池 年間エネルギーシミュレーション"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="設定ファイルのパス"
    )
    parser.add_argument(
        "--ea20", default=None, help="EA-20 気象データ CSV のパス"
    )
    parser.add_argument(
        "--list", action="store_true", help="登録済み製品一覧を表示して終了"
    )
    args = parser.parse_args()

    # 製品一覧表示モード
    if args.list:
        from src.product_loader import print_catalog
        print_catalog()
        return

    # sys.path に src を追加
    sys.path.insert(0, str(Path(__file__).parent))

    from src.simulation import calc_summary, run_simulation
    from src.output import save_csv, save_excel

    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = cfg.get("output", {})

    # --- シミュレーション実行 ---
    result = run_simulation(
        config_path=args.config,
        ea20_path=args.ea20,
    )

    # --- サマリー表示 ---
    summary = calc_summary(result)
    print("\n" + "=" * 40)
    print(" 年間エネルギー収支 サマリー")
    print("=" * 40)
    for key, val in summary.items():
        print(f"  {key:<22}: {val}")

    # --- 出力 ---
    print()
    save_csv(result, out.get("csv_path", "output/simulation_result.csv"))
    save_excel(result, summary, out.get("excel_path", "output/simulation_result.xlsx"))

    print("\n完了!")


if __name__ == "__main__":
    main()

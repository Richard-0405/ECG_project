"""
批次生成 PTB-XL 的 ECG 圖片，並依 5 類分類儲存到不同資料夾。

使用方式：
    放在 /data/ecg_project/ 根目錄下，直接執行：
        python generate_5class_images.py

類別定義：
    Class 0: Normal        -> PTB-XL 代碼 NORM
    Class 1: AF            -> PTB-XL 代碼 AFIB, AFLT
    Class 2: VFL           -> PTB-XL 沒有對應（會是空資料夾，正常現象）
    Class 3: SVTA          -> PTB-XL 代碼 SVTAC, PSVT
    Class 4: Others        -> 以上都不是的其他異常

風格：fully_random + wrinkles + augment + hw_text
     每張圖都會有隨機的旋轉、雜訊、裁切、色溫、皺摺、手寫註記，
     模擬手機拍紙本 ECG 的多樣性。
"""

import ast
import subprocess
import sys
from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# 路徑設定
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
PTB_XL_BASE = PROJECT_ROOT / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
METADATA_CSV = PTB_XL_BASE / 'ptbxl_database.csv'
OUTPUT_ROOT = PROJECT_ROOT / 'ecg_5class_images'
GENERATOR = PROJECT_ROOT / 'ecg-image-kit' / 'codes' / 'ecg-image-generator' / 'gen_ecg_image_from_data.py'

# ============================================================
# 執行設定（先測試再開全量）
# ============================================================
USE_HIGH_RES = False        # False=用 records100(_lr)；True=用 records500(_hr)
TIMEOUT_PER_FILE = 90       # 單張最長秒數
LIMIT_PER_CLASS = 3         # 每類先跑幾筆測試；設 None 跑全部

# ============================================================
# 類別定義
# ============================================================
CLASS_NAMES = {
    0: 'Normal',
    1: 'AF',
    2: 'VFL',
    3: 'SVTA',
    4: 'Others',
}

# 每個 class 對應的 PTB-XL SCP 代碼
CLASS_SCP_CODES = {
    0: {'NORM'},                # Normal
    1: {'AFIB', 'AFLT'},        # 心房顫動/撲動
    2: set(),                   # VFL 在 PTB-XL 無對應
    3: {'SVTAC', 'PSVT'},       # 上心室頻脈
    # Class 4 Others 是 fallback（沒匹配上面任何一類的「異常」）
}


def classify_ecg(scp_codes_dict):
    """
    把一筆 ECG 的 scp_codes (dict) 分到 5 類之一。

    規則（依優先順序）：
    1. 有 AFIB/AFLT 就算 Class 1 AF
    2. 有 SVTAC/PSVT 就算 Class 3 SVTA
    3. 有 NORM 且沒有上述異常 → Class 0 Normal
    4. 其他 → Class 4 Others
    """
    codes = set(scp_codes_dict.keys())

    if codes & CLASS_SCP_CODES[1]:      # 先看有沒有 AF 相關
        return 1
    if codes & CLASS_SCP_CODES[3]:      # 再看 SVTA
        return 3
    if 'NORM' in codes:                 # 再看是不是正常
        return 0
    return 4                            # 都不是 → 其他


def main():
    # 檢查必要路徑
    assert METADATA_CSV.exists(), f"找不到 metadata: {METADATA_CSV}"
    assert GENERATOR.exists(), f"找不到生成器: {GENERATOR}"

    # 建立 5 個類別資料夾
    class_dirs = {}
    for cid, cname in CLASS_NAMES.items():
        d = OUTPUT_ROOT / f'class_{cid}_{cname}'
        d.mkdir(parents=True, exist_ok=True)
        class_dirs[cid] = d

    # 讀 metadata
    print(f"讀取 {METADATA_CSV.name} ...")
    df = pd.read_csv(METADATA_CSV)
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

    # 每筆 ECG 分類
    df['class_id'] = df['scp_codes'].apply(classify_ecg)

    # 統計
    class_counts = Counter(df['class_id'])
    print("\n=== 分類結果 ===")
    for cid in sorted(CLASS_NAMES.keys()):
        print(f"  Class {cid} ({CLASS_NAMES[cid]:8s}): {class_counts.get(cid, 0):5d} 筆")
    print(f"  總計: {len(df)} 筆\n")

    # 決定用哪一欄 (filename_lr 或 filename_hr)
    filename_col = 'filename_hr' if USE_HIGH_RES else 'filename_lr'
    suffix = '_hr' if USE_HIGH_RES else '_lr'
    print(f"使用 {'records500' if USE_HIGH_RES else 'records100'} (欄位: {filename_col})\n")

    # 依類別處理
    total_ok = 0
    total_fail = 0
    total_skip = 0
    fail_log = OUTPUT_ROOT / 'failures.log'

    for cid in sorted(CLASS_NAMES.keys()):
        class_df = df[df['class_id'] == cid]
        if LIMIT_PER_CLASS:
            class_df = class_df.head(LIMIT_PER_CLASS)

        print(f"\n=== 處理 Class {cid} ({CLASS_NAMES[cid]}): {len(class_df)} 筆 ===")

        if len(class_df) == 0:
            print(f"  ⚠️  Class {cid} 沒有資料（預期行為，例如 VFL）")
            continue

        for idx, row in enumerate(class_df.itertuples(), 1):
            # filename_lr 例如: 'records100/00000/00001_lr'
            rel_path = getattr(row, filename_col)
            dat_file = PTB_XL_BASE / (rel_path + '.dat')
            hea_file = PTB_XL_BASE / (rel_path + '.hea')

            if not dat_file.exists() or not hea_file.exists():
                # 檔案還沒下載完會落到這裡
                total_skip += 1
                if idx <= 3:   # 只印前 3 筆警告避免洗版
                    print(f"  [{idx}] 檔案未下載: {dat_file.name}")
                continue

            # 已生成過就跳過
            record_name = dat_file.stem  # 例如 '00001_lr'
            expected_png = class_dirs[cid] / f'{record_name}-0.png'
            if expected_png.exists():
                total_skip += 1
                continue

            # 呼叫生成器（--fully_random 一次啟用所有隨機增強：
            # 旋轉、裁切、雜訊、色溫、皺摺、手寫註記、網格顏色、DPI 變異…
            # 模擬手機拍紙本 ECG 的多樣性）
            cmd = [
                sys.executable, str(GENERATOR),
                '-i', str(dat_file),
                '-hea', str(hea_file),
                '-o', str(class_dirs[cid]),
                '-st', '0',
                '--store_config', '2',
                '-r', '200',
                '--print_header',
                '--fully_random',
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    timeout=TIMEOUT_PER_FILE,
                )
                total_ok += 1
                if idx % 50 == 0 or idx <= 5:
                    print(f"  [{idx}/{len(class_df)}] ✓ {record_name}")
            except subprocess.CalledProcessError as e:
                total_fail += 1
                err_tail = e.stderr.decode('utf-8', errors='ignore')[-300:]
                with open(fail_log, 'a') as f:
                    f.write(f"[Class {cid}] {record_name}\n{err_tail}\n\n")
                if total_fail <= 5:
                    print(f"  [{idx}] ✗ {record_name}: {err_tail[:150]}")
            except subprocess.TimeoutExpired:
                total_fail += 1
                print(f"  [{idx}] ✗ 逾時: {record_name}")

    # 最後總結
    print(f"\n{'='*50}")
    print(f"完成！成功 {total_ok} / 失敗 {total_fail} / 跳過 {total_skip}")
    print(f"輸出目錄: {OUTPUT_ROOT}")
    if total_fail > 0:
        print(f"失敗紀錄: {fail_log}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
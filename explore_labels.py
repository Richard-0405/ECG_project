import pandas as pd
import ast
from collections import Counter

# === 路徑 ===
BASE = '/data/ecg_project/physionet.org/files/ptb-xl/1.0.3'
df = pd.read_csv(f'{BASE}/ptbxl_database.csv')
scp = pd.read_csv(f'{BASE}/scp_statements.csv', index_col=0)

# === 1. 基本資訊 ===
print("=" * 60)
print("PTB-XL 資料集概觀")
print("=" * 60)
print(f"總筆數: {len(df)}")
print(f"欄位: {df.columns.tolist()}")
print()

# === 2. scp_codes 欄位（這是真正的診斷標籤） ===
# 這個欄位是字串格式的 dict，要轉成真的 dict
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

# 前 5 筆長怎樣
print("前 5 筆 scp_codes：")
for i in range(5):
    print(f"  ecg_id={df.iloc[i]['ecg_id']}: {df.iloc[i]['scp_codes']}")
print()

# === 3. 統計所有出現過的診斷碼 ===
all_codes = Counter()
for codes_dict in df['scp_codes']:
    for code in codes_dict.keys():
        all_codes[code] += 1

print(f"共有 {len(all_codes)} 種不同的診斷碼")
print(f"\n所有診斷碼出現次數（由多到少）：")
for code, count in all_codes.most_common():
    # 從 scp_statements.csv 查這個代碼的意思
    if code in scp.index:
        description = scp.loc[code, 'description']
        diagnostic_class = scp.loc[code, 'diagnostic_class']
    else:
        description = '(no description)'
        diagnostic_class = '?'
    print(f"  {code:10s} {count:6d} 筆  [{diagnostic_class}]  {description}")
print()

# === 4. 比賽用的 5 大 diagnostic_class ===
print("=" * 60)
print("scp_statements.csv 的 diagnostic_class 分組")
print("=" * 60)
# diagnostic_class 是 PTB-XL 官方整理的 5 大類
print(scp['diagnostic_class'].value_counts(dropna=False))
print()

# === 5. 每筆 ECG 至少屬於哪些大類 ===
def get_diagnostic_classes(codes_dict):
    """從 scp_codes 轉出 diagnostic_class 的集合"""
    classes = set()
    for code in codes_dict.keys():
        if code in scp.index:
            dc = scp.loc[code, 'diagnostic_class']
            if pd.notna(dc):
                classes.add(dc)
    return classes

df['diagnostic_classes'] = df['scp_codes'].apply(get_diagnostic_classes)

# 統計每個大類各有多少筆
class_counter = Counter()
for classes in df['diagnostic_classes']:
    for c in classes:
        class_counter[c] += 1

print("每個 diagnostic_class 對應的 ECG 筆數：")
for c, n in class_counter.most_common():
    print(f"  {c}: {n}")
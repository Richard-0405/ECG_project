import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fractions import Fraction
from scipy import stats, signal

# utils
from utils.mapping import mapping
from utils.load_file import load_file
import utils.HRService_code_360 as HRService_code_360
from utils.inference import inference
from utils.save_rhythm import save_rhythm
from utils.dwt import dwt
from utils.inference_torch import inference_torch

mypath = sys.argv[1]
print("=========")
print("sys.argv[0]:main.py", sys.argv[0])
print("sys.argv[1]:directory", sys.argv[1])
print("sys.argv[2]:fileName", sys.argv[2])
print("=========")

# === 採樣頻率與視窗設定 ===
Fs = 360                  # 📌 這裡填入你「實際輸入訊號」的採樣頻率 (例如 250, 360, 500)
window_seconds = 5        # 模型每次判讀的秒數 (預設5秒)
window_size = int(Fs * window_seconds) # 自動計算視窗長度
Fs_resample = 180         # 模型「推論時需要」的目標採樣頻率

fraction = Fraction(Fs_resample/Fs).limit_denominator()
p = fraction.numerator
q = fraction.denominator
window_step = 5           # 視窗移動步長 (秒)

print("List mypath: ", mypath)

# 1. 自動取得資料夾內最新上傳的 .csv 檔案
csv_files = glob.glob(os.path.join(mypath, '*.csv'))

if not csv_files:
    print(f"錯誤：在 {mypath} 中找不到任何 .csv 檔案！")
    sys.exit() # 找不到檔案就終止程式

latest_csv = max(csv_files, key=os.path.getmtime)
print(f"最新上傳的檔案是: {latest_csv}")

# 2. 讀取最新檔案的訊號
print("Load_file:", latest_csv)
sig = load_file(latest_csv)
print('sig_length:' + str(len(sig)))

# 3. 切割視窗並推論
# 計算總共可以切出多少個 window
window_num = int(np.floor((len(sig) - window_size) / (Fs * window_step)))
if window_num <= 0:
    window_num = 1 # 確保至少執行一次推論

pred = np.zeros((window_num, 1))
start_point_list = []
end_point_list = []

for i in range(window_num):
    start_point = window_size * i
    end_point = min(start_point + window_size, len(sig))
    start_point_list.append(start_point)
    end_point_list.append(end_point)

    # 擷取該視窗的訊號並做 Z-score 正規化
    sig_slice = sig[start_point:end_point]
    
    # 防呆：如果訊號極短，補零至 window_size 避免後續報錯
    if len(sig_slice) < window_size:
        sig_slice = np.pad(sig_slice, (0, window_size - len(sig_slice)), 'constant')
        
    sig_zscore = stats.zscore(sig_slice)
    
    # 核心前處理：重採樣 (降頻/升頻至模型要的 180Hz)
    sig_res = signal.resample_poly(sig_zscore, p, q)

    # 模型推論
    sig_dwt = dwt(sig_res, 'db1', 1)
    pred_temp = inference_torch(sig_dwt)
    pred[i] = pred_temp[0]

# 儲存預測結果的 rhythm 特徵
save_rhythm(start_point_list, end_point_list, np.transpose(pred), mypath)

# 定義疾病標籤
label_mapping = {
    0: "Normal",
    1: "AF (心房顫動)",
    2: "VFL (心室撲動/顫動)",
    3: "SVTA (上心室頻脈)",
    4: "Others (其他異常)"
}

# 4. 統計 label 結果
labels = pred.flatten().astype(int)
unique, counts = np.unique(labels, return_counts=True)
label_count_dict = dict(zip(unique, counts))

print("\n=== Label 統計結果 ===")
total_windows = len(labels)
for lab in range(5):
    cnt = label_count_dict.get(lab, 0)
    print(f"{label_mapping[lab]} (label={lab}) : {cnt} ({cnt / total_windows * 100:.3f}%)")

# 5. 繪製並儲存包含百分比的圓餅圖
pie_labels = []
pie_sizes = []

for lab in range(5):
    cnt = label_count_dict.get(lab, 0)
    if cnt > 0:  # 只將有偵測到的疾病加入圓餅圖，讓圖表更乾淨
        pie_labels.append(label_mapping[lab])
        pie_sizes.append(cnt)

plt.figure(figsize=(7, 6))
# 繪製圓餅圖，autopct='%1.1f%%' 代表顯示到小數點後第一位的百分比
plt.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
plt.title('ECG Rhythm Distribution (Abnormality %)')
plt.tight_layout()

save_path = os.path.join(mypath, "ECG_piechart.png")
plt.savefig(save_path, dpi=2
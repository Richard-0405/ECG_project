# 心電圖智慧助理（ECG Intelligence Assistant）

一個以 Streamlit 打造的 **ECG 影像 → 數位訊號 → 心律不整分類 → 對話式衛教** 網頁應用。
使用者上傳一張 12 導程心電圖影像，系統會自動完成以下流程，並由 AI 助理提供健康建議。

```
ECG 影像 ──► 影像前處理 ──► 數位化（hengck23 三階段模型）──► 心律分類（boki）──► 互動圖表 + GPT 對話
```

---

## 功能

- **影像前處理**：灰階化 + CLAHE 對比度增強，去除陰影與雜訊
- **ECG 數位化**：使用 [hengck23](https://github.com/hengck23) 的三階段模型，將影像還原為 12 導程時域訊號 CSV
- **心律分類**：boki 模型對 Lead II rhythm strip 做 5 秒滑動視窗推論，輸出五類
  - Normal（正常）
  - AF（心房顫動）
  - VFL（心室撲動／顫動）
  - SVTA（上心室頻脈）
  - Others（其他異常）
- **互動式圓餅圖**：以 Plotly 呈現各類別視窗分佈
- **AI 對話助理（GPT）**：會依使用者的姓名、年齡、病史、位置提供個人化衛教建議，並具備以下工具：
  - Google Maps → 搜尋附近診所
  - Spoonacular → 健康食譜查詢
  - LINE Messaging API → 將今日健康報告推播給家屬
- **裝置自動偵測**：啟動時顯示推論裝置（CPU / CUDA）

---

## 專案結構

```
ecg_project/
├── app.py                         # Streamlit 主程式
├── ecg_digitize.py                # 影像 → 12 導程 CSV 的 CLI 腳本
├── visualize_ecg.py               # ECG 訊號視覺化工具
├── requirements.txt               # Python 相依套件
├── .streamlit/
│   └── secrets.toml.example       # API 金鑰範本（secrets.toml 被 .gitignore）
├── boki/                          # 心律分類模型
│   ├── mymain.py
│   ├── utils/                     # 前處理、推論、讀檔等工具
│   └── weight/                    # 量化模型權重（.pt）
├── hengck23-submit-physionet/     # 影像數位化模型（PhysioNet Challenge）
│   ├── stage0_*.py                # 方向／透視校正
│   ├── stage1_*.py                # 格點偵測與重整
│   ├── stage2_*.py                # 訊號萃取
│   └── weight/                    # 三階段模型權重（.pth）
└── ecg_digitizer/                 # （選用）PhysioNet 2024 範例，另外 clone
```

> `ecg_digitizer/` 是 [physionetchallenges/python-example-2024](https://github.com/physionetchallenges/python-example-2024) 的外部 repo，本專案並未將它納入版本控管。如需使用，請在專案根目錄另外執行：
> ```bash
> git clone https://github.com/physionetchallenges/python-example-2024.git ecg_digitizer
> ```

---

## 環境需求

- **Python 3.12**（不建議使用 3.14，PyTorch 目前尚未支援）
- Windows / macOS / Linux
- 建議 8 GB 以上 RAM
- 若要使用 GPU 推論，需 NVIDIA 顯卡與對應版本的 CUDA（見下方 GPU 版 PyTorch 安裝說明）

---

## 安裝步驟

### 1. 建立虛擬環境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. 安裝相依套件

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU 版 PyTorch（選用）**
> `requirements.txt` 預設安裝的是 CPU 版。若要使用 CUDA，請先解除安裝再依官方指令重裝：
> ```bash
> pip uninstall -y torch torchvision
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```
> 可依你的 CUDA 版本調整 `cu121` / `cu124` 等字尾。

### 3. 設定 API 金鑰

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

接著編輯 `.streamlit/secrets.toml`，填入你自己的金鑰：

| 金鑰 | 申請位置 | 用途 |
|---|---|---|
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys | 對話助理 |
| `GOOGLE_MAPS_API_KEY` | https://console.cloud.google.com/apis/credentials | 診所搜尋（需啟用 Places API） |
| `SPOONACULAR_API_KEY` | https://spoonacular.com/food-api | 健康食譜 |
| `LINE_CHANNEL_ACCESS_TOKEN` | https://developers.line.biz/console/ | LINE 推播 |
| `LINE_TARGET_USER_ID` | LINE Developers → Messaging API | 收件者 User ID |

> `secrets.toml` 已在 `.gitignore` 中，不會被 commit。

### 4. 準備模型權重

- `boki/weight/` 已包含 `Quantized_v4_9876.pt` 等心律分類模型權重
- `hengck23-submit-physionet/weight/` 需要三個 `.pth` 權重檔（stage0/stage1/stage2）。如果你的 repo 沒附帶這些權重，請向專案提供者索取

---

## 執行

```bash
streamlit run app.py
```

啟動後瀏覽器會開啟 `http://localhost:8501`。

### 使用流程

1. 左側側邊欄 **影像分析** 區塊上傳 12 導程 ECG 影像（PNG / JPG）
2. 按下 **開始分析**，系統會依序完成：前處理 → 訊號數位化 → 節律分類
3. 側邊欄下方顯示節律分佈圓餅圖（滑鼠懸停可看各類別視窗數）
4. 右側主畫面的 AI 助理會自動針對分析結果提供個人化衛教建議
5. 在 **個人資料** 區塊填寫姓名、年齡、性別、病史、位置，讓 AI 給更精確的建議
6. 需要通報家屬時按 **傳送今日健康報告**，會透過 LINE 推播給設定的家屬

---

## 技術細節

### ECG 影像數位化（hengck23 三階段）

1. **Stage 0** — 方向／透視校正：輸出旋轉後的標準化影像
2. **Stage 1** — 格點偵測：找出 ECG 網格並 rectify
3. **Stage 2** — 訊號萃取：從 rectified 影像還原出 4 組 series（12 導程 + Lead II rhythm strip）

最終 CSV 欄位：`Time, I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`，取樣率預設 **500 Hz**。

### 心律分類（boki）

- 輸入：Lead II rhythm strip（取樣率 500 Hz）
- 處理：5 秒非重疊視窗 → z-score → `resample_poly` 到 180 Hz → Daubechies-1 DWT
- 模型：量化 1D-CNN（4 層 conv + 1 層 FC），使用 `qnnpack` 後端
- 輸出：每個視窗一個類別標籤（0–4），統計主要節律與分佈

---

## 常見問題

**Q：按下「開始分析」後出現 `ModuleNotFoundError: No module named 'timm'`？**
A：確認 `requirements.txt` 裡的套件都已安裝（`pip install -r requirements.txt`），並確保 Streamlit 是從 **venv 的 Python** 啟動。

**Q：啟動後圖表中文變亂碼？**
A：請確認系統有 Noto Sans TC / 微軟正黑體等中文字型；本專案已用 Google Fonts 引入 Noto Sans TC / Noto Serif TC。

**Q：是否一定要 GPU？**
A：不用。CPU 也能跑，只是單張影像處理約需 20–40 秒。GPU 可縮短至 5 秒內。

---

## 授權

專案內 `hengck23-submit-physionet/` 與 `boki/` 等第三方模型請依其各自授權使用。

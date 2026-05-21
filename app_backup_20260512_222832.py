import streamlit as st
import requests # 🌟 新增：用來呼叫外部 API 的套件
import datetime
import json
import googlemaps  # 🌟 新增：匯入 Google Maps 套件
import pandas as pd
from PIL import Image
import os
from openai import OpenAI
import cv2        # 🌟 新增：用於影像前處理 (請確認已 pip install opencv-python)
import numpy as np # 🌟 新增：用於影像陣列處理
import subprocess # 🌟 新增：用於呼叫命令列執行 Kaggle 模型
import sys
import torch
from scipy import stats, signal as scipy_signal
from fractions import Fraction

# 將 boki 加入 import 路徑，讓我們可以直接呼叫它的推論 pipeline
BOKI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boki")
if BOKI_DIR not in sys.path:
    sys.path.insert(0, BOKI_DIR)
from utils.dwt import dwt as boki_dwt
from utils.inference_torch import inference_torch as boki_inference

BOKI_LABELS = {
    0: "Normal (正常)",
    1: "AF (心房顫動)",
    2: "VFL (心室撲動/顫動)",
    3: "SVTA (上心室頻脈)",
    4: "Others (其他異常)",
}

DEVICE_STR = "CUDA ({})".format(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU"


def classify_with_boki(csv_path, fs=200, window_seconds=5, fs_target=180):
    """讀取數位化後的 ECG CSV，用 boki 推論每 5 秒視窗的心律類別。

    CSV 欄位：Time, I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    → 使用 Lead II（完整 rhythm strip）作為輸入。

    fs=200：ecg-image-kit 用 -r 200 渲染，hengck23 stage2 crop (118..2080 px)
    對應 10 秒 rhythm strip。先前用 fs=500 會讓 boki 以為 HR 300-400 bpm，
    把所有訊號都判成心房顫動/心室撲動。

    回傳：dominant_label(int), counts(dict[label]=n), total_windows(int)
    """
    df = pd.read_csv(csv_path)
    sig = df["II"].to_numpy(dtype=float)

    window_size = int(fs * window_seconds)
    frac = Fraction(fs_target / fs).limit_denominator()
    p, q = frac.numerator, frac.denominator

    step = window_size  # 不重疊
    n_windows = max(1, (len(sig) - window_size) // step + 1)
    preds = []
    for i in range(n_windows):
        s = sig[i * step : i * step + window_size]
        if len(s) < window_size:
            s = np.pad(s, (0, window_size - len(s)), "constant")
        s = stats.zscore(s)
        s = np.nan_to_num(s)
        s = scipy_signal.resample_poly(s, p, q)
        s = boki_dwt(s, "db1", 1)
        pred = boki_inference(s)
        preds.append(int(pred[0]))

    counts = {lab: preds.count(lab) for lab in range(5)}
    dominant = max(counts, key=counts.get)
    return dominant, counts, len(preds)

# streamlit run app.py
# 自動從 secrets.toml 讀取 API Key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
gmaps = googlemaps.Client(key=st.secrets["GOOGLE_MAPS_API_KEY"]) # 🌟 新增：初始化 Google Maps 客戶端

st.set_page_config(page_title="心電圖智慧助理", page_icon="◐", layout="wide")


# ==========================================
# 視覺樣式（注入 CSS）
# ==========================================
def inject_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+TC:wght@400;500;700&family=Noto+Serif+TC:wght@500;700&family=JetBrains+Mono:wght@500&display=swap');
        :root {
            --accent: #14b8a6;
            --accent-soft: rgba(20,184,166,0.12);
            --border: rgba(148,163,184,0.18);
            --muted: #94a3b8;
            --text: #e2e8f0;
        }
        html, body, .stApp, .stMarkdown, .stText,
        button, input, textarea, select,
        div[data-testid="stSidebar"],
        div[data-testid="stChatMessage"],
        div[data-testid="stChatInput"] {
            font-family: "Inter", "Noto Sans TC", "PingFang TC",
                         "Microsoft JhengHei", system-ui, sans-serif;
            letter-spacing: 0.005em;
            font-feature-settings: "ss01", "cv11", "tnum";
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        /* Display / headings — serif for editorial feel */
        .page-hero h1, h1, h2, h3 {
            font-family: "Noto Serif TC", "Source Han Serif TC",
                         "Inter", serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.015em;
        }
        /* Monospace / code */
        code, pre, .device-badge, .eyebrow, .page-hero .sub {
            font-family: "JetBrains Mono", "Inter", monospace !important;
        }
        /* 保留 Streamlit 的 Material Symbols 圖示字型 */
        [class*="material-symbols"], [class*="material-icons"],
        span[data-testid*="icon"] {
            font-family: "Material Symbols Rounded",
                         "Material Symbols Outlined",
                         "Material Icons" !important;
        }
        .block-container { padding-top: 2.2rem; max-width: 1180px; }

        /* Eyebrow / section label */
        .eyebrow {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--muted);
            margin: 0 0 0.35rem 0;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin: 0 0 0.9rem 0;
            color: var(--text);
        }

        /* Page header */
        .page-hero {
            display: flex; align-items: baseline; gap: 0.9rem;
            padding: 0 0 0.4rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1.6rem;
        }
        .page-hero h1 {
            font-size: 1.85rem; font-weight: 700; margin: 0;
            letter-spacing: -0.01em;
        }
        .page-hero .sub {
            font-size: 0.85rem; color: var(--muted);
            letter-spacing: 0.14em; text-transform: uppercase;
        }

        /* Sidebar headings — replace default h2 look */
        section[data-testid="stSidebar"] h2 {
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.22em !important;
            text-transform: uppercase !important;
            color: var(--muted) !important;
            margin-top: 1.6rem !important;
            margin-bottom: 0.6rem !important;
            padding: 0 !important;
        }
        section[data-testid="stSidebar"] h1 {
            font-size: 1.25rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.005em !important;
        }

        /* Buttons: primary = teal accent */
        .stButton > button {
            border-radius: 8px;
            border: 1px solid var(--border);
            font-weight: 500;
            transition: all 0.15s ease;
            padding: 0.55rem 1rem;
        }
        .stButton > button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }
        .stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
            color: #0f172a;
        }
        .stButton > button[kind="primary"]:hover {
            background: #0d9488;
            border-color: #0d9488;
            color: #0f172a;
        }

        /* Device badge */
        .device-badge {
            display: inline-flex; align-items: center; gap: 0.45rem;
            padding: 0.25rem 0.65rem;
            border: 1px solid var(--border);
            border-radius: 999px;
            font-size: 0.72rem; color: var(--muted);
            letter-spacing: 0.08em; text-transform: uppercase;
        }
        .device-badge::before {
            content: ""; width: 6px; height: 6px; border-radius: 50%;
            background: var(--accent); box-shadow: 0 0 8px var(--accent);
        }

        /* Status bar (subtle inline) */
        .status-line {
            font-size: 0.82rem; color: var(--muted);
            padding: 0.55rem 0.8rem;
            border-left: 2px solid var(--accent);
            background: var(--accent-soft);
            border-radius: 2px;
            margin: 0.4rem 0 0.8rem 0;
        }

        /* Cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 10px;
        }

        /* Image captions tighter */
        div[data-testid="stImage"] figcaption {
            font-size: 0.78rem; color: var(--muted);
            text-transform: uppercase; letter-spacing: 0.14em;
            margin-top: 0.3rem;
        }

        /* Chat input polish — 寬度對齊內容區 */
        div[data-testid="stChatInput"] { border-radius: 10px; }
        div[data-testid="stBottom"] > div,
        div[data-testid="stBottomBlockContainer"] {
            max-width: 1180px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        /* Divider subtler */
        hr { border-color: var(--border) !important; opacity: 0.6; }

        /* ---------- Animations ---------- */
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; } to { opacity: 1; }
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(20,184,166,0.55); }
            50%      { box-shadow: 0 0 0 6px rgba(20,184,166,0); }
        }
        @keyframes shimmer {
            0%   { background-position: -200px 0; }
            100% { background-position: 200px 0; }
        }
        .page-hero,
        .status-line,
        div[data-testid="stImage"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stChatMessage"] {
            animation: fadeUp 0.45s ease both;
        }
        .eyebrow, .section-title {
            animation: fadeIn 0.5s ease both;
        }
        .device-badge::before {
            animation: pulse 2s ease-in-out infinite;
        }
        /* Buttons press feedback */
        .stButton > button:active { transform: translateY(1px); }

        /* Spinner box polish */
        div[data-testid="stSpinner"] {
            background: linear-gradient(
                90deg,
                rgba(20,184,166,0.06) 0%,
                rgba(20,184,166,0.15) 50%,
                rgba(20,184,166,0.06) 100%);
            background-size: 200px 100%;
            animation: shimmer 1.6s linear infinite;
            border-radius: 6px;
            padding: 0.2rem 0.4rem !important;
        }

        /* ---------- RWD ---------- */
        @media (max-width: 768px) {
            .block-container {
                padding: 1.2rem 0.8rem !important;
                max-width: 100% !important;
            }
            .page-hero {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.25rem;
                padding-bottom: 0.8rem;
                margin-bottom: 1.1rem;
            }
            .page-hero h1 { font-size: 1.45rem; }
            .page-hero .sub { font-size: 0.7rem; letter-spacing: 0.1em; }
            section[data-testid="stSidebar"] { min-width: 85vw !important; }
            div[data-testid="stChatMessage"] { padding: 0.5rem 0.6rem; }
            .stButton > button { padding: 0.7rem 1rem; font-size: 0.95rem; }
        }
        @media (max-width: 480px) {
            .page-hero h1 { font-size: 1.25rem; }
            .eyebrow { font-size: 0.66rem; letter-spacing: 0.18em; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()

# ==========================================
# 🌟 新增：影像前處理功能 (Perspective & Illumination Correction)
# ==========================================
def preprocess_ecg_image(pil_image):
    """將上傳的 ECG 影像進行灰階與去陰影前處理"""
    # 1. 將 PIL Image 轉換為 OpenCV 格式 (NumPy array)
    open_cv_image = np.array(pil_image)
    
    # 確保轉換為 RGB 格式 (去除透明通道)
    if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 4:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2RGB)
    elif len(open_cv_image.shape) == 2:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2RGB)
        
    # 2. 轉為單通道灰階圖
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    
    # 3. 光線均勻化 (去陰影)：使用 CLAHE (限制對比度自適應直方圖均衡化)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(gray)
    
    # 如果未來需要自動透視校正，可以在這階段加入 OpenCV 的 findContours 與 getPerspectiveTransform
    
    return enhanced_img


def save_uploaded_csv(uploaded_file):
    file_stem = os.path.splitext(uploaded_file.name)[0]
    csv_path = f"{file_stem}_uploaded.csv"
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return csv_path

# ==========================================
# 外部 API 工具 (MCP Tool)
# ==========================================
def search_nearby_restaurants(location, keyword="健康"):
    """使用 Google Maps API 真實搜尋附近餐廳"""
    try:
        # 將使用者的位置與關鍵字組合成搜尋字串，預設尋找健康相關餐廳
        search_query = f"{location} {keyword} 餐廳"
        places_result = gmaps.places(query=search_query, language='zh-TW')
        
        if places_result['status'] == 'OK':
            # 只取前 3 筆結果避免資訊過載
            results = places_result['results'][:3]
            restaurants_info = []
            
            for place in results:
                name = place.get('name')
                address = place.get('formatted_address')
                rating = place.get('rating', '暫無評分') 
                
                restaurants_info.append(f"- **{name}** (Google 評分: {rating})\n  地址: {address}")
            
            formatted_response = f"在「{location}」找到以下 {keyword} 相關餐廳：\n" + "\n".join(restaurants_info)
            return formatted_response
        else:
            return f"抱歉，在「{location}」找不到相關的餐廳。狀態碼：{places_result['status']}"
            
    except Exception as e:
        return f"搜尋餐廳時發生錯誤: {str(e)}"
def search_nearby_clinics(location, specialty="心臟內科"):
    """使用 Google Maps API 真實搜尋附近診所"""
    try:
        # 將使用者的位置與科別組合成搜尋關鍵字
        search_query = f"{location} {specialty} 診所 醫院"
        places_result = gmaps.places(query=search_query, language='zh-TW')
        # 呼叫 Google Maps Places API 進行文字搜尋
        
        if places_result['status'] == 'OK':
            # 只取前 3 筆結果避免資訊過載
            results = places_result['results'][:3]
            clinics_info = []
            
            for place in results:
                name = place.get('name')
                address = place.get('formatted_address')
                # 取得評分，若無則顯示暫無評分
                rating = place.get('rating', '暫無評分') 
                
                clinics_info.append(f"- **{name}** (Google 評分: {rating})\n  地址: {address}")
            
            # 將結果組合回傳給 LLM
            formatted_response = f"在「{location}」找到以下 {specialty} 相關院所：\n" + "\n".join(clinics_info)
            return formatted_response
        else:
            return f"抱歉，在「{location}」找不到相關的 {specialty} 院所。狀態碼：{places_result['status']}"
            
    except Exception as e:
        return f"搜尋時發生錯誤: {str(e)}"
    
def search_healthy_recipe(keyword):
    """使用 Spoonacular API 真實查詢健康食譜"""
    try:
        # 準備 API 金鑰與請求網址
        api_key = st.secrets["SPOONACULAR_API_KEY"]
        url = "https://api.spoonacular.com/recipes/complexSearch"
        
        # 設定查詢參數 (我們加上了 diet=primal 或限制鈉含量等健康過濾條件)
        params = {
            "query": keyword,
            "apiKey": api_key,
            "number": 3,           # 只回傳 3 筆避免資訊過多
            "diet": "pescetarian", # 舉例：海鮮素/健康飲食 (可依需求調整)
            "addRecipeInformation": True # 直接拿取食譜摘要
        }
        
        # 發送 GET 請求
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code == 200 and data.get("results"):
            results = data["results"]
            recipe_info = []
            
            for recipe in results:
                title = recipe.get("title")
                ready_in_minutes = recipe.get("readyInMinutes", "未知")
                source_url = recipe.get("sourceUrl", "")
                
                recipe_info.append(f"- **{title}** (預計花費時間: {ready_in_minutes} 分鐘)\n  查看完整食譜: {source_url}")
            
            formatted_response = f"為您找到包含「{keyword}」的健康食譜：\n" + "\n".join(recipe_info)
            return formatted_response
        else:
            return f"抱歉，目前在食譜庫中找不到包含「{keyword}」的合適健康食譜。"
            
    except Exception as e:
        print("【食譜 API 執行錯誤】:", str(e))
        return f"查詢食譜時發生錯誤: {str(e)}"

# ==========================================
# LINE 推播通知功能
# ==========================================
def send_line_report(report_text):
    """透過 LINE API 發送文字報告給指定家屬"""
    try:
        url = "https://api.line.me/v2/bot/message/push"
        headers = {
            "Authorization": f"Bearer {st.secrets['LINE_CHANNEL_ACCESS_TOKEN']}",
            "Content-Type": "application/json"
        }
        data = {
            "to": st.secrets["LINE_TARGET_USER_ID"],
            "messages": [
                {
                    "type": "text",
                    "text": report_text
                }
            ]
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return True
        else:
            print(f"【LINE API 錯誤】: {response.text}")
            return False
            
    except Exception as e:
        print(f"【LINE 發送例外錯誤】: {str(e)}")
        return False

# ==========================================
# 運動菜單生成功能
# ==========================================
def generate_workout_plan(goal, duration_minutes, intensity, medical_history, weight):
    """根據目標、時間、病史與體重，生成超詳細的客製化運動菜單"""
    try:
        # 1. 醫療安全評估機制
        has_heart_issue = any(k in medical_history for k in ["心", "高血壓", "中風"])
        
        # 計算各階段時間分配
        warmup_time = max(5, int(duration_minutes * 0.15))
        cooldown_time = max(5, int(duration_minutes * 0.15))
        main_time = duration_minutes - warmup_time - cooldown_time

        if has_heart_issue:
            intensity = "低" 
            safe_warning = "⚠️ **【醫療安全機制啟動】**：偵測到您的心血管病史，已自動排除需憋氣發力的大重量訓練，改以「低強度、高次數、維持呼吸」的機械式或輕量訓練為主。"
            safe_weight = max(2, int(weight * 0.03)) 
            exercises = [
                f"坐姿機械推胸：重量設定約 {safe_weight * 2} kg，3組 x 12-15下 (推起時吐氣，絕對不可憋氣)",
                f"坐姿滑輪下拉：重量設定約 {safe_weight * 2} kg，3組 x 15下",
                f"輕啞鈴深蹲：雙手各持 {safe_weight} kg，3組 x 10下 (下蹲吸氣，起立吐氣)",
                f"坐姿腿伸伸：重量設定約 {safe_weight * 2.5} kg，3組 x 12下"
            ]
        else:
            safe_warning = "💡 **【教練提示】**：以下重量為依據您體重估算的起始建議，請根據實際體感隨時微調，『動作標準』永遠比『重量』更重要！"
            
            if goal == "增肌" or goal == "肌力訓練":
                db_weight = max(5, int(weight * 0.1)) 
                exercises = [
                    f"啞鈴臥推：單手 {db_weight} kg，4組 x 8-10下",
                    f"啞鈴單臂划船：單手 {db_weight} kg，4組 x 10-12下",
                    f"高腳杯深蹲：手持 {db_weight * 1.5} kg，4組 x 8-10下",
                    f"機械式腿推舉：重量設定約 {weight * 0.8} kg，4組 x 10下"
                ]
            elif goal == "減脂":
                db_weight = max(3, int(weight * 0.05))
                exercises = [
                    f"啞鈴肩推：單手 {db_weight} kg，3組 x 15下",
                    f"戰繩 / 快速波比跳：30秒 x 4組 (組間休息 30 秒)",
                    f"啞鈴弓箭步：雙手各持 {db_weight} kg，3組 x 12下 (單腳)",
                    f"深蹲跳：3組 x 15下"
                ]
            else:
                db_weight = max(3, int(weight * 0.06))
                exercises = [
                    f"機械式胸推：重量設定約 {db_weight * 3} kg，3組 x 12下",
                    f"機械式背肌划船：重量設定約 {db_weight * 3} kg，3組 x 12下",
                    f"農夫走路：雙手各持 {db_weight * 1.5} kg，走 30 秒 x 3組",
                    f"棒式核心支撐：撐 45 秒 x 3組"
                ]

        plan = f"為您客製化的 **{duration_minutes} 分鐘【{goal}】菜單** (強度設定：{intensity})\n\n"
        plan += safe_warning + "\n\n"
        plan += f"📍 **【暖身】 ({warmup_time} 分鐘)**\n"
        plan += "- 原地快走/慢跑、動態伸展 (肩頸繞環、擴胸、深蹲伸展)\n\n"
        plan += f"📍 **【主運動】 ({main_time} 分鐘)**\n"
        for ex in exercises:
            plan += f"- {ex}\n"
        plan += f"\n📍 **【緩和與伸展】 ({cooldown_time} 分鐘)**\n"
        plan += "- 靜態拉筋 (胸大肌拉伸、大腿前側拉伸、嬰兒式放鬆)\n"
        
        return plan
    except Exception as e:
        return f"【系統除錯訊息】產生菜單時發生錯誤：{str(e)}。請直接將此錯誤訊息告訴使用者。"

# 定義要告訴 GPT 我們有哪些工具可以使用
tools_definition = [
    # 找餐廳 (新增)
    {
        "type": "function",
        "function": {
            "name": "search_nearby_restaurants",
            "description": "根據使用者的位置與需求，尋找附近的餐廳、健康餐盒或特定美食。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市或地區，例如：台南市、台北市"},
                    "keyword": {"type": "string", "description": "餐廳類型或關鍵字，例如：健康餐盒、素食、低卡、火鍋"}
                },
                "required": ["location"]
            }
        }
    },
    # 👇 新增第四個工具：生成運動菜單
    {
        "type": "function",
        "function": {
            "name": "generate_workout_plan",
            "description": "當使用者要求安排運動、健身、減脂或增肌菜單時呼叫此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "運動目標，如：減脂、增肌、維持健康"},
                    "duration_minutes": {"type": "integer", "description": "運動總時長(分鐘)"},
                    "intensity": {"type": "string", "description": "強度，如：高、中、低"}
                },
                "required": ["goal", "duration_minutes"]
            }
        }
    },
      # 找醫院
    {
        "type": "function",
        "function": {
            "name": "search_nearby_clinics",
            "description": "根據使用者的位置與需求，尋找附近的醫療院所或診所。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市或地區，例如：台南市、台北市"},
                    "specialty": {"type": "string", "description": "科別，例如：心臟內科、家醫科"}
                },
                "required": ["location"]
            }
        }
    },
      # 找食譜
    {
        "type": "function",
        "function": {
            "name": "search_healthy_recipe",
            "description": "根據使用者提供的食材關鍵字，查詢適合心血管保養的健康食譜。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "食材關鍵字，例如：鮭魚、燕麥、蔬菜"}
                },
                "required": ["keyword"]
            }
        }
    },
    # 👇 這是新增的第三個工具：發送 LINE 報告
    {
        "type": "function",
        "function": {
            "name": "send_line_report",
            "description": "當使用者要求通報家屬、傳送健康報告或總結今日狀況時使用。將使用者的健康報告或對話重點摘要，透過 LINE 傳送給家屬。",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_text": {
                        "type": "string", 
                        "description": "要傳送給家屬的報告內容。請你根據使用者的病史、今日的對話內容，整理出一份專業、溫暖且包含重點的摘要文字。"
                    }
                },
                "required": ["report_text"]
            }
        }
    }
]

# ==========================================
# 側邊欄 (Sidebar) 區塊
# ==========================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; gap:0.6rem; padding-bottom:0.4rem;">
          <div style="font-size:0.72rem; letter-spacing:0.22em; text-transform:uppercase;
                      color:var(--muted); font-weight:600;">Console</div>
          <div style="font-size:1.25rem; font-weight:700; letter-spacing:-0.005em;">工作台</div>
          <div class="device-badge">{DEVICE_STR}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    st.header("ECG 數位化")
    uploaded_file = st.file_uploader(
        "上傳 12 導程 ECG 影像",
        type=["png", "jpg", "jpeg"],
        label_visibility="visible",
    )

    digitize_clicked = st.button("執行 ECG 數位化", use_container_width=True, type="primary")

    st.divider()

    st.header("心律不整辨識")
    uploaded_csv = st.file_uploader(
        "可選：自行上傳 CSV",
        type=["csv"],
        help="若有上傳 CSV，辨識會使用這份檔案；否則預設使用剛完成數位化的 ECG CSV。",
    )
    classify_clicked = st.button("執行心律不整辨識", use_container_width=True)

    # ---- ECG 數位化：只在按下數位化按鈕時觸發 ----
    if digitize_clicked:
        if uploaded_file is None:
            st.warning("請先上傳 ECG 影像")
        else:
            img = Image.open(uploaded_file)
            img_bytes = uploaded_file.getvalue()

            try:
                with st.spinner("影像前處理中"):
                    processed_cv_img = preprocess_ecg_image(img)

                with st.spinner(f"訊號數位化中 — {DEVICE_STR}"):
                    file_stem = os.path.splitext(uploaded_file.name)[0]
                    final_csv_path = f"{file_stem}.csv"
                    # 為避免 Windows cp950 在 subprocess 參數與 cv2.imwrite 無法處理非 ASCII 檔名，
                    # 中繼檔一律使用 ASCII 安全名稱，成功後再把 CSV 重新命名為使用者的檔名。
                    temp_img_path = "temp_preprocessed_input.jpg"
                    temp_csv_path = "temp_ecg_signal.csv"
                    # 舊暫存檔若被 Excel 等程式鎖住會導致 PermissionError，先嘗試清除；
                    # 若清不掉就改用 PID-based 暫存名，避免整個分析卡在寫檔階段
                    for p in (temp_img_path, temp_csv_path):
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except PermissionError:
                                pass
                    if os.path.exists(temp_csv_path):
                        temp_csv_path = f"temp_ecg_signal_{os.getpid()}.csv"
                    if os.path.exists(temp_img_path):
                        temp_img_path = f"temp_preprocessed_input_{os.getpid()}.jpg"
                    ok, buf = cv2.imencode(".jpg", processed_cv_img)
                    if not ok:
                        raise RuntimeError("前處理影像編碼失敗")
                    with open(temp_img_path, "wb") as f:
                        f.write(buf.tobytes())
                    result = subprocess.run(
                        [sys.executable, "ecg_digitize.py",
                         "--input", temp_img_path,
                         "--output", temp_csv_path],
                        capture_output=True, text=True, timeout=300,
                        encoding="utf-8", errors="replace",
                    )
                    if result.returncode != 0:
                        raise RuntimeError(result.stderr)
                    # 以 Python 原生檔案 API 搬移，unicode 安全
                    if os.path.exists(final_csv_path):
                        os.remove(final_csv_path)
                    os.replace(temp_csv_path, final_csv_path)
                    save_path = final_csv_path

                st.session_state.analysis = {
                    "image_bytes": img_bytes,
                    "filename": uploaded_file.name,
                    "csv_path": save_path,
                    "source": "數位化 ECG CSV",
                }
                st.success(f"ECG 數位化完成：{save_path}")

            except Exception as e:
                st.error(f"ECG 數位化失敗：{e}")

    # ---- 心律不整辨識：預設讀取數位化 CSV；使用者上傳 CSV 時優先使用上傳檔 ----
    if classify_clicked:
        try:
            if uploaded_csv is not None:
                save_path = save_uploaded_csv(uploaded_csv)
                source = "使用者上傳 CSV"
            elif st.session_state.get("analysis") and st.session_state.analysis.get("csv_path"):
                save_path = st.session_state.analysis["csv_path"]
                source = "數位化 ECG CSV"
            else:
                save_path = None
                source = None

            if save_path is None:
                st.warning("請先執行 ECG 數位化，或自行上傳 CSV。")
            else:
                with st.spinner(f"心律不整辨識中 — {DEVICE_STR}"):
                    dominant, counts, total = classify_with_boki(save_path)

                final_result = BOKI_LABELS[dominant]
                probability = counts.get(dominant, 0) / total if total else 0

                analysis = {} if uploaded_csv is not None else (st.session_state.get("analysis") or {})
                analysis.update({
                    "csv_path": save_path,
                    "source": source,
                    "dominant": dominant,
                    "counts": counts,
                    "total": total,
                    "final_result": final_result,
                    "probability": probability,
                })
                if uploaded_csv is not None:
                    analysis["filename"] = uploaded_csv.name
                st.session_state.analysis = analysis

                st.session_state.messages.append({
                    "role": "system",
                    "content": (
                        f"【系統提示】使用者剛剛完成心律不整辨識，"
                        f"最高機率結果為「{final_result}」（{probability:.1%}）。\n"
                        "請在接下來的對話中主動關心這個結果，並提供衛教建議。"
                    ),
                })
                st.success("心律不整辨識完成")

        except Exception as e:
            st.error(f"心律不整辨識失敗：{e}")

    # ---- 渲染結果（每次 rerun 都會從 session_state 重繪，聊天後也不會消失） ----
    analysis = st.session_state.get("analysis")
    if analysis is not None:
        from io import BytesIO
        if analysis.get("image_bytes"):
            st.image(
                Image.open(BytesIO(analysis["image_bytes"])),
                caption=analysis.get("filename", "ECG image"),
                use_container_width=True,
            )
        st.markdown(
            f"<div class='status-line'>訊號已輸出至 "
            f"<code>{analysis['csv_path']}</code></div>",
            unsafe_allow_html=True,
        )

        if analysis.get("final_result"):
            st.markdown(
                "<div class='eyebrow' style='margin-top:0.8rem'>Detection Result</div>"
                "<div class='section-title'>最大機率病症</div>",
                unsafe_allow_html=True,
            )
            st.metric(
                label=analysis.get("source", "ECG CSV"),
                value=analysis["final_result"],
                delta=f"{analysis.get('probability', 0):.1%}",
            )
        else:
            st.info("已完成 ECG 數位化。若要辨識心律不整，請按「執行心律不整辨識」。")

        if st.button("清除結果", use_container_width=True):
            st.session_state.analysis = None
            st.rerun()

    st.divider()

    st.header("個人資料")
    with st.expander("編輯", expanded=True):
        user_name = st.text_input("姓名", placeholder="王小明", value="使用者")
        age = st.number_input("年齡", min_value=1, max_value=120, value=30)
        gender = st.selectbox("性別", ["男", "女", "其他"])
        # 👇 新增身高與體重
        height_input = st.number_input("身高 (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
        weight_input = st.number_input("體重 (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1)
        medical_history = st.text_input("過去病史", placeholder="高血壓、糖尿病")
        location = st.text_input("目前位置", value="台南市")

    st.divider()

    st.header("家屬通報")
    if st.button("傳送今日健康報告", use_container_width=True):
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        
        # 👇 動態獲取心電圖分析結果
        ecg_summary = "今日尚未上傳心電圖資料。"
        if st.session_state.get("analysis") is not None:
            if st.session_state.analysis.get("final_result"):
                ecg_summary = f"已完成心律不整辨識，最高機率結果為：「{st.session_state.analysis['final_result']}」。"
            else:
                ecg_summary = "已完成 ECG 數位化，尚未執行心律不整辨識。"
        
        report_content = f"""心電圖智慧助理 — 每日健康報告

日期　{today_str}
姓名　{user_name}（{age} 歲）
身高　{height_input} cm
體重　{weight_input} kg
位置　{location}

— 系統摘要 —
【心電圖狀態】：{ecg_summary}

— 建議 —
請持續關注身體狀況，若心電圖顯示異常，請盡速攜帶詳細報告就醫。

本訊息由心電圖智慧助理自動發送。"""

        with st.spinner("傳送中"):
            success = send_line_report(report_content)

        if success:
            st.success("報告已送達家屬 LINE")
        else:
            st.error("發送失敗，請檢查終端機訊息或 API Key 設定")
# ==========================================
# 主畫面
# ==========================================
st.markdown(
    """
    <div class="page-hero">
      <h1>心電圖智慧助理</h1>
      <span class="sub">ECG Intelligence Assistant</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "您好，我可以協助解讀心電圖分析結果、回答健康相關問題，或為您查詢附近的醫療院所與飲食建議。",
        }
    ]

for message in st.session_state.messages:
    # 1. 自動判斷資料型態，安全地取得 role 與 content
    role = message["role"] if isinstance(message, dict) else message.role
    content = message.get("content") if isinstance(message, dict) else message.content
    
    # 2. 如果不是系統或工具訊息，而且 content 不是空的，才顯示在畫面上
    if role not in ["system", "tool"] and content:
        with st.chat_message(role):
            st.markdown(content)

if prompt := st.chat_input("輸入您的問題或需求"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        # 👇 升級版的 System Prompt：加入身高體重與嚴格的防護指令
        system_prompt = f"""
        你是一位專業且溫暖的醫療助理，同時也具備專業健身教練的知識。
        正在與你對話的使用者是「{user_name}」，年齡 {age}，性別 {gender}，身高 {height_input} cm，體重 {weight_input} kg，病史：{medical_history}，位置：{location}。
        
        【重要操作指令】：
        1. 當使用者要求「尋找附近診所」或「推薦食譜」但沒提到地點時，自動使用「{location}」查詢。
        2. 當使用者要求運動建議時，請務必呼叫 `generate_workout_plan` 工具。
        
        3. ⚠️【不可跨越的底線 - 絕對嚴格執行】：
           - 當你收到工具回傳的【運動菜單】時，『絕對不可以』擅自摘要、省略或隱藏內容！
           - 菜單中的「暖身/主運動/緩和時間分配」，以及「每一個具體的訓練動作、重量 (kg)、組數、次數、呼吸提示」，都必須【一字不漏、完整地】列在你的回答中。
           - 不可以自己捏造或通靈其他的訓練計畫。
        """
        
        messages_to_send = [{"role": "system", "content": system_prompt}] + st.session_state.messages
        
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages_to_send,
            tools=tools_definition,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        if response_message.tool_calls:
            st.markdown(
                "<div class='status-line'>正在查詢與計算資料...</div>",
                unsafe_allow_html=True,
            )
            st.session_state.messages.append(response_message) 
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_response = "系統訊息：未知的工具或執行失敗" 
                
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "search_nearby_clinics":
                        function_response = search_nearby_clinics(
                            location=function_args.get("location"),
                            specialty=function_args.get("specialty", "心臟內科")
                        )
                    
                    elif function_name == "search_healthy_recipe":
                        function_response = search_healthy_recipe(
                            keyword=function_args.get("keyword")
                        )
                        
                    elif function_name == "send_line_report":
                        success = send_line_report(
                            report_text=function_args.get("report_text")
                        )
                        function_response = "系統訊息：已成功將健康報告發送至家屬的 LINE。" if success else "系統訊息：發送失敗。"
                        
                    # 👇 新增執行運動菜單的邏輯 (並傳入側邊欄的體重)
                    elif function_name == "generate_workout_plan":
                        function_response = generate_workout_plan(
                            goal=function_args.get("goal", "維持健康"),
                            duration_minutes=function_args.get("duration_minutes", 30),
                            intensity=function_args.get("intensity", "中"),
                            medical_history=medical_history,
                            weight=weight_input 
                        )
                    elif function_name == "search_nearby_restaurants":
                        function_response = search_nearby_restaurants(
                            location=function_args.get("location"),
                            keyword=function_args.get("keyword", "健康餐盒")
                        )    
                    
                    else:
                        function_response = f"系統訊息：找不到名為 {function_name} 的工具。"
                        
                except Exception as e:
                    function_response = f"系統訊息：解析工具參數時發生錯誤 ({str(e)})"
                
                st.session_state.messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                })
                    
            messages_to_send = [{"role": "system", "content": system_prompt}] + st.session_state.messages
            second_response = client.chat.completions.create(
                model="gpt-5-nano",
                messages=messages_to_send
            )
            bot_reply = second_response.choices[0].message.content
            st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            
        else:
            bot_reply = response_message.content
            st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})

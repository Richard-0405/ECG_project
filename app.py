import streamlit as st
import requests # 🌟 新增：用來呼叫外部 API 的套件
import datetime
import io
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
import re
import shutil
import torch
import uuid
from scipy import stats, signal as scipy_signal
from fractions import Fraction
from pathlib import Path
import plotly.graph_objects as go

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


def get_secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


BACKEND_URL = (os.getenv("ECG_BACKEND_URL") or get_secret("BACKEND_URL", "http://127.0.0.1:8000")).rstrip("/")
APP_ROOT = Path(__file__).resolve().parent
USER_DATA_ROOT = Path(os.getenv("ECG_USER_DATA_ROOT", APP_ROOT / "user_data"))
RUNTIME_DATA_ROOT = Path(os.getenv("ECG_RUNTIME_DATA_ROOT", APP_ROOT / "runtime_data"))
GUEST_RUNTIME_RETENTION_DAYS = int(os.getenv("ECG_GUEST_RUNTIME_RETENTION_DAYS", "1"))

TEXT = {
    "zh": {
        "language": "語言",
        "language_zh": "中文",
        "language_en": "English",
        "workbench": "工作台",
        "memory": "使用者記憶",
        "backend_offline": "後端尚未連線，目前為訪客模式，不會儲存資料。",
        "logged_in": "已登入：{name}",
        "logout": "登出並切換訪客模式",
        "guest_mode": "訪客模式：可以使用分析功能，但不會儲存歷史資料。",
        "name": "姓名",
        "id_last4": "身分證後四碼",
        "login": "登入",
        "create_user": "建立使用者",
        "enter_name_code": "請輸入姓名與 4 碼數字。",
        "login_success": "登入成功",
        "login_failed": "登入失敗，請確認姓名與四碼。",
        "user_created": "使用者已建立並登入",
        "create_failed": "建立失敗，請確認後端伺服器是否已啟動。",
        "loaded": "已載入：{time}",
        "admin": "後台管理",
        "history_count": "歷史 ECG 紀錄：{count} 筆",
        "delete_user": "刪除此使用者",
        "deleted_user": "已刪除使用者",
        "login_to_history": "登入後才會顯示歷史 ECG 紀錄。",
        "digitize": "ECG 數位化",
        "upload_image": "上傳 12 導程 ECG 影像",
        "run_digitize": "執行 ECG 數位化",
        "arrhythmia": "心律不整辨識",
        "optional_csv": "可選：自行上傳 CSV",
        "csv_help": "若有上傳 CSV，辨識會使用這份檔案；否則預設使用剛完成數位化的 ECG CSV。",
        "run_classify": "執行心律不整辨識",
        "need_image": "請先上傳 ECG 影像",
        "preprocess": "影像前處理中",
        "digitizing": "訊號數位化中 — {device}",
        "digitize_done": "ECG 數位化完成：{path}",
        "digitize_failed": "ECG 數位化失敗：{error}",
        "need_csv": "請先執行 ECG 數位化，或自行上傳 CSV。",
        "classifying": "心律不整辨識中 — {device}",
        "uploaded_csv": "使用者上傳 CSV",
        "digitized_csv": "數位化 ECG CSV",
        "classify_done": "心律不整辨識完成",
        "classify_failed": "心律不整辨識失敗：{error}",
        "signal_output": "訊號已輸出至",
        "detection_result": "Detection Result",
        "top_condition": "最大機率病症",
        "digitized_only": "已完成 ECG 數位化。若要辨識心律不整，請按「執行心律不整辨識」。",
        "clear_result": "清除結果",
        "profile": "個人資料",
        "edit": "編輯",
        "age": "年齡",
        "gender": "性別",
        "gender_options": ["男", "女", "其他"],
        "height": "身高 (cm)",
        "weight": "體重 (kg)",
        "history": "過去病史",
        "history_placeholder": "高血壓、糖尿病",
        "location": "目前位置",
        "notes": "備註 / 長期記憶",
        "knowledge_level": "醫療知識程度",
        "knowledge_options": ["初階（一般人）", "中階（稍微有醫療知識）", "高階（醫療背景相關人員）"],
        "knowledge_values": ["beginner", "intermediate", "advanced"],
        "history_view": "過往辨識結果",
        "assistant_tab": "AI 助理",
        "profile_edit": "個人資料編輯",
        "rhythm_detection": "心律偵測",
        "show_history": "查看過往辨識結果",
        "hide_history": "收起過往辨識結果",
        "history_range": "資料範圍",
        "history_range_options": ["近 7 天", "近 1 個月", "全部"],
        "no_history": "目前沒有可顯示的 ECG 紀錄。",
        "select_record": "選擇一筆紀錄",
        "select_leads": "選擇導程",
        "csv_not_found": "找不到這筆紀錄的 CSV 檔案。",
        "download_csv": "下載此筆 CSV",
        "download_report": "下載此筆 JSON 報告",
        "medical_disclaimer": "提醒：本系統僅供展示與輔助理解，不能取代醫師診斷。若有胸痛、呼吸困難、昏厥、冒冷汗或症狀快速惡化，請立即撥打當地緊急電話或前往急診。",
        "privacy_notice": "訪客暫存檔會定期清除；登入使用者的歷史資料會依帳號分開保存。",
        "environment_panel": "今日環境風險",
        "environment_refresh": "更新環境資料",
        "environment_unavailable": "目前無法取得環境資料，請確認位置名稱或稍後再試。",
        "temperature": "溫度",
        "humidity": "濕度",
        "wind_speed": "風速",
        "pm25": "PM2.5",
        "us_aqi": "AQI",
        "env_risk_low": "環境風險低：目前空氣品質與天氣狀況看起來穩定。",
        "env_risk_moderate": "環境風險中等：敏感族群或心血管高風險者，建議留意戶外活動時間。",
        "env_risk_high": "環境風險偏高：若有胸悶、喘、心悸或身體不適，建議減少戶外活動並尋求醫療協助。",
        "emergency_button": "緊急通知家人",
        "emergency_sent": "緊急通知已送出。",
        "emergency_failed": "緊急通知發送失敗，請確認 LINE 設定。",
        "guest_profile": "目前是訪客模式，個人資料不會儲存。登入或建立使用者後才會寫入後端。",
        "save_profile": "儲存使用者資料",
        "login_first": "請先登入或建立使用者。",
        "profile_saved": "使用者資料已儲存",
        "profile_save_failed": "儲存失敗，請確認後端伺服器是否已啟動。",
        "line_report": "家屬通報",
        "send_report": "傳送今日健康報告",
        "sending": "傳送中",
        "line_success": "報告已送達家屬 LINE",
        "line_failed": "發送失敗，請檢查終端機訊息或 API Key 設定",
        "title": "心電圖智慧助理",
        "chat_placeholder": "輸入您的問題或需求",
        "default_assistant": "您好，我可以協助解讀心電圖分析結果、回答健康相關問題，或為您查詢附近的醫療院所與飲食建議。",
        "tool_status": "正在查詢與計算資料...",
    },
    "en": {
        "language": "Language",
        "language_zh": "中文",
        "language_en": "English",
        "workbench": "Workbench",
        "memory": "User Memory",
        "backend_offline": "Backend is offline. Guest mode is active and data will not be saved.",
        "logged_in": "Logged in: {name}",
        "logout": "Log Out / Guest Mode",
        "guest_mode": "Guest mode: analysis is available, but history will not be saved.",
        "name": "Name",
        "id_last4": "Last 4 ID digits",
        "login": "Log In",
        "create_user": "Create User",
        "enter_name_code": "Please enter a name and a 4-digit code.",
        "login_success": "Login successful",
        "login_failed": "Login failed. Please check the name and 4-digit code.",
        "user_created": "User created and logged in",
        "create_failed": "Creation failed. Please make sure the backend is running.",
        "loaded": "Loaded: {time}",
        "admin": "Admin",
        "history_count": "ECG history: {count} record(s)",
        "delete_user": "Delete This User",
        "deleted_user": "User deleted",
        "login_to_history": "Log in to view ECG history.",
        "digitize": "ECG Digitization",
        "upload_image": "Upload 12-lead ECG image",
        "run_digitize": "Run ECG Digitization",
        "arrhythmia": "Arrhythmia Detection",
        "optional_csv": "Optional: upload CSV",
        "csv_help": "If a CSV is uploaded, detection will use it; otherwise it uses the latest digitized ECG CSV.",
        "run_classify": "Run Arrhythmia Detection",
        "need_image": "Please upload an ECG image first.",
        "preprocess": "Preprocessing image",
        "digitizing": "Digitizing signal — {device}",
        "digitize_done": "ECG digitization completed: {path}",
        "digitize_failed": "ECG digitization failed: {error}",
        "need_csv": "Please digitize an ECG first, or upload a CSV.",
        "classifying": "Running arrhythmia detection — {device}",
        "uploaded_csv": "Uploaded CSV",
        "digitized_csv": "Digitized ECG CSV",
        "classify_done": "Arrhythmia detection completed",
        "classify_failed": "Arrhythmia detection failed: {error}",
        "signal_output": "Signal saved to",
        "detection_result": "Detection Result",
        "top_condition": "Highest Probability Condition",
        "digitized_only": "ECG digitization is complete. Click “Run Arrhythmia Detection” to classify the rhythm.",
        "clear_result": "Clear Result",
        "profile": "Profile",
        "edit": "Edit",
        "age": "Age",
        "gender": "Gender",
        "gender_options": ["Male", "Female", "Other"],
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "history": "Medical History",
        "history_placeholder": "Hypertension, diabetes",
        "location": "Current Location",
        "notes": "Notes / Long-Term Memory",
        "knowledge_level": "Medical Knowledge Level",
        "knowledge_options": ["Beginner (general public)", "Intermediate (some medical knowledge)", "Advanced (medical background)"],
        "knowledge_values": ["beginner", "intermediate", "advanced"],
        "history_view": "Past Detection Results",
        "assistant_tab": "AI Assistant",
        "profile_edit": "Edit Profile",
        "rhythm_detection": "Rhythm Detection",
        "show_history": "View Past Results",
        "hide_history": "Hide Past Results",
        "history_range": "Date Range",
        "history_range_options": ["Last 7 days", "Last 1 month", "All"],
        "no_history": "No ECG records are available yet.",
        "select_record": "Select a record",
        "select_leads": "Select leads",
        "csv_not_found": "The CSV file for this record was not found.",
        "download_csv": "Download CSV",
        "download_report": "Download JSON Report",
        "medical_disclaimer": "Reminder: this demo is for support and education only, not a medical diagnosis. If you have chest pain, shortness of breath, fainting, cold sweats, or rapidly worsening symptoms, call local emergency services or go to the ER immediately.",
        "privacy_notice": "Guest temporary files are cleaned up regularly; logged-in user history is stored separately by account.",
        "environment_panel": "Today's Environmental Risk",
        "environment_refresh": "Refresh Environment Data",
        "environment_unavailable": "Environment data is unavailable. Please check the location name or try again later.",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "wind_speed": "Wind",
        "pm25": "PM2.5",
        "us_aqi": "AQI",
        "env_risk_low": "Low environmental risk: current air quality and weather look stable.",
        "env_risk_moderate": "Moderate environmental risk: sensitive users or people with cardiovascular risk should watch outdoor activity time.",
        "env_risk_high": "Higher environmental risk: if chest tightness, shortness of breath, palpitations, or discomfort occurs, reduce outdoor activity and seek medical help.",
        "emergency_button": "Emergency Alert",
        "emergency_sent": "Emergency alert sent.",
        "emergency_failed": "Emergency alert failed. Please check LINE settings.",
        "guest_profile": "Guest mode is active. Profile data will not be saved until you log in or create a user.",
        "save_profile": "Save Profile",
        "login_first": "Please log in or create a user first.",
        "profile_saved": "Profile saved",
        "profile_save_failed": "Save failed. Please make sure the backend is running.",
        "line_report": "Family Notification",
        "send_report": "Send Today's Health Report",
        "sending": "Sending",
        "line_success": "Report sent to LINE",
        "line_failed": "Send failed. Please check terminal messages or API keys.",
        "title": "ECG Intelligence Assistant",
        "chat_placeholder": "Type your question or request",
        "default_assistant": "Hello. I can help interpret ECG analysis results, answer health questions, and look up nearby clinics or nutrition suggestions.",
        "tool_status": "Querying and calculating...",
    },
}


def get_lang():
    return st.session_state.get("language", "zh")


def t(key, **kwargs):
    value = TEXT.get(get_lang(), TEXT["zh"]).get(key, TEXT["zh"].get(key, key))
    if isinstance(value, str) and kwargs:
        return value.format(**kwargs)
    return value


def first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_environment_summary(location_text, lang="zh"):
    location_text = (location_text or "").strip()
    if not location_text:
        return None

    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": location_text,
            "count": 1,
            "language": "zh" if lang == "zh" else "en",
            "format": "json",
        },
        timeout=10,
    )
    geo_response.raise_for_status()
    geo_results = geo_response.json().get("results") or []
    if not geo_results:
        return None

    place = geo_results[0]
    latitude = place["latitude"]
    longitude = place["longitude"]

    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=10,
    )
    weather_response.raise_for_status()
    weather = weather_response.json()

    air_response = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,us_aqi",
            "timezone": "auto",
        },
        timeout=10,
    )
    air_response.raise_for_status()
    air = air_response.json()

    current = weather.get("current") or {}
    hourly = air.get("hourly") or {}
    times = hourly.get("time") or []
    current_hour = str(current.get("time") or "")[:13]
    air_index = 0
    for idx, value in enumerate(times):
        if str(value).startswith(current_hour):
            air_index = idx
            break

    def hourly_value(name):
        values = hourly.get(name) or []
        if air_index < len(values):
            return values[air_index]
        return values[0] if values else None

    pm25 = hourly_value("pm2_5")
    us_aqi = hourly_value("us_aqi")
    risk_level = "low"
    if first_present(us_aqi, 0) >= 101 or first_present(pm25, 0) >= 35:
        risk_level = "high"
    elif first_present(us_aqi, 0) >= 51 or first_present(pm25, 0) >= 12:
        risk_level = "moderate"

    place_name = ", ".join(
        part for part in [place.get("name"), place.get("admin1"), place.get("country")] if part
    )
    return {
        "location": place_name or location_text,
        "latitude": latitude,
        "longitude": longitude,
        "time": current.get("time"),
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "pm25": pm25,
        "pm10": hourly_value("pm10"),
        "us_aqi": us_aqi,
        "risk_level": risk_level,
    }


def environment_risk_message(summary):
    if not summary:
        return ""
    if summary.get("risk_level") == "high":
        return t("env_risk_high")
    if summary.get("risk_level") == "moderate":
        return t("env_risk_moderate")
    return t("env_risk_low")


def format_environment_report(summary):
    if not summary:
        return "今日環境資料：目前無法取得。"
    return (
        f"今日環境資料：{summary.get('location')}\n"
        f"溫度：{summary.get('temperature')} °C\n"
        f"濕度：{summary.get('humidity')} %\n"
        f"風速：{summary.get('wind_speed')} km/h\n"
        f"PM2.5：{summary.get('pm25')} µg/m³\n"
        f"AQI：{summary.get('us_aqi')}\n"
        f"環境提醒：{environment_risk_message(summary)}"
    )


def render_environment_panel(location_text):
    st.subheader(t("environment_panel"))
    refresh = st.button(t("environment_refresh"), use_container_width=True)
    if refresh:
        fetch_environment_summary.clear()
    try:
        summary = fetch_environment_summary(location_text, get_lang())
    except Exception:
        summary = None

    if not summary:
        st.info(t("environment_unavailable"))
        st.session_state.environment_summary = None
        return None

    metric_cols = st.columns(5)
    metric_cols[0].metric(t("temperature"), f"{summary.get('temperature')} °C")
    metric_cols[1].metric(t("humidity"), f"{summary.get('humidity')} %")
    metric_cols[2].metric(t("wind_speed"), f"{summary.get('wind_speed')} km/h")
    metric_cols[3].metric(t("pm25"), f"{summary.get('pm25')} µg/m³")
    metric_cols[4].metric(t("us_aqi"), summary.get("us_aqi"))

    risk_message = environment_risk_message(summary)
    if summary.get("risk_level") == "high":
        st.error(risk_message)
    elif summary.get("risk_level") == "moderate":
        st.warning(risk_message)
    else:
        st.success(risk_message)
    st.caption(summary.get("location"))
    st.session_state.environment_summary = summary
    return summary


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
# 自動從 secrets.toml 或雲端環境變數讀取 API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or get_secret("OPENAI_API_KEY"))
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY") or get_secret("GOOGLE_MAPS_API_KEY")) # 🌟 新增：初始化 Google Maps 客戶端

st.set_page_config(page_title="ECG Intelligence Assistant", page_icon="◐", layout="wide")


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


def safe_file_stem(filename, fallback="ecg"):
    stem = Path(filename or fallback).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return (stem or fallback)[:80]


def get_ecg_session_id():
    if "ecg_session_id" not in st.session_state:
        st.session_state.ecg_session_id = uuid.uuid4().hex
    return st.session_state.ecg_session_id


def get_ecg_workspace(user_id=None):
    session_id = get_ecg_session_id()
    if user_id:
        workspace = USER_DATA_ROOT / f"user_{user_id}" / "sessions" / session_id
    else:
        workspace = RUNTIME_DATA_ROOT / "guest_sessions" / session_id
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def new_ecg_run_dir(user_id=None):
    run_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4().hex[:8]}"
    run_dir = get_ecg_workspace(user_id) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def cleanup_old_runtime_files(retention_days=GUEST_RUNTIME_RETENTION_DAYS):
    cutoff = datetime.datetime.now().timestamp() - (retention_days * 86400)
    guest_root = RUNTIME_DATA_ROOT / "guest_sessions"
    if not guest_root.exists():
        return
    for session_dir in guest_root.iterdir():
        try:
            if session_dir.is_dir() and session_dir.stat().st_mtime < cutoff:
                shutil.rmtree(session_dir, ignore_errors=True)
        except OSError:
            pass


def save_uploaded_csv(uploaded_file, user_id=None):
    run_dir = new_ecg_run_dir(user_id)
    file_stem = safe_file_stem(uploaded_file.name, "uploaded_ecg")
    csv_path = run_dir / f"{file_stem}_uploaded.csv"
    csv_path.write_bytes(uploaded_file.getvalue())
    return str(csv_path)


if "runtime_cleanup_done" not in st.session_state:
    cleanup_old_runtime_files()
    st.session_state.runtime_cleanup_done = True


def resolve_ecg_csv_path(csv_path, user_id=None):
    if not csv_path:
        return None
    path = Path(csv_path)
    if path.is_absolute() and path.exists():
        return path

    root = Path(__file__).resolve().parent
    candidates = [root / csv_path, root / Path(csv_path).name]
    if user_id:
        candidates.extend((root / "user_data").glob(f"user_{user_id}_*/csv/{Path(csv_path).name}"))
    candidates.extend((root / "user_data").glob(f"**/{Path(csv_path).name}"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_csv_text(csv_path):
    resolved = resolve_ecg_csv_path(csv_path)
    if not resolved:
        return None
    try:
        return Path(resolved).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Path(resolved).read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return None


def load_record_csv(record, user_id=None):
    if record.get("csv_content"):
        return pd.read_csv(io.StringIO(record["csv_content"])), record.get("csv_filename") or record.get("csv_path")

    resolved = resolve_ecg_csv_path(record.get("csv_path"), user_id)
    if not resolved:
        return None, None
    return pd.read_csv(resolved), resolved


def plot_ecg_dataframe(df, selected_leads):
    if "Time" in df.columns:
        x = df["Time"]
        x_title = "Time (s)"
    else:
        x = df.index
        x_title = "Sample"

    fig = go.Figure()
    for lead in selected_leads:
        if lead in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[lead],
                    mode="lines",
                    name=lead,
                    line=dict(width=1),
                )
            )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=28, b=20),
        xaxis_title=x_title,
        yaxis_title="mV",
        legend_title="Lead",
    )
    return fig, df


def plot_ecg_waveform(csv_path, selected_leads):
    df = pd.read_csv(csv_path)
    return plot_ecg_dataframe(df, selected_leads)


def backend_request(method, path, **kwargs):
    try:
        response = requests.request(
            method,
            f"{BACKEND_URL}{path}",
            timeout=15,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def load_memory_users():
    return backend_request("GET", "/users") or []


def login_memory_user(name, id_last4):
    return backend_request(
        "POST",
        "/login",
        json={"name": name, "id_last4": id_last4},
    )


def save_memory_user(user_id, payload):
    if user_id:
        return backend_request("PUT", f"/users/{user_id}", json=payload)
    return backend_request("POST", "/users", json=payload)


def delete_memory_user(user_id):
    if not user_id:
        return None
    return backend_request("DELETE", f"/users/{user_id}")


def load_ecg_records(user_id):
    if not user_id:
        return []
    return backend_request("GET", f"/users/{user_id}/records") or []


def default_messages():
    return [{"role": "assistant", "content": t("default_assistant")}]


def load_chat_messages(user_id):
    if not user_id:
        return default_messages()
    rows = backend_request("GET", f"/users/{user_id}/chat") or []
    messages = [
        {"role": row["role"], "content": row["content"]}
        for row in rows
        if row.get("role") in {"user", "assistant"} and row.get("content")
    ]
    return messages or default_messages()


def save_chat_message(user_id, role, content):
    if not user_id or role not in {"user", "assistant"} or not content:
        return None
    return backend_request(
        "POST",
        f"/users/{user_id}/chat",
        json={"role": role, "content": content},
    )


def save_ecg_record(user_id, analysis):
    if not user_id or not analysis:
        return None
    csv_path = analysis.get("csv_path")
    payload = {
        "csv_path": csv_path,
        "csv_filename": analysis.get("csv_filename") or (Path(csv_path).name if csv_path else None),
        "csv_content": analysis.get("csv_content") or read_csv_text(csv_path),
        "source": analysis.get("source"),
        "final_result": analysis.get("final_result"),
        "probability": analysis.get("probability"),
        "total_windows": analysis.get("total"),
        "counts": analysis.get("counts"),
    }
    if not payload["csv_path"]:
        return None
    return backend_request("POST", f"/users/{user_id}/records", json=payload)


def user_payload(name, age, gender, height_cm, weight_kg, medical_history, location, notes, id_last4=None, knowledge_level="beginner"):
    return {
        "name": name,
        "id_last4": id_last4,
        "age": int(age) if age is not None else None,
        "gender": gender,
        "height_cm": float(height_cm) if height_cm is not None else None,
        "weight_kg": float(weight_kg) if weight_kg is not None else None,
        "medical_history": medical_history,
        "location": location,
        "notes": notes,
        "knowledge_level": knowledge_level or "beginner",
    }


def knowledge_level_instruction(level):
    if level == "advanced":
        return (
            "使用者具醫療背景，可以使用較完整的臨床語彙、導程判讀、風險分層、鑑別診斷與限制說明；"
            "但仍需清楚標示 AI 不能取代醫師診斷。"
        )
    if level == "intermediate":
        return (
            "使用者有一點醫療知識，可以加入基本醫學名詞，例如 AF、心率、導程、風險因子，"
            "但要同時用簡短白話解釋。"
        )
    return (
        "使用者是一般人，請用白話、短句、少術語、多比喻說明，並明確提醒哪些情況需要立即就醫。"
    )


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
        api_key = os.getenv("SPOONACULAR_API_KEY") or get_secret("SPOONACULAR_API_KEY")
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
            "Authorization": f"Bearer {os.getenv('LINE_CHANNEL_ACCESS_TOKEN') or get_secret('LINE_CHANNEL_ACCESS_TOKEN')}",
            "Content-Type": "application/json"
        }
        data = {
            "to": os.getenv("LINE_TARGET_USER_ID") or get_secret("LINE_TARGET_USER_ID"),
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

if "main_panel" not in st.session_state:
    st.session_state.main_panel = "assistant"

# ==========================================
# 側邊欄 (Sidebar) 區塊
# ==========================================
with st.sidebar:
    st.selectbox(
        "Language / 語言",
        ["zh", "en"],
        format_func=lambda code: TEXT[code]["language_zh"] if code == "zh" else TEXT[code]["language_en"],
        key="language",
    )
    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; gap:0.6rem; padding-bottom:0.4rem;">
          <div style="font-size:0.72rem; letter-spacing:0.22em; text-transform:uppercase;
                      color:var(--muted); font-weight:600;">Console</div>
          <div style="font-size:1.25rem; font-weight:700; letter-spacing:-0.005em;">{t("workbench")}</div>
          <div class="device-badge">{DEVICE_STR}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    if st.button(
        t("assistant_tab"),
        use_container_width=True,
        type="primary" if st.session_state.main_panel == "assistant" else "secondary",
        key="sidebar_assistant_tab",
    ):
        st.session_state.main_panel = "assistant"
        st.rerun()

    if st.button(
        t("history_view"),
        use_container_width=True,
        type="primary" if st.session_state.main_panel == "history" else "secondary",
        key="sidebar_history_tab",
    ):
        st.session_state.main_panel = "history"
        st.rerun()

    if st.button(
        t("profile_edit"),
        use_container_width=True,
        type="primary" if st.session_state.main_panel == "profile" else "secondary",
        key="sidebar_profile_tab",
    ):
        st.session_state.main_panel = "profile"
        st.rerun()

    if st.button(
        t("rhythm_detection"),
        use_container_width=True,
        type="primary" if st.session_state.main_panel == "detection" else "secondary",
        key="sidebar_detection_tab",
    ):
        st.session_state.main_panel = "detection"
        st.rerun()

    st.divider()

    st.header(t("memory"))
    backend_ready = backend_request("GET", "/health") is not None
    if not backend_ready:
        st.warning(t("backend_offline"))

    selected_user = st.session_state.get("active_user")
    if selected_user:
        st.success(t("logged_in", name=selected_user["name"]))
        if st.button(t("logout"), use_container_width=True):
            st.session_state.active_user = None
            st.session_state.active_user_id = None
            st.session_state.messages = default_messages()
            st.rerun()
    else:
        st.info(t("guest_mode"))
        login_name = st.text_input(t("name"), key="login_name")
        login_last4 = st.text_input(
            t("id_last4"),
            max_chars=4,
            type="password",
            key="login_last4",
        )
        login_col, create_col = st.columns(2)
        with login_col:
            if st.button(t("login"), use_container_width=True):
                if not login_name.strip() or len(login_last4.strip()) != 4 or not login_last4.strip().isdigit():
                    st.error(t("enter_name_code"))
                else:
                    user = login_memory_user(login_name.strip(), login_last4.strip())
                    if user:
                        st.session_state.active_user = user
                        st.session_state.active_user_id = user["id"]
                        st.session_state.messages = load_chat_messages(user["id"])
                        st.success(t("login_success"))
                        st.rerun()
                    else:
                        st.error(t("login_failed"))
        with create_col:
            if st.button(t("create_user"), use_container_width=True):
                if not login_name.strip() or len(login_last4.strip()) != 4 or not login_last4.strip().isdigit():
                    st.error(t("enter_name_code"))
                else:
                    user = save_memory_user(
                        None,
                        user_payload(
                            login_name.strip(),
                            None,
                            None,
                            None,
                            None,
                            "",
                            "",
                            "",
                            login_last4.strip(),
                        ),
                    )
                    if user:
                        st.session_state.active_user = user
                        st.session_state.active_user_id = user["id"]
                        st.session_state.messages = default_messages()
                        st.success(t("user_created"))
                        st.rerun()
                    else:
                        st.error(t("create_failed"))

    selected_user = st.session_state.get("active_user")
    if selected_user:
        st.caption(t("loaded", time=selected_user.get("updated_at", "")))

    # Workspace tools are rendered in the main panel to keep the sidebar tidy.

selected_user = st.session_state.get("active_user")
profile_defaults = selected_user or {}
user_name = profile_defaults.get("name", "使用者")
age = int(profile_defaults.get("age") or 30)
gender_options = t("gender_options")
saved_gender = profile_defaults.get("gender")
gender = saved_gender if saved_gender in gender_options else gender_options[0]
height_input = float(profile_defaults.get("height_cm") or 170.0)
weight_input = float(profile_defaults.get("weight_kg") or 65.0)
medical_history = profile_defaults.get("medical_history") or ""
location = profile_defaults.get("location") or "台南市"
memory_notes = profile_defaults.get("notes") or ""
knowledge_options = t("knowledge_options")
knowledge_values = t("knowledge_values")
saved_knowledge = profile_defaults.get("knowledge_level") or "beginner"
knowledge_index = knowledge_values.index(saved_knowledge) if saved_knowledge in knowledge_values else 0
knowledge_label = knowledge_options[knowledge_index]
knowledge_level = knowledge_values[knowledge_index]

# ==========================================
# 主畫面
# ==========================================
st.markdown(
    f"""
    <div class="page-hero">
      <h1>{t("title")}</h1>
      <span class="sub">ECG Intelligence Assistant</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
  if st.session_state.main_panel == "detection":
    st.subheader(t("rhythm_detection"))
    st.warning(t("medical_disclaimer"))
    st.caption(t("privacy_notice"))

    st.header(t("digitize"))
    uploaded_file = st.file_uploader(
        t("upload_image"),
        type=["png", "jpg", "jpeg"],
        label_visibility="visible",
    )
    digitize_clicked = st.button(t("run_digitize"), use_container_width=True, type="primary")

    st.divider()

    st.header(t("arrhythmia"))
    uploaded_csv = st.file_uploader(
        t("optional_csv"),
        type=["csv"],
        help=t("csv_help"),
    )
    classify_clicked = st.button(t("run_classify"), use_container_width=True)

    if digitize_clicked:
        if uploaded_file is None:
            st.warning(t("need_image"))
        else:
            img = Image.open(uploaded_file)
            img_bytes = uploaded_file.getvalue()
            active_user_id = st.session_state.get("active_user_id")

            try:
                with st.spinner(t("preprocess")):
                    processed_cv_img = preprocess_ecg_image(img)

                with st.spinner(t("digitizing", device=DEVICE_STR)):
                    file_stem = safe_file_stem(uploaded_file.name, "digitized_ecg")
                    run_dir = new_ecg_run_dir(active_user_id)
                    temp_img_path = run_dir / f"{file_stem}_preprocessed.jpg"
                    final_csv_path = run_dir / f"{file_stem}.csv"
                    ok, buf = cv2.imencode(".jpg", processed_cv_img)
                    if not ok:
                        raise RuntimeError("前處理影像編碼失敗")
                    temp_img_path.write_bytes(buf.tobytes())
                    result = subprocess.run(
                        [
                            sys.executable,
                            "ecg_digitize.py",
                            "--input",
                            str(temp_img_path),
                            "--output",
                            str(final_csv_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=300,
                        encoding="utf-8",
                        errors="replace",
                    )
                    if result.returncode != 0:
                        raise RuntimeError(result.stderr)
                    save_path = str(final_csv_path)

                st.session_state.analysis = {
                    "image_bytes": img_bytes,
                    "filename": uploaded_file.name,
                    "csv_path": save_path,
                    "csv_filename": Path(save_path).name,
                    "source": t("digitized_csv"),
                }
                save_ecg_record(active_user_id, st.session_state.analysis)
                st.success(t("digitize_done", path=save_path))

            except Exception as e:
                st.error(t("digitize_failed", error=e))

    if classify_clicked:
        try:
            active_user_id = st.session_state.get("active_user_id")
            if uploaded_csv is not None:
                save_path = save_uploaded_csv(uploaded_csv, active_user_id)
                source = t("uploaded_csv")
            elif st.session_state.get("analysis") and st.session_state.analysis.get("csv_path"):
                save_path = st.session_state.analysis["csv_path"]
                source = t("digitized_csv")
            else:
                save_path = None
                source = None

            if save_path is None:
                st.warning(t("need_csv"))
            else:
                with st.spinner(t("classifying", device=DEVICE_STR)):
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
                    analysis["csv_filename"] = Path(save_path).name
                st.session_state.analysis = analysis
                save_ecg_record(active_user_id, analysis)

                st.session_state.messages.append({
                    "role": "system",
                    "content": (
                        f"【系統提示】使用者剛剛完成心律不整辨識，"
                        f"最高機率結果為「{final_result}」（{probability:.1%}）。\n"
                        "請在接下來的對話中主動關心這個結果，並提供衛教建議。"
                    ),
                })
                st.success(t("classify_done"))

        except Exception as e:
            st.error(t("classify_failed", error=e))

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
            f"<div class='status-line'>{t('signal_output')} "
            f"<code>{analysis['csv_path']}</code></div>",
            unsafe_allow_html=True,
        )

        if analysis.get("final_result"):
            st.markdown(
                f"<div class='eyebrow' style='margin-top:0.8rem'>{t('detection_result')}</div>"
                f"<div class='section-title'>{t('top_condition')}</div>",
                unsafe_allow_html=True,
            )
            st.metric(
                label=analysis.get("source", "ECG CSV"),
                value=analysis["final_result"],
                delta=f"{analysis.get('probability', 0):.1%}",
            )
        else:
            st.info(t("digitized_only"))

        if st.button(t("clear_result"), use_container_width=True):
            st.session_state.analysis = None
            st.rerun()

  elif st.session_state.main_panel == "profile":
    st.subheader(t("profile_edit"))
    user_name = st.text_input(
        t("name"),
        placeholder="王小明",
        value=user_name,
    )
    age = st.number_input(
        t("age"),
        min_value=1,
        max_value=120,
        value=age,
    )
    gender_index = gender_options.index(gender) if gender in gender_options else 0
    gender = st.selectbox(t("gender"), gender_options, index=gender_index)
    height_input = st.number_input(
        t("height"),
        min_value=100.0,
        max_value=250.0,
        value=height_input,
        step=0.1,
    )
    weight_input = st.number_input(
        t("weight"),
        min_value=30.0,
        max_value=200.0,
        value=weight_input,
        step=0.1,
    )
    medical_history = st.text_input(
        t("history"),
        placeholder=t("history_placeholder"),
        value=medical_history,
    )
    location = st.text_input(t("location"), value=location)
    memory_notes = st.text_area(t("notes"), value=memory_notes, height=110)
    knowledge_label = st.selectbox(
        t("knowledge_level"),
        knowledge_options,
        index=knowledge_index,
    )
    knowledge_level = knowledge_values[knowledge_options.index(knowledge_label)]

    if not selected_user:
        st.caption(t("guest_profile"))
    if st.button(t("save_profile"), use_container_width=True, type="primary"):
        if not selected_user:
            st.warning(t("login_first"))
        else:
            payload = user_payload(
                user_name,
                age,
                gender,
                height_input,
                weight_input,
                medical_history,
                location,
                memory_notes,
                knowledge_level=knowledge_level,
            )
            saved_user = save_memory_user(st.session_state.get("active_user_id"), payload)
            if saved_user:
                st.session_state.active_user = saved_user
                st.session_state.active_user_id = saved_user["id"]
                st.success(t("profile_saved"))
                st.rerun()
            else:
                st.error(t("profile_save_failed"))

    st.divider()

    st.header(t("line_report"))
    if st.button(t("send_report"), use_container_width=True):
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        ecg_summary = "今日尚未上傳心電圖資料。"
        if st.session_state.get("analysis") is not None:
            if st.session_state.analysis.get("final_result"):
                ecg_summary = f"已完成心律不整辨識，最高機率結果為：「{st.session_state.analysis['final_result']}」。"
            else:
                ecg_summary = "已完成 ECG 數位化，尚未執行心律不整辨識。"
        try:
            environment_summary = st.session_state.get("environment_summary") or fetch_environment_summary(location, get_lang())
        except Exception:
            environment_summary = None
        environment_report = format_environment_report(environment_summary)

        report_content = f"""心電圖智慧助理 — 每日健康報告

日期　{today_str}
姓名　{user_name}（{age} 歲）
身高　{height_input} cm
體重　{weight_input} kg
位置　{location}

— 系統摘要 —
【心電圖狀態】：{ecg_summary}

— 今日環境 —
{environment_report}

— 建議 —
請持續關注身體狀況，若心電圖顯示異常，請盡速攜帶詳細報告就醫。

本訊息由心電圖智慧助理自動發送。"""

        with st.spinner(t("sending")):
            success = send_line_report(report_content)

        if success:
            st.success(t("line_success"))
        else:
            st.error(t("line_failed"))

    if selected_user:
        st.divider()
        with st.expander(t("admin"), expanded=False):
            records = load_ecg_records(selected_user["id"])
            st.write(t("history_count", count=len(records)))
            if st.button(t("delete_user"), use_container_width=True):
                if delete_memory_user(selected_user["id"]):
                    st.session_state.active_user = None
                    st.session_state.active_user_id = None
                    st.session_state.messages = default_messages()
                    st.success(t("deleted_user"))
                    st.rerun()

  elif st.session_state.main_panel == "history":
    st.subheader(t("history_view"))
    st.caption(t("privacy_notice"))
    if not selected_user:
        st.info(t("login_to_history"))
    else:
        range_options = t("history_range_options")
        selected_range = st.selectbox(t("history_range"), range_options, index=1)
        records = load_ecg_records(selected_user["id"])
        now = datetime.datetime.now()
        if selected_range == range_options[0]:
            cutoff = now - datetime.timedelta(days=7)
        elif selected_range == range_options[1]:
            cutoff = now - datetime.timedelta(days=31)
        else:
            cutoff = None

        filtered_records = []
        for record in records:
            created_at = record.get("created_at") or ""
            try:
                created_dt = datetime.datetime.fromisoformat(created_at)
            except Exception:
                created_dt = None
            if cutoff is None or created_dt is None or created_dt >= cutoff:
                filtered_records.append(record)

        if not filtered_records:
            st.info(t("no_history"))
        else:
            history_df = pd.DataFrame(filtered_records)
            display_df = history_df.copy()
            if "probability" in display_df.columns:
                display_df["probability"] = display_df["probability"].apply(
                    lambda value: f"{float(value):.1%}" if value is not None else ""
                )
            display_cols = [
                "id",
                "created_at",
                "final_result",
                "probability",
                "total_windows",
                "source",
                "csv_path",
            ]
            st.dataframe(
                display_df[[col for col in display_cols if col in display_df.columns]],
                use_container_width=True,
                hide_index=True,
            )

            record_options = {
                f"{record.get('created_at', '')} | {record.get('final_result') or 'Digitized ECG'} | {record.get('csv_path')}": record
                for record in filtered_records
            }
            selected_record = record_options[
                st.selectbox(t("select_record"), list(record_options.keys()))
            ]
            record_df, record_source = load_record_csv(selected_record, selected_user["id"])
            if record_df is None:
                st.warning(t("csv_not_found"))
            else:
                available_leads = [
                    lead for lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                    if lead in record_df.columns
                ]
                default_leads = ["II"] if "II" in available_leads else available_leads[:1]
                selected_leads = st.multiselect(
                    t("select_leads"),
                    available_leads,
                    default=default_leads,
                )
                if selected_leads:
                    fig, _ = plot_ecg_dataframe(record_df, selected_leads)
                    st.plotly_chart(fig, use_container_width=True)
                csv_name = f"{safe_file_stem(selected_record.get('csv_filename') or selected_record.get('csv_path') or 'ecg_record')}.csv"
                report_name = f"{safe_file_stem(selected_record.get('csv_filename') or selected_record.get('csv_path') or 'ecg_record')}_report.json"
                download_col1, download_col2 = st.columns(2)
                with download_col1:
                    st.download_button(
                        t("download_csv"),
                        data=record_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name=csv_name,
                        mime="text/csv",
                        use_container_width=True,
                    )
                with download_col2:
                    st.download_button(
                        t("download_report"),
                        data=json.dumps(selected_record, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                        file_name=report_name,
                        mime="application/json",
                        use_container_width=True,
                    )

  else:
    st.warning(t("medical_disclaimer"))
    render_environment_panel(location)
    st.divider()
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_messages(st.session_state.get("active_user_id"))

    for message in st.session_state.messages:
        # 1. 自動判斷資料型態，安全地取得 role 與 content
        role = message["role"] if isinstance(message, dict) else message.role
        content = message.get("content") if isinstance(message, dict) else message.content
        
        # 2. 如果不是系統或工具訊息，而且 content 不是空的，才顯示在畫面上
        if role not in ["system", "tool"] and content:
            with st.chat_message(role):
                st.markdown(content)

    emergency_col, _ = st.columns([1.2, 5])
    with emergency_col:
        if st.button(t("emergency_button"), use_container_width=True, type="primary"):
            emergency_text = f"""緊急通知

    我現在感到極度不舒服，可能需要立即協助。
    請盡快聯絡我，或前往我的位置確認狀況。

    姓名：{user_name}
    位置：{location}
    時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
            if send_line_report(emergency_text):
                st.success(t("emergency_sent"))
            else:
                st.error(t("emergency_failed"))

    if prompt := st.chat_input(t("chat_placeholder")):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat_message(st.session_state.get("active_user_id"), "user", prompt)
        
        with st.chat_message("assistant"):
            # 👇 升級版的 System Prompt：加入身高體重與嚴格的防護指令
            system_prompt = f"""
            你是一位專業且溫暖的醫療助理，同時也具備專業健身教練的知識。
            目前介面語言：{"English，請優先使用英文回覆。" if get_lang() == "en" else "中文，請優先使用繁體中文回覆。"}
            正在與你對話的使用者是「{user_name}」，年齡 {age}，性別 {gender}，身高 {height_input} cm，體重 {weight_input} kg，病史：{medical_history}，位置：{location}。
            長期記憶/備註：{memory_notes}
            使用者醫療知識程度：{knowledge_label}。{knowledge_level_instruction(knowledge_level)}
            今日環境資料：{format_environment_report(st.session_state.get("environment_summary"))}
            
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
                    f"<div class='status-line'>{t('tool_status')}</div>",
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
                save_chat_message(st.session_state.get("active_user_id"), "assistant", bot_reply)
                
            else:
                bot_reply = response_message.content
                st.markdown(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                save_chat_message(st.session_state.get("active_user_id"), "assistant", bot_reply)

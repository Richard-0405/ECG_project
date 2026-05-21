import io
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


ROOT = Path(__file__).parent
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def get_secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


BACKEND_URL = (os.getenv("ECG_BACKEND_URL") or get_secret("BACKEND_URL", "http://127.0.0.1:8000")).rstrip("/")
ADMIN_PASSWORD = os.getenv("ECG_ADMIN_PASSWORD") or get_secret("ADMIN_PASSWORD", "")


def api_get(path):
    try:
        response = requests.get(f"{BACKEND_URL}{path}", timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"無法連線後端：{exc}")
        return None


def resolve_csv_path(csv_path):
    path = Path(csv_path)
    if path.is_absolute() and path.exists():
        return path
    root_path = ROOT / csv_path
    if root_path.exists():
        return root_path
    for candidate in (ROOT / "user_data").glob(f"**/{Path(csv_path).name}"):
        if candidate.exists():
            return candidate
    return None


def load_record_csv(record):
    if record.get("csv_content"):
        return pd.read_csv(io.StringIO(record["csv_content"])), record.get("csv_filename") or "database_csv"

    csv_path = record.get("csv_path")
    if not csv_path:
        return None, None
    resolved = resolve_csv_path(csv_path)
    if not resolved:
        return None, csv_path
    return pd.read_csv(resolved), resolved


def plot_ecg(df, selected_leads):
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
        height=460,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title=x_title,
        yaxis_title="mV",
        legend_title="Lead",
    )
    return fig


def format_probability(value):
    if value is None:
        return ""
    try:
        return f"{float(value):.1%}"
    except Exception:
        return str(value)


st.set_page_config(page_title="ECG Doctor Admin", page_icon="ECG", layout="wide")

if ADMIN_PASSWORD:
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        st.title("ECG Doctor Admin")
        st.caption("請輸入後台管理密碼")
        password = st.text_input("管理密碼", type="password")
        if st.button("登入後台", type="primary"):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("密碼錯誤")
        st.stop()
else:
    st.warning("尚未設定 ADMIN_PASSWORD。正式公開前請務必在 Streamlit secrets 設定後台密碼。")

st.title("ECG Doctor Admin")
st.caption("醫師後台：查看使用者資料、ECG 波形、分類結果與最近 7 天對話紀錄")

users = api_get("/users")
if users is None:
    st.stop()

if not users:
    st.info("目前沒有使用者資料。")
    st.stop()

with st.sidebar:
    st.header("使用者")
    user_options = {
        f"ID {user['id']} - {user['name']}": user
        for user in users
    }
    selected_label = st.selectbox("選擇使用者", list(user_options.keys()))
    selected_user = user_options[selected_label]
    st.caption(f"Backend: {BACKEND_URL}")
    if ADMIN_PASSWORD and st.button("登出後台", use_container_width=True):
        st.session_state.admin_authenticated = False
        st.rerun()

records = api_get(f"/users/{selected_user['id']}/records") or []
chat_messages = api_get(f"/users/{selected_user['id']}/chat") or []

profile_tab, ecg_tab, chat_tab, files_tab = st.tabs([
    "個人資料",
    "ECG / 分類",
    "對話紀錄",
    "儲存資訊",
])

with profile_tab:
    st.subheader("個人資料")
    col1, col2, col3 = st.columns(3)
    col1.metric("使用者 ID", selected_user.get("id"))
    col2.metric("姓名", selected_user.get("name") or "")
    col3.metric("身分證後四碼", selected_user.get("id_last4") or "")

    profile_rows = [
        ("年齡", selected_user.get("age")),
        ("性別", selected_user.get("gender")),
        ("身高 (cm)", selected_user.get("height_cm")),
        ("體重 (kg)", selected_user.get("weight_kg")),
        ("醫療知識程度", selected_user.get("knowledge_level")),
        ("病史", selected_user.get("medical_history")),
        ("位置", selected_user.get("location")),
        ("備註 / 長期記憶", selected_user.get("notes")),
        ("建立時間", selected_user.get("created_at")),
        ("更新時間", selected_user.get("updated_at")),
    ]
    st.dataframe(
        pd.DataFrame(profile_rows, columns=["欄位", "內容"]),
        use_container_width=True,
        hide_index=True,
    )

with ecg_tab:
    st.subheader("ECG 紀錄與分類")
    if not records:
        st.info("此使用者目前沒有 ECG 紀錄。")
    else:
        records_df = pd.DataFrame(records)
        display_df = records_df.copy()
        if "probability" in display_df.columns:
            display_df["probability"] = display_df["probability"].apply(format_probability)
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
            f"Record {record['id']} - {record.get('created_at', '')} - {record.get('final_result') or 'Digitized CSV'}": record
            for record in records
        }
        selected_record_label = st.selectbox("選擇 ECG 紀錄", list(record_options.keys()))
        selected_record = record_options[selected_record_label]

        col1, col2, col3 = st.columns(3)
        col1.metric("最大機率病症", selected_record.get("final_result") or "尚未分類")
        col2.metric("機率", format_probability(selected_record.get("probability")))
        col3.metric("視窗數", selected_record.get("total_windows") or "")

        st.write("分類計數")
        st.json(selected_record.get("counts") or {})

        df, source = load_record_csv(selected_record)
        if source:
            st.caption(f"CSV：{source}")
        if df is None:
            st.warning("找不到這筆紀錄的 CSV 內容。若是在雲端部署，請確認前台已更新並會送出 csv_content。")
        else:
            available_leads = [lead for lead in LEADS if lead in df.columns]
            default_leads = ["II"] if "II" in available_leads else available_leads[:1]
            selected_leads = st.multiselect(
                "選擇導程",
                available_leads,
                default=default_leads,
            )
            if selected_leads:
                st.plotly_chart(plot_ecg(df, selected_leads), use_container_width=True)
            with st.expander("CSV 前 20 列"):
                st.dataframe(df.head(20), use_container_width=True)

with chat_tab:
    st.subheader("最近 7 天對話紀錄")
    if not chat_messages:
        st.info("此使用者目前沒有最近 7 天的對話紀錄。")
    else:
        chat_df = pd.DataFrame(chat_messages)
        st.dataframe(
            chat_df[[col for col in ["created_at", "role", "content"] if col in chat_df.columns]],
            use_container_width=True,
            hide_index=True,
        )
        for message in chat_messages:
            role = "user" if message.get("role") == "user" else "assistant"
            with st.chat_message(role):
                st.caption(message.get("created_at", ""))
                st.markdown(message.get("content", ""))

with files_tab:
    st.subheader("儲存資訊")
    st.write("正式雲端版會以後端資料庫為主要資料來源。")
    storage_rows = []
    for record in records:
        csv_content = record.get("csv_content") or ""
        storage_rows.append({
            "record_id": record.get("id"),
            "csv_filename": record.get("csv_filename") or Path(record.get("csv_path", "")).name,
            "csv_path": record.get("csv_path"),
            "stored_in_database": bool(csv_content),
            "csv_size_chars": len(csv_content),
            "created_at": record.get("created_at"),
        })
    if storage_rows:
        st.dataframe(pd.DataFrame(storage_rows), use_container_width=True, hide_index=True)
    else:
        st.info("目前沒有 ECG 儲存資訊。")

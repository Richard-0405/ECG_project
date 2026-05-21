import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


BACKEND_URL = os.getenv("ECG_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
ROOT = Path(__file__).parent
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def api_get(path):
    try:
        response = requests.get(f"{BACKEND_URL}{path}", timeout=5)
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
    return path


def load_csv_preview(csv_path):
    resolved = resolve_csv_path(csv_path)
    if not resolved.exists():
        return None, resolved
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

st.title("ECG Doctor Admin")
st.caption("醫師後台檢視頁面：使用者資料、心電圖 CSV、分類結果與最近 7 天對話紀錄")

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

records = api_get(f"/users/{selected_user['id']}/records") or []
chat_messages = api_get(f"/users/{selected_user['id']}/chat") or []

profile_tab, ecg_tab, chat_tab, files_tab = st.tabs([
    "個人資料",
    "ECG / 分類",
    "對話紀錄",
    "個人資料夾",
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
        ("過去病史", selected_user.get("medical_history")),
        ("目前位置", selected_user.get("location")),
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
    st.subheader("ECG 與分類紀錄")
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
        col1.metric("最大機率結果", selected_record.get("final_result") or "尚未分類")
        col2.metric("機率", format_probability(selected_record.get("probability")))
        col3.metric("視窗數", selected_record.get("total_windows") or "")

        st.write("類別統計")
        st.json(selected_record.get("counts") or {})

        csv_path = selected_record.get("csv_path")
        if csv_path:
            df, resolved = load_csv_preview(csv_path)
            st.caption(f"CSV：{resolved}")
            if df is None:
                st.warning("找不到 CSV 檔案，可能尚未同步到個人資料夾。")
            else:
                available_leads = [lead for lead in LEADS if lead in df.columns]
                default_leads = ["II"] if "II" in available_leads else available_leads[:1]
                selected_leads = st.multiselect(
                    "選擇顯示導程",
                    available_leads,
                    default=default_leads,
                )
                if selected_leads:
                    st.plotly_chart(plot_ecg(df, selected_leads), use_container_width=True)
                with st.expander("CSV 前 20 筆資料"):
                    st.dataframe(df.head(20), use_container_width=True)

with chat_tab:
    st.subheader("最近 7 天對話紀錄")
    if not chat_messages:
        st.info("此使用者目前沒有最近 7 天對話紀錄。")
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
    st.subheader("個人資料夾")
    data_root = ROOT / "user_data"
    folders = list(data_root.glob(f"user_{selected_user['id']}_*")) if data_root.exists() else []
    if not folders:
        st.info("尚未建立個人資料夾。重新啟動後端或新增資料後會自動同步。")
    else:
        folder = folders[0]
        st.code(str(folder))
        files = [
            {
                "name": path.name,
                "relative_path": str(path.relative_to(folder)),
                "size": path.stat().st_size,
                "modified": path.stat().st_mtime,
            }
            for path in folder.rglob("*")
            if path.is_file()
        ]
        st.dataframe(pd.DataFrame(files), use_container_width=True, hide_index=True)

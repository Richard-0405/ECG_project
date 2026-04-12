import streamlit as st
import requests # 🌟 新增：用來呼叫外部 API 的套件
import datetime
import json
import googlemaps  # 🌟 新增：匯入 Google Maps 套件
import pandas as pd
from PIL import Image
import os
from openai import OpenAI

# streamlit run app.py
# 自動從 secrets.toml 讀取 API Key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
gmaps = googlemaps.Client(key=st.secrets["GOOGLE_MAPS_API_KEY"]) # 🌟 新增：初始化 Google Maps 客戶端

st.set_page_config(page_title="AI 心電圖健康助理", page_icon="🩺", layout="wide")

# ==========================================
# 外部 API 工具 (MCP Tool)
# ==========================================
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

# 定義要告訴 GPT 我們有哪些工具可以使用
tools_definition = [
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
    st.title("⚙️ 控制面板")
    
    st.header("📂 上傳資料")
    uploaded_file = st.file_uploader("請上傳心電圖 (ECG) 影像", type=['png', 'jpg', 'jpeg'])
    
    if st.button("🚀 執行心律不整偵測", use_container_width=True):
        if uploaded_file is not None:
            # 在畫面上顯示使用者上傳的圖片
            img = Image.open(uploaded_file)
            st.image(img, caption="已上傳的心電圖", use_container_width=True)
            
            # ==========================================
            # 步驟 1：將影像轉換為 CSV 並儲存
            # ==========================================
            with st.spinner("🔄 步驟 1/2：正在將心電圖影像轉換為數位訊號 (CSV)..."):
                # 📍 這裡放你的「模型 1」的處理程式碼
                # 假設你的模型會產出一個 DataFrame 或直接存成檔案
                
                # --- 模擬轉換過程 ---
                dummy_data = {"Time": [0.1, 0.2, 0.3], "Voltage": [0.5, 0.8, 0.2]}
                df_signal = pd.DataFrame(dummy_data)
                
                # 儲存 CSV 到本地端 (符合你「並存著」的需求)
                save_path = "temp_ecg_signal.csv"
                df_signal.to_csv(save_path, index=False)
                # ------------------
                
            st.success(f"✅ 影像轉換成功！訊號已儲存至 `{save_path}`")
            
            # ==========================================
            # 步驟 2：讀取 CSV 進行心律不整分類
            # ==========================================
            with st.spinner("🧠 步驟 2/2：正在載入 AI 模型進行心律不整分析..."):
                # 📍 這裡放你的「模型 2」的處理程式碼
                # 讀取剛剛存好的 CSV 檔案，丟進你的分類模型
                
                # --- 模擬預測過程 ---
                # df_to_predict = pd.read_csv(save_path)
                # prediction = your_classification_model.predict(df_to_predict)
                # final_result = "心律不整 (Arrhythmia)" if prediction == 1 else "正常 (Normal)"
                final_result = "心律不整 - 心房顫動 (此為模擬結果)" 
                # ------------------
                
            # 顯示最終結果！
            st.error(f"⚠️ 分析完成！預測結果：**{final_result}**") # 如果是正常可以改用 st.success
            
            # 將結果偷偷告訴 GPT，讓它能跟使用者討論這個狀況
            st.session_state.messages.append({
                "role": "system", 
                "content": f"【系統提示】使用者剛剛上傳了一張心電圖，經過兩階段 AI 模型判定，最終結果為：「{final_result}」。請在接下來的對話中，主動關心這個結果，並提供衛教建議。"
            })
            
        else:
            st.warning("請先上傳 ECG 影像！")
            
    st.divider() 
    
    st.header("👤 個人資訊設定")
    with st.expander("展開設定", expanded=True):
        user_name = st.text_input("姓名", placeholder="例如：王小明", value="使用者")
        age = st.number_input("年齡", min_value=1, max_value=120, value=30)
        gender = st.selectbox("性別", ["男", "女", "其他"])
        medical_history = st.text_input("過去病史", placeholder="例如：高血壓、糖尿病...")
        location = st.text_input("目前位置", value="台南市")

    st.divider()
    
    st.header("📱 家屬通報系統")
    if st.button("📤 傳送今日健康報告給家屬", use_container_width=True):
        # 1. 組合今日的報告內容
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        
        # 這裡可以根據你的系統狀態動態生成報告，這裡先示範基本格式
        report_content = f"""【AI 健康助理 - 每日報告】
📅 日期：{today_str}
👤 姓名：{user_name} (年齡:{age})
📍 位置：{location}

🩺 系統摘要：
今日已完成基本健康評估。心電圖數據尚未上傳（或顯示正常）。使用者有查詢附近的診所與健康食譜。

💡 AI 建議：
建議持續監控血壓，並維持少油少鹽的飲食習慣。

-- 此訊息由 AI 心電圖與健康助理自動發送 --"""

        # 2. 呼叫我們剛剛寫好的 LINE 函數
        with st.spinner("正在發送 LINE 訊息給家屬..."):
            success = send_line_report(report_content)
            
        if success:
            st.success("✅ 報告已成功發送至家屬的 LINE！")
            st.balloons() # 加上一個慶祝的小動畫
        else:
            st.error("❌ 發送失敗，請檢查終端機的錯誤訊息或 API Key 設定。")
# ==========================================
# 主畫面 (Main Area) - 對話框區塊
# ==========================================
st.title("🩺 AI 心電圖與健康助理")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是您的專屬助理。您可以問我健康問題，或請我幫您尋找附近的診所！"}
    ]

for message in st.session_state.messages:
    # 1. 自動判斷資料型態，安全地取得 role 與 content
    role = message["role"] if isinstance(message, dict) else message.role
    content = message.get("content") if isinstance(message, dict) else message.content
    
    # 2. 如果不是系統或工具訊息，而且 content 不是空的，才顯示在畫面上
    if role not in ["system", "tool"] and content:
        with st.chat_message(role):
            st.markdown(content)

if prompt := st.chat_input("請輸入您的問題或需求..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        # 👇 在系統提示詞中加入姓名，並告訴 AI 可以在對話中稱呼使用者
        system_prompt = f"你是一位專業且溫暖的醫療助理。正在與你對話的使用者姓名是「{user_name}」，年齡 {age}，性別 {gender}，病史：{medical_history}，位置：{location}。請依據這些資訊提供協助，並在適當的時候稱呼對方的名字。"
        
        # 準備對話紀錄
        messages_to_send = [{"role": "system", "content": system_prompt}] + st.session_state.messages
        
        # 第一次呼叫 GPT：讓它決定要不要使用工具
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_to_send,
            tools=tools_definition,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        # 檢查 GPT 是否決定呼叫工具 (Function Calling)
        if response_message.tool_calls:
            st.info("🔄 AI 正在呼叫外部工具查詢資料...")
            st.session_state.messages.append(response_message) # 記錄 AI 想呼叫工具的動作
            
            # 執行工具（防彈升級版）
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_response = "系統訊息：未知的工具或執行失敗" # 預設值，保證每次都有東西可以回傳
                
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
                    
                    else:
                        function_response = f"系統訊息：找不到名為 {function_name} 的工具。"
                        
                except Exception as e:
                    function_response = f"系統訊息：解析工具參數時發生錯誤 ({str(e)})"
                
                # 確保【每一個】 tool_call_id 都一定會被加回對話紀錄中！
                st.session_state.messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                })
                    
            
            # 第二次呼叫 GPT：讓它根據工具回傳的資料，整理成人類看的懂的文字
            messages_to_send = [{"role": "system", "content": system_prompt}] + st.session_state.messages
            second_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages_to_send
            )
            bot_reply = second_response.choices[0].message.content
            st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            
        else:
            # 如果 GPT 覺得不需要用工具，就直接回答
            bot_reply = response_message.content
            st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})


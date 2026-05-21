# ECG App 雲端部署流程

本專案建議先使用：

- 前台：Streamlit Community Cloud，入口檔 `app.py`
- 醫師後台：Streamlit Community Cloud，入口檔 `admin_app.py`
- 後端：Render Web Service，入口檔 `backend.py`
- 資料庫：Render PostgreSQL

正式雲端版的主要資料會存在 PostgreSQL。ECG CSV 內容會一起存進資料庫欄位 `csv_content`，所以前台與後台即使部署在不同主機，也能看到同一筆 ECG 波形。

## 1. 推送程式碼到 GitHub

確認本機在專案資料夾：

```powershell
cd C:\Users\user\Desktop\Group5_project\ECG_project-tingyi_0430
git status
git add .
git commit -m "prepare cloud deployment"
git push
```

如果你前面是第一次推送，且 GitHub repo 已經被本機版本覆蓋過，之後一般只需要 `git push`。

## 2. 在 Render 建立 PostgreSQL

1. 打開 Render Dashboard。
2. 選擇 New。
3. 選擇 PostgreSQL。
4. 建議名稱：`ecg-postgres`。
5. Region 選和後端 Web Service 相同區域。
6. 建立完成後，進入資料庫頁面，複製：
   - Internal Database URL：給 Render 後端使用。
   - External Database URL：給本機執行 `migrate_to_cloud.py` 匯入舊資料時使用。

Render PostgreSQL 官方文件：
https://render.com/docs/postgresql

## 3. 在 Render 建立 FastAPI 後端

1. Render Dashboard 選擇 New。
2. 選擇 Web Service。
3. 連接 GitHub repo：`Richard-0405/ECG_project`。
4. Root Directory 留空，除非你的 repo 結構後來有改。
5. Runtime 選 Python。
6. Build Command：

```bash
pip install -r requirements.txt
```

7. Start Command：

```bash
uvicorn backend:app --host 0.0.0.0 --port $PORT
```

8. Environment Variables 新增：

```text
DATABASE_URL=<貼上 Render PostgreSQL 的 Internal Database URL>
ECG_LOCAL_EXPORT_ENABLED=0
ECG_IMPORT_USER_DATA_ON_STARTUP=0
ECG_CHAT_RETENTION_DAYS=7
```

9. Deploy 完成後，打開：

```text
https://你的後端網址.onrender.com/health
```

如果看到類似下面內容，代表後端成功：

```json
{
  "ok": true,
  "database": "postgresql",
  "local_export_enabled": false
}
```

FastAPI on Render 官方文件：
https://render.com/docs/deploy-fastapi

## 4. 匯入目前本機舊資料到雲端資料庫

在本機 PowerShell 或 Miniforge Prompt 執行：

```powershell
cd C:\Users\user\Desktop\Group5_project\ECG_project-tingyi_0430
conda activate ecg_project
$env:DATABASE_URL="貼上 Render PostgreSQL 的 External Database URL"
python migrate_to_cloud.py
```

成功後會顯示匯入的使用者、ECG 紀錄、聊天紀錄數量。

注意：

- 這一步使用 External Database URL。
- Render 後端環境變數使用 Internal Database URL。
- External URL 不要放到 GitHub。

## 5. 部署前台 Streamlit app.py

1. 打開 Streamlit Community Cloud。
2. 選擇 Create app。
3. 選 GitHub repo：`Richard-0405/ECG_project`。
4. Branch 選 `main`。
5. Main file path 填：

```text
app.py
```

6. 在 App settings > Secrets 貼上：

```toml
BACKEND_URL = "https://你的後端網址.onrender.com"

OPENAI_API_KEY = "你的 OpenAI API Key"
GOOGLE_MAPS_API_KEY = "你的 Google Maps API Key"
SPOONACULAR_API_KEY = "你的 Spoonacular API Key"
LINE_CHANNEL_ACCESS_TOKEN = "你的 LINE Channel Access Token"
LINE_TARGET_USER_ID = "家人的 LINE User ID"
```

Streamlit 部署官方文件：
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

Streamlit secrets 官方文件：
https://docs.streamlit.io/develop/concepts/connections/secrets-management

## 6. 部署醫師後台 admin_app.py

同一個 GitHub repo 再建立一個 Streamlit app。

Main file path 填：

```text
admin_app.py
```

Secrets 填：

```toml
BACKEND_URL = "https://你的後端網址.onrender.com"
ADMIN_PASSWORD = "請換成你自己的後台密碼"
```

正式公開前務必設定 `ADMIN_PASSWORD`，否則任何知道網址的人都可能看到後台資料。

## 7. 每次更新程式後

本機修改完成後：

```powershell
git status
git add .
git commit -m "你的更新訊息"
git push
```

Render 和 Streamlit Cloud 會從 GitHub 重新部署。若沒有自動部署，可以到各自後台按 Manual Deploy 或 Reboot。

## 8. 正式公開前檢查

- GitHub 不能有 `.streamlit/secrets.toml`。
- GitHub 不能有真實病患 `user_data/`。
- GitHub 不能有 `ecg_memory.db`。
- Render 後端 `/health` 要顯示 `database: postgresql`。
- 前台登入後新增 ECG 紀錄，後台要能立即看到該使用者與波形。
- 醫師後台一定要設定 `ADMIN_PASSWORD`。

## 9. 長期正式版建議

目前 CSV 內容直接存在 PostgreSQL，對專題展示和小量使用最簡單。若未來使用者很多，建議改成：

- PostgreSQL：使用者、辨識結果、聊天紀錄、檔案 metadata。
- Cloudflare R2 / AWS S3 / Google Cloud Storage：ECG CSV、原始上傳影像、波形圖片。
- 後端只在資料庫保存檔案 URL。

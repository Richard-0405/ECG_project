@echo off
setlocal

cd /d "%~dp0"

set "CONDA_BAT="

for %%P in (
  "%USERPROFILE%\anaconda3\condabin\conda.bat"
  "%USERPROFILE%\miniconda3\condabin\conda.bat"
  "%USERPROFILE%\miniforge3\condabin\conda.bat"
  "%LOCALAPPDATA%\anaconda3\condabin\conda.bat"
  "%LOCALAPPDATA%\miniconda3\condabin\conda.bat"
  "%LOCALAPPDATA%\miniforge3\condabin\conda.bat"
  "C:\ProgramData\anaconda3\condabin\conda.bat"
  "C:\ProgramData\miniconda3\condabin\conda.bat"
  "C:\ProgramData\miniforge3\condabin\conda.bat"
  "%USERPROFILE%\anaconda3\Scripts\activate.bat"
  "%USERPROFILE%\miniconda3\Scripts\activate.bat"
  "%USERPROFILE%\miniforge3\Scripts\activate.bat"
  "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat"
  "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat"
  "%LOCALAPPDATA%\miniforge3\Scripts\activate.bat"
  "C:\ProgramData\anaconda3\Scripts\activate.bat"
  "C:\ProgramData\miniconda3\Scripts\activate.bat"
  "C:\ProgramData\miniforge3\Scripts\activate.bat"
) do (
  if exist "%%~P" set "CONDA_BAT=%%~P"
)

if "%CONDA_BAT%"=="" (
  echo Cannot find conda.bat or activate.bat.
  echo Please edit start_admin.bat and set CONDA_BAT to your conda path.
  pause
  exit /b 1
)

for %%P in ("%CONDA_BAT%") do set "CONDA_BAT_SHORT=%%~sP"

start "ECG Backend" cmd /k "call %CONDA_BAT_SHORT% activate ecg_project && uvicorn backend:app --reload --host 127.0.0.1 --port 8000"
timeout /t 2 /nobreak >nul
start "ECG Doctor Admin" cmd /k "call %CONDA_BAT_SHORT% activate ecg_project && streamlit run admin_app.py --server.port 8502"

endlocal

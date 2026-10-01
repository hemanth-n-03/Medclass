@echo off
echo ========================================================
echo        Starting MedClass AI Complete Application
echo ========================================================

cd /d "%~dp0\.."

echo [1/2] Launching FastAPI Backend on http://127.0.0.1:8000 ...
start "MedClass AI Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"

echo [2/2] Launching Next.js Frontend on http://localhost:3000 ...
start "MedClass AI Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================================
echo  MedClass AI services launched!
echo  - Frontend Web UI:    http://localhost:3000
echo  - Backend API:        http://127.0.0.1:8000/api
echo  - Interactive Docs:   http://127.0.0.1:8000/docs
echo ========================================================

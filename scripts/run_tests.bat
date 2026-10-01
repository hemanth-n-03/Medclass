@echo off
echo ========================================================
echo        Running MedClass AI Automated Test Suite
echo ========================================================

cd /d "%~dp0\.."

echo.
echo [1/3] Running Machine Learning Pipeline Unit Tests...
.venv\Scripts\python.exe -m pytest ml\tests\test_ml_pipeline.py -v
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ML Pipeline tests failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Running Backend API Integration Tests...
.venv\Scripts\python.exe -m pytest backend\tests\test_api.py -v
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Backend API tests failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [3/3] Validating Frontend Production Build...
cd frontend
call npm run build
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Frontend build validation failed!
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================================
echo  All MedClass AI automated tests passed successfully!
echo ========================================================

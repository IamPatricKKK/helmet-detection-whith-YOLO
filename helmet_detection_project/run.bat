@echo off
echo ========================================
echo    HELMET DETECTION PROJECT
echo ========================================
echo.

echo Chon che do chay:
echo 1. Full Pipeline (Thu thap + Train + Inference)
echo 2. Quick Start (Chi Inference)
echo 3. Data Collection Only
echo 4. Training Only
echo 5. Install Dependencies
echo.

set /p choice="Nhap lua chon (1-5): "

if "%choice%"=="1" (
    echo Dang chay Full Pipeline...
    python main.py --mode full --training-mode quick --epochs 20
) else if "%choice%"=="2" (
    echo Dang chay Quick Start...
    python main.py --mode quick
) else if "%choice%"=="3" (
    echo Dang chay Data Collection...
    python main.py --mode data
) else if "%choice%"=="4" (
    echo Dang chay Training...
    python main.py --mode train --training-mode quick --epochs 20
) else if "%choice%"=="5" (
    echo Dang cai dat dependencies...
    pip install -r requirements.txt
) else (
    echo Lua chon khong hop le!
)

echo.
echo Nhan phim bat ky de thoat...
pause > nul



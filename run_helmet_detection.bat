@echo off
echo ========================================
echo    HEAD DETECTION + HELMET DETECTION
echo ========================================
echo.

echo Chon che do:
echo 1. Chay ung dung nhan dien mat + mu bao hiem
echo 2. Train model mu bao hiem
echo 3. Chay ung dung cu (khong co mu bao hiem)
echo.

set /p choice="Nhap lua chon (1-3): "

if "%choice%"=="1" (
    echo Dang chay ung dung nhan dien mat + mu bao hiem...
    python apps/head_detection_app_with_helmet.py
) else if "%choice%"=="2" (
    echo Dang train model mu bao hiem...
    cd helmet_detection_project
    python training/train_model.py --mode quick --epochs 20
) else if "%choice%"=="3" (
    echo Dang chay ung dung cu...
    python apps/head_detection_app.py
) else (
    echo Lua chon khong hop le!
)

echo.
echo Nhan phim bat ky de thoat...
pause > nul

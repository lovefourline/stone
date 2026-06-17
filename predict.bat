@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo   RockClass Predict — 岩石图片预测
set KMP_DUPLICATE_LIB_OK=TRUE

python predict.py %*

pause

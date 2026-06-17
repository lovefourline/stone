@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: 一键启动岩石分类模型训练
:: 自动处理 OpenMP DLL 冲突

echo.
echo   ╔══════════════════════════════════════════════╗
echo   ║   RockClass V7 — 岩石薄片图像三分类训练      ║
echo   ╚══════════════════════════════════════════════╝
echo.

set KMP_DUPLICATE_LIB_OK=TRUE

:: 尝试用 conda 环境（如果存在）
where conda >nul 2>&1
if %errorlevel%==0 (
    call conda activate base 2>nul
)

:: 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [错误] 未找到 Python！请先安装 Python 3.10+
    pause
    exit /b 1
)

:: 检查依赖
python -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo   [提示] PyTorch 未安装，正在安装...
    echo.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install tqdm matplotlib pillow numpy pandas
)

echo   开始训练...
echo.

python train.py

echo.
echo   训练完成！
pause

@echo off
echo 正在启动身份证扫描件生成器...
python main.py
if errorlevel 1 (
    echo.
    echo [错误] 程序启动失败
    echo 请确保已安装所有依赖: pip install -r requirements.txt
    pause
)
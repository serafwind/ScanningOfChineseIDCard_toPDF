@echo off
echo ====================================
echo 身份证扫描件生成器 - 打包脚本
echo ====================================
echo.

echo 正在检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.7或更高版本
    pause
    exit /b 1
)

echo 正在安装依赖...
pip install -r requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)

echo.
echo 正在安装PyInstaller...
pip install pyinstaller
if errorlevel 1 (
    echo [错误] PyInstaller安装失败
    pause
    exit /b 1
)

echo.
echo 开始打包...
pyinstaller --onefile --windowed --name "身份证扫描件生成器" --icon=NONE main.py

if errorlevel 1 (
    echo [错误] 打包失败
    pause
    exit /b 1
)

echo.
echo ====================================
echo 打包完成！
echo 可执行文件位于: dist\身份证扫描件生成器.exe
echo ====================================
pause
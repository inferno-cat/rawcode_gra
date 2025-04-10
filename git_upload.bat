@echo off
:: 切换到脚本所在目录（确保路径正确）
cd /d %~dp0

:: 检查是否在 Git 仓库中
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo 错误：当前目录不是 Git 仓库。
    pause
    exit /b 1
)

:: 显示 Git 状态
echo 当前 Git 状态：
git status

:: 交互式选择操作
:menu
echo.
echo 请选择操作：
echo 1. 添加所有更改并提交
echo 2. 仅提交已暂存的更改
echo 3. 退出
set /p choice="输入选项 (1/2/3): "

if "%choice%"=="1" (
    git add .
    echo 已添加所有更改到暂存区。
) else if "%choice%"=="2" (
    echo 跳过添加步骤，仅提交已暂存的更改。
) else if "%choice%"=="3" (
    exit /b 0
) else (
    echo 无效选项，请重新输入。
    goto menu
)

:: 输入提交消息
:commit
set /p commit_msg="请输入提交消息（必填）: "
if "%commit_msg%"=="" (
    echo 提交消息不能为空！
    goto commit
)

:: 执行提交
git commit -m "%commit_msg%"
if errorlevel 1 (
    echo 提交失败，请检查错误。
    pause
    exit /b 1
)

:: 推送到远程仓库
echo 正在推送到远程仓库...
git push
if errorlevel 1 (
    echo 推送失败，请检查错误。
    pause
    exit /b 1
)

echo 操作完成！
pause
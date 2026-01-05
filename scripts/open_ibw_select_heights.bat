@echo off
set SCRIPT_DIR=%~dp0
python "%SCRIPT_DIR%launch_select_heights.py" "%~1"

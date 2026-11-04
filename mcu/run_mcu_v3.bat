@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
echo === full cue controller on the emulated Cortex-M4F ===
python run_mcu.py > run_log3.txt 2>&1
type run_log3.txt
echo === done ===
pause

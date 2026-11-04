@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
echo === installing unicorn and capstone from PyPI ===
python -m pip install unicorn capstone
echo === running Freeze Index on the emulated Cortex-M4F ===
python run_mcu.py > run_log.txt 2>&1
type run_log.txt
echo === done ===
pause

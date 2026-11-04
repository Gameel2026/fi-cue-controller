@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
echo === installing unicorn and capstone from PyPI (slow connection settings) ===
python -m pip install --timeout 120 --retries 10 unicorn capstone
python -c "import unicorn" || (echo UNICORN NOT INSTALLED > run_log.txt & echo === install failed, run again === & pause & exit /b 1)
echo === running Freeze Index on the emulated Cortex-M4F ===
python run_mcu.py > run_log.txt 2>&1
type run_log.txt
echo === done ===
pause

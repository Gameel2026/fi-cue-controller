@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
echo === s16: latency 4 s vs 2 s window, development data ===
python s16_latency_window.py > s16_log.txt 2>&1
type s16_log.txt
echo === done ===
pause

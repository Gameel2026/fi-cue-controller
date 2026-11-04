@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
echo === s15: per-patient and latency analysis ===
python s15_patient_latency.py > s15_log.txt 2>&1
type s15_log.txt
echo === done ===
pause

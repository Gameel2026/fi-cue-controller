@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
echo === s19: cue precision, AP and gate during akinetic FoG ===
python s19_precision_gate_ap.py > s19_log.txt 2>&1
type s19_log.txt
echo === done ===
pause

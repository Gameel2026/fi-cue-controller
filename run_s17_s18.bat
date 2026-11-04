@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
echo === s17: home smoothing and latency ===
python s17_home_smoothing.py > s17_log.txt 2>&1
type s17_log.txt
echo === s18: nested hyperparameter tuning (about 1 hour) ===
python s18_nested_tuning.py > s18_log.txt 2>&1
type s18_log.txt
echo === done ===
pause

@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
echo === s20: cue precision vs chance (B1) and frozen CNN on the contest test (B2) ===
python s20_precision_chance_cnn_test.py > s20_log.txt 2>&1
type s20_log.txt
echo === done ===
pause

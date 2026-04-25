@echo off
echo [1/3] Creating Virtual Environment...
python -m venv venv
call venv\Scripts\activate

echo [2/3] Upgrading Pip...
python -m pip install --upgrade pip

echo [3/3] Installing Dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo Setup Complete! 
echo Remember to install FFmpeg and add it to your PATH for MP3 support.
echo To activate the environment later, run: call venv\Scripts\activate
pause
@echo off
echo ========================================
echo  AI Cyclone Early Warning System Setup
echo ========================================
echo.
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
echo.
echo Installing requirements...
pip install -r requirements.txt
echo.
echo Creating project directories...
mkdir data\satellite 2>nul
mkdir data\ibtracs 2>nul
mkdir data\processed 2>nul
mkdir models 2>nul
mkdir uploads 2>nul
mkdir results\plots 2>nul
mkdir results\metrics 2>nul
mkdir results\predictions 2>nul
echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Download satellite dataset from Kaggle and place in data\satellite\
echo 2. Download IBTrACS CSV and place in data\ibtracs\
echo 3. Run: python train_all.py (optional - works in demo mode without training)
echo 4. Run: run.bat to start the application
echo.
pause

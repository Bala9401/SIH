@echo off
echo Setting up AI Cyclone Early Warning System...

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Creating necessary directories...
mkdir data\processed 2>nul
mkdir models 2>nul
mkdir results\plots 2>nul
mkdir results\metrics 2>nul
mkdir results\predictions 2>nul
mkdir uploads 2>nul

echo Setup completed successfully!

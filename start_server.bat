@echo off
echo Starting Combined Server on port 8000...
echo.
echo Make sure you have activated your virtual environment and installed dependencies:
echo   venv\Scripts\activate
echo   pip install -r requirements.txt
echo.
python -m uvicorn combined_server:app --host 0.0.0.0 --port 8000





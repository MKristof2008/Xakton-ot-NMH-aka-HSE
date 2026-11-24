@echo off
rem Запуск сервера из любой папки: скрипт сам находит свой каталог.
cd /d "%~dp0backend"
if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
) else (
  py -3 -m uvicorn main:app --host 127.0.0.1 --port 8000
)

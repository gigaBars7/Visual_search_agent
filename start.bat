@echo off
cd /d "%~dp0"

set "PHOTO_ROOT="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "[Environment]::GetFolderPath([Environment+SpecialFolder]::MyPictures)"`) do set "PHOTO_ROOT=%%I"

if not defined PHOTO_ROOT (
    echo Could not determine the pictures folder.
    exit /b 1
)

echo Photo root: %PHOTO_ROOT%

docker compose up -d agent api
docker compose run --rm cli
docker compose down

@echo off
cd /d "%~dp0"

docker compose up -d agent api
docker compose run --rm cli
docker compose down

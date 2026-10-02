@echo off
setlocal
rem ------------------------------------------------------------
rem Llama Studio (Tauri) launcher.
rem Cargo builds incrementally; running it every time also ensures frontend
rem changes embedded by Tauri are present in the executable.
rem  - the old Electron GUI launcher is preserved as start-v1.cmd
rem ------------------------------------------------------------
cd /d "%~dp0app\src-tauri"
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

echo Building Llama Studio (incremental when up to date)...
powershell -NoProfile -Command "$f = Get-Item -LiteralPath 'build.rs'; $f.LastWriteTime = [DateTime]::Now"
if errorlevel 1 (
  echo Could not refresh the embedded frontend build timestamp.
  pause
  exit /b 1
)
cargo build
if errorlevel 1 (
  echo.
  echo Build failed. Check the output above.
  pause
  exit /b 1
)

start "" "target\debug\llama-studio.exe"
endlocal

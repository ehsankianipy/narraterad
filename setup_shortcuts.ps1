# setup_shortcuts.ps1 — NarrateRad
# Creates two desktop shortcuts for Sadia:
#   1. NarrateRad — launches the app
#   2. Update NarrateRad — pulls latest from GitHub
# Run once from PowerShell:
#   powershell -ExecutionPolicy Bypass -File setup_shortcuts.ps1

$NARRATERAD_DIR = "$env:USERPROFILE\narraterad"

# ── Find the desktop (handles OneDrive desktop too) ───────────────────────────
$DESKTOP = $null
$possible = @(
    "$env:USERPROFILE\Desktop",
    "$env:USERPROFILE\OneDrive\Desktop",
    "$env:USERPROFILE\OneDrive - Personal\Desktop",
    [System.Environment]::GetFolderPath("Desktop")
)
foreach ($path in $possible) {
    if (Test-Path $path) { $DESKTOP = $path; break }
}
if (-not $DESKTOP) {
    Write-Host "Could not find Desktop folder. Creating shortcuts in narraterad folder instead." -ForegroundColor Yellow
    $DESKTOP = $NARRATERAD_DIR
}

Write-Host ""
Write-Host "Setting up NarrateRad desktop shortcuts..." -ForegroundColor Cyan
Write-Host "Desktop location: $DESKTOP"

# ── Create launcher batch file ────────────────────────────────────────────────
$launcherContent = @"
@echo off
title NarrateRad
cd /d "%USERPROFILE%\narraterad"

echo.
echo  Starting NarrateRad...
echo  Please wait...
echo.

REM Start Ollama in background
start "" /B ollama serve
timeout /t 2 /nobreak >nul

REM Start the web server
start /B .venv\Scripts\uvicorn main:app --port 8000 --ws-ping-interval 20 --ws-ping-timeout 60
timeout /t 4 /nobreak >nul

REM Open Chrome
start chrome http://localhost:8000 2>nul
if errorlevel 1 start msedge http://localhost:8000 2>nul
if errorlevel 1 start http://localhost:8000

echo.
echo  NarrateRad is running at http://localhost:8000
echo.
echo  DO NOT close this window while using the app.
echo  Press any key to STOP NarrateRad.
echo.
pause >nul

REM Stop the server
taskkill /F /IM uvicorn.exe >nul 2>&1
echo NarrateRad stopped.
timeout /t 2 /nobreak >nul
"@

$launcherPath = "$NARRATERAD_DIR\NarrateRad.bat"
$launcherContent | Out-File -FilePath $launcherPath -Encoding ASCII
Write-Host "Launcher created" -ForegroundColor Green

# ── Create updater batch file ─────────────────────────────────────────────────
$updaterContent = @"
@echo off
title NarrateRad Updater
echo.
echo  Updating NarrateRad...
echo  Please wait...
echo.

cd /d "%USERPROFILE%\narraterad"
git pull

echo.
echo  Update complete!
echo  Close this window and relaunch NarrateRad to use the latest version.
echo.
pause
"@

$updaterPath = "$NARRATERAD_DIR\Update NarrateRad.bat"
$updaterContent | Out-File -FilePath $updaterPath -Encoding ASCII
Write-Host "Updater created" -ForegroundColor Green

# ── Place shortcuts on desktop ────────────────────────────────────────────────
$WshShell = New-Object -ComObject WScript.Shell

# Launcher shortcut
$launcherShortcut = $WshShell.CreateShortcut("$DESKTOP\NarrateRad.lnk")
$launcherShortcut.TargetPath = $launcherPath
$launcherShortcut.WorkingDirectory = $NARRATERAD_DIR
$launcherShortcut.Description = "Launch NarrateRad"
$launcherShortcut.Save()
Write-Host "NarrateRad icon added to Desktop" -ForegroundColor Green

# Updater shortcut
$updaterShortcut = $WshShell.CreateShortcut("$DESKTOP\Update NarrateRad.lnk")
$updaterShortcut.TargetPath = $updaterPath
$updaterShortcut.WorkingDirectory = $NARRATERAD_DIR
$updaterShortcut.Description = "Pull latest NarrateRad updates from GitHub"
$updaterShortcut.Save()
Write-Host "Update NarrateRad icon added to Desktop" -ForegroundColor Green

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Done! Two icons are now on your Desktop:" -ForegroundColor Green
Write-Host "  - NarrateRad          → double-click to launch the app" -ForegroundColor White
Write-Host "  - Update NarrateRad   → double-click when Ehsan tells you to update" -ForegroundColor White
Write-Host ""
Read-Host "  Press Enter to exit"

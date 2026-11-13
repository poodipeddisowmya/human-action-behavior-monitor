# Keep the monitor running at http://localhost:8000, restarting it if it stops.
#
# Started automatically at Windows sign-in by the "Behavior Monitor" shortcut
# in the user's Startup folder (see scripts\install_autostart.ps1), or run it
# by hand:
#     powershell -ExecutionPolicy Bypass -File scripts\keep_running.ps1
#
# To stop it for good: run scripts\uninstall_autostart.ps1, then end the
# "python" processes in Task Manager (or sign out and back in).

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
$console = Join-Path $root "logs\server-console.log"

# Only one copy may run: a second one would fight over port 8000 and the camera.
$mutex = New-Object System.Threading.Mutex($false, "Global\HumanActionBehaviorMonitorKeepRunning")
if (-not $mutex.WaitOne(0)) { exit 0 }

while ($true) {
    $listening = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if (-not $listening) {
        "$(Get-Date -Format s) starting monitor" | Out-File -FilePath $console -Encoding utf8
        $env:PYTHONUNBUFFERED = "1"
        # Runs until it exits; output replaces the previous run's console log.
        & $python "scripts\run_pipeline.py" *>> $console
        "$(Get-Date -Format s) monitor exited with code $LASTEXITCODE; restarting in 5 s" |
            Out-File -FilePath $console -Encoding utf8 -Append
    }
    Start-Sleep -Seconds 5
}

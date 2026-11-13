# Start the monitor automatically whenever this Windows user signs in.
# Adds one shortcut to the user's Startup folder; nothing system-wide changes.
# Undo with scripts\uninstall_autostart.ps1.

$root = Split-Path -Parent $PSScriptRoot
$startup = [Environment]::GetFolderPath("Startup")
$link = Join-Path $startup "Behavior Monitor.lnk"

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($link)
$sc.TargetPath = "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe"
$sc.Arguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$root\scripts\keep_running.ps1`""
$sc.WorkingDirectory = $root
$sc.WindowStyle = 7          # minimised
$sc.Description = "Keeps the Human Action & Behavior Monitor running at http://localhost:8000"
$sc.Save()
"Installed: $link"

# Stop the monitor from starting at sign-in (removes the Startup shortcut).
$link = Join-Path ([Environment]::GetFolderPath("Startup")) "Behavior Monitor.lnk"
if (Test-Path $link) { Remove-Item $link; "Removed: $link" } else { "Not installed." }

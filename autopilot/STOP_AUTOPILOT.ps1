# Остановка фонового автопилота BAYKAM
#
#   .\STOP_AUTOPILOT.ps1
#
# Автопилот не открывает портов, поэтому останавливаем по командной строке
# процесса: ищем python с аргументами "-m autopilot run".

[CmdletBinding()]
param()

$ErrorActionPreference = "SilentlyContinue"

$targets = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -match "^(python|py)(\.exe)?$" -and
        $_.CommandLine -match "autopilot" -and
        $_.CommandLine -match "\brun\b"
    }

if (-not $targets) {
    Write-Host "Автопилот не запущен." -ForegroundColor Yellow
    exit 0
}

foreach ($proc in $targets) {
    Write-Host ("Останавливаю PID {0}: {1}" -f $proc.ProcessId, $proc.CommandLine)
    Stop-Process -Id $proc.ProcessId -Force
}

Write-Host "Готово." -ForegroundColor Green

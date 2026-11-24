# Автопилот BAYKAM — запуск 24/7 в PowerShell
#
#   .\RUN_AUTOPILOT.ps1                 # проверить зависимости и запустить цикл
#   .\RUN_AUTOPILOT.ps1 -Once           # один цикл и выход
#   .\RUN_AUTOPILOT.ps1 -NoLlm          # без вызовов модели
#   .\RUN_AUTOPILOT.ps1 -SkipInstall    # не проверять/ставить пакеты
#   .\RUN_AUTOPILOT.ps1 -Status         # только сводка состояния
#
# Логи: autopilot\data\logs\autopilot.log

[CmdletBinding()]
param(
    [switch]$Once,
    [switch]$NoLlm,
    [switch]$SkipInstall,
    [switch]$Status,
    [string]$Config = ""
)

$ErrorActionPreference = "Stop"
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch { }
$env:PYTHONIOENCODING = "utf-8"
$AutopilotDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $AutopilotDir
Set-Location $RepoDir

function Write-Step($text) { Write-Host "==> $text" -ForegroundColor Cyan }

# --- Python ---------------------------------------------------------------
$python = (Get-Command py -ErrorAction SilentlyContinue)
if ($python) {
    $pyArgs = @("-3")
    $pyExe = "py"
} else {
    $python = (Get-Command python -ErrorAction SilentlyContinue)
    if (-not $python) { throw "Python не найден. Установите Python 3.10+ и повторите." }
    $pyArgs = @()
    $pyExe = "python"
}

$version = & $pyExe @pyArgs -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
Write-Step "Python $version"

# --- зависимости ----------------------------------------------------------
if (-not $SkipInstall) {
    Write-Step "проверяю зависимости"
    & $pyExe @pyArgs -m pip install -q -r autopilot/requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Не удалось установить зависимости из autopilot/requirements.txt" }

    $hasChromium = Test-Path (Join-Path $env:LOCALAPPDATA "ms-playwright")
    if (-not $hasChromium) {
        Write-Step "ставлю браузер для Playwright (страницы с JavaScript)"
        & $pyExe @pyArgs -m playwright install chromium
    }
}

# --- ключ модели ----------------------------------------------------------
$envFile = Join-Path $RepoDir "ml\.env"
if (Test-Path $envFile) {
    Write-Step "ключ модели: ml\.env"
} elseif ($env:LLM_API_KEY) {
    Write-Step "ключ модели: переменная окружения LLM_API_KEY"
} else {
    Write-Host "LLM_API_KEY не найден: арбитр отключится, проверка пойдёт по базе знаний." -ForegroundColor Yellow
    $NoLlm = $true
}

# --- запуск ---------------------------------------------------------------
if ($Status) {
    $statusArgs = @("-m", "autopilot")
    if ($Config) { $statusArgs += @("--config", $Config) }
    $statusArgs += "status"
    & $pyExe @pyArgs @statusArgs
    exit $LASTEXITCODE
}

$cmdArgs = @("-m", "autopilot")
if ($Config) { $cmdArgs += @("--config", $Config) }
$cmdArgs += "run"
if ($Once) { $cmdArgs += "--once" }
if ($NoLlm) { $cmdArgs += "--no-llm" }

Write-Step ("запускаю: {0} {1} {2}" -f $pyExe, ($pyArgs -join " "), ($cmdArgs -join " "))
& $pyExe @pyArgs @cmdArgs
exit $LASTEXITCODE

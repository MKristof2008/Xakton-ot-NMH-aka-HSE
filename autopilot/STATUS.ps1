# Сводка состояния автопилота BAYKAM
#
#   .\autopilot\STATUS.ps1
#   .\autopilot\STATUS.ps1 -Json
#   .\autopilot\STATUS.ps1 -History 5
#
# Показывает: собрано материалов, разметка fake/real/uncertain,
# активную модель с метриками, последнее дообучение и состояние источников.

[CmdletBinding()]
param(
    [switch]$Json,
    [int]$History = 0,
    [string]$Config = ""
)

$ErrorActionPreference = "Stop"
$AutopilotDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Split-Path -Parent $AutopilotDir
Set-Location $RepoDir

# кириллица в консоли Windows: иначе вылетает 'charmap' codec error
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch { }
$env:PYTHONIOENCODING = "utf-8"

$pyArgs = @()
$pyExe = "py"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    $pyExe = "python"
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "Python не найден."
    }
} else {
    $pyArgs = @("-3")
}

$cmdArgs = @("-m", "autopilot")
if ($Config) { $cmdArgs += @("--config", $Config) }
$cmdArgs += @("status")
if ($Json) { $cmdArgs += "--json" }
if ($History -gt 0) { $cmdArgs += @("--history", "$History") }

& $pyExe @pyArgs @cmdArgs
exit $LASTEXITCODE

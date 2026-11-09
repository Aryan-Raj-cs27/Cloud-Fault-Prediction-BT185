# Windows equivalent of the Makefile:  .\make.ps1 <target>
param([Parameter(Position = 0)][string]$Target = "help")
$ErrorActionPreference = "Stop"
$Py = ".venv\Scripts\python.exe"

function Run([string[]]$cmd) {
    Write-Host ">> $($cmd -join ' ')" -ForegroundColor Cyan
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "failed: $($cmd -join ' ')" }
}

switch ($Target) {
    "setup" {
        Run @("uv", "venv", ".venv", "--python", "3.11")
        Run @("uv", "pip", "install", "--python", $Py, "torch", "--index-url", "https://download.pytorch.org/whl/cu128")
        Run @("uv", "pip", "install", "--python", $Py, "-r", "requirements.txt")
        Run @("uv", "pip", "install", "--python", $Py, "-r", "requirements-dev.txt")
        Run @("uv", "pip", "install", "--python", $Py, "-e", ".[dev]", "--no-deps")
        Run @($Py, "-m", "pre_commit", "install")
    }
    "data"         { Run @($Py, "scripts/download_data.py"); Run @($Py, "scripts/make_dataset.py") }
    "tune"         { Run @($Py, "scripts/tune.py") }
    "train"        { Run @($Py, "scripts/train.py", "--skip-existing") }
    "progress"     { Run @($Py, "scripts/progress.py") }
    "eval"         { Run @($Py, "scripts/evaluate.py") }
    "simulate"     { Run @($Py, "scripts/run_simulation.py") }
    "demo-samples" { Run @($Py, "scripts/make_demo_samples.py") }
    "reproduce" {
        foreach ($t in "data", "tune", "train", "eval", "simulate", "demo-samples") { & $PSCommandPath $t }
    }
    "lint" {
        Run @($Py, "-m", "ruff", "check", "src", "scripts", "app", "tests")
        Run @($Py, "-m", "ruff", "format", "--check", "src", "scripts", "app", "tests")
    }
    "test"  { Run @($Py, "-m", "pytest", "-m", "not slow", "--cov", "--cov-fail-under=80") }
    "check" { & $PSCommandPath lint; Run @($Py, "-m", "mypy", "src", "app"); & $PSCommandPath test }
    "api"   { Run @($Py, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8000") }
    "build-web" { Push-Location frontend; try { Run @("npm", "ci"); Run @("npm", "run", "build") } finally { Pop-Location } }
    "serve" { & $PSCommandPath build-web; Run @($Py, "-m", "uvicorn", "app.api:server", "--host", "0.0.0.0", "--port", "8000") }
    "user"  { Run @($Py, "scripts/create_user.py", "--username", "admin") }
    "package" { Run @($Py, "scripts/package_artifacts.py") }
    "frontend" { Push-Location frontend; try { Run @("npm", "install"); Run @("npm", "run", "dev") } finally { Pop-Location } }
    default {
        Write-Host "targets: setup data tune train progress eval simulate demo-samples reproduce lint test check api frontend build-web serve user package"
    }
}

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Running pytest..."
python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    throw "pytest failed with exit code $LASTEXITCODE"
}

Write-Host "Validating Docker Compose config..."
docker compose config | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "docker compose config failed with exit code $LASTEXITCODE"
}

Write-Host "Validation completed."

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $root "vietnam-history-chatbot\.env"
if (-not (Test-Path $envFile)) {
  Write-Host "Missing vietnam-history-chatbot\.env. Copy .env.example to .env and fill 9ROUTER_API_KEY first." -ForegroundColor Yellow
}
$env:RAG_RELEASE_PROFILE_ONLY = "true"
$env:RAG_DATA_PROFILE = "cloud_primary_final"
$env:RAG_API_DATA_PROFILE = "cloud_primary_final"
$env:RAG_API_RETRIEVAL_PROVIDER = "local"
Set-Location $root
python scripts/web-demo/persistent-rag-runtime-service.py


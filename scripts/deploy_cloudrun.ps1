<#
Redeploy Profectus AI to Cloud Run from source and print the service URL.

Usage (PowerShell):
  .\scripts\deploy_cloudrun.ps1

  # Override defaults
  .\scripts\deploy_cloudrun.ps1 -ProjectId "profectus-customer-service-adk" -Region "europe-west4"

  # Use a different service name or auth mode
  .\scripts\deploy_cloudrun.ps1 -Service "profectus-ai" -Public $true

Quick setup (copy/paste in pwsh):
  $env:PROJECT_ID = "profectus-customer-service-adk"
  $env:REGION = "europe-west4"
  $env:SERVICE_NAME = "profectus-agent"
  $env:GOOGLE_API_KEY = "AIz..."  # for local testing
  # Or set persistently:
  setx GOOGLE_API_KEY "AIz..."
  gcloud config set project profectus-customer-service-adk
  gcloud config set run/region europe-west4

Override defaults (optional):
  $env:SESSION_STORE = "firestore"    # firestore|memory
  $env:HISTORY_LIMIT = 12
  $env:HISTORY_MAX_CHARS = 4000

Notes:
  - Requires gcloud installed and logged in (gcloud auth login).
  - Uses gcloud config values when ProjectId/Region are not provided.
  - Expects the Secret Manager secret GOOGLE_API_KEY to already exist.
  - Deploys from the repo root using the Dockerfile.
#>

[CmdletBinding()]
param(
    [string]$ProjectId = "",
    [string]$Region = "",
    [string]$Service = "profectus-agent",
    [string]$ServiceAccount = "",
    [string]$SecretName = "GOOGLE_API_KEY",
    [string]$SecretVersion = "latest",
    [string]$SessionStore = "firestore",
    [int]$HistoryLimit = 12,
    [int]$HistoryMaxChars = 4000,
    [int]$Cpu = 4,
    [string]$Memory = "8Gi",
    [int]$Concurrency = 1,
    [int]$MinInstances = 1,
    [bool]$Public = $true,
    [bool]$SmokeTest = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-GcloudConfigValue([string]$Key) {
    try {
        $value = & gcloud config get-value $Key 2>$null
        return ($value -as [string]).Trim()
    } catch {
        return ""
    }
}

if (-not $ProjectId) {
    $ProjectId = $env:PROJECT_ID
}
if (-not $ProjectId) {
    $ProjectId = $env:GOOGLE_CLOUD_PROJECT
}
if (-not $ProjectId) {
    $ProjectId = Get-GcloudConfigValue "project"
}
if (-not $Region) {
    $Region = $env:REGION
}
if (-not $Region) {
    $Region = $env:CLOUD_RUN_REGION
}
if (-not $Region) {
    $Region = Get-GcloudConfigValue "run/region"
}
if (-not $Service -or $Service.Trim().Length -eq 0) {
    if ($env:SERVICE_NAME) {
        $Service = $env:SERVICE_NAME
    }
}

if (-not $ProjectId) {
    throw "ProjectId not set. Pass -ProjectId or run 'gcloud config set project ...'."
}
if (-not $Region) {
    throw "Region not set. Pass -Region or run 'gcloud config set run/region ...'."
}
if (-not $ServiceAccount) {
    $ServiceAccount = "profectus-run-sa@$ProjectId.iam.gserviceaccount.com"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# Check if secret exists
Write-Host "Checking if secret '$SecretName' exists..."
try {
    & gcloud secrets describe $SecretName --project $ProjectId > $null 2>&1
} catch {
    Write-Host "ERROR: Secret '$SecretName' does not exist in project '$ProjectId'."
    Write-Host ""
    Write-Host "Create it with one of these commands:"
    Write-Host "  echo 'your-api-key-here' | gcloud secrets create $SecretName --data-file=- --project $ProjectId"
    Write-Host "  gcloud secrets create $SecretName --data-file='path-to-key.txt' --project $ProjectId"
    Write-Host ""
    Write-Host "Then grant access to the service account:"
    Write-Host "  gcloud secrets add-iam-policy-binding $SecretName --member='serviceAccount:$ServiceAccount' --role='roles/secretmanager.secretAccessor' --project $ProjectId"
    exit 1
}

$envVars = @(
    "PROFECTUS_SESSION_STORE=$SessionStore",
    "PROFECTUS_SESSION_HISTORY_LIMIT=$HistoryLimit",
    "PROFECTUS_HISTORY_MAX_CHARS=$HistoryMaxChars",
    "GOOGLE_CLOUD_PROJECT=$ProjectId"
) -join ","

$authFlag = if ($Public) { "--allow-unauthenticated" } else { "--no-allow-unauthenticated" }

# Build secrets argument (avoid PowerShell variable parsing issues with colons)
$secretsArg = "$SecretName=$SecretName" + ":" + "$SecretVersion"

Write-Host "Deploying to Cloud Run..."
Write-Host "Project: $ProjectId"
Write-Host "Region:  $Region"
Write-Host "Service: $Service"
Write-Host "CPU:     $Cpu"
Write-Host "Memory:  $Memory"
Write-Host "Conc.:   $Concurrency"
Write-Host "Min:     $MinInstances"

& gcloud run deploy $Service `
    --source . `
    --region $Region `
    $authFlag `
    --service-account $ServiceAccount `
    --update-secrets $secretsArg `
    --set-env-vars $envVars `
    --cpu $Cpu `
    --memory $Memory `
    --concurrency $Concurrency `
    --min-instances $MinInstances `
    --timeout 3600

$url = & gcloud run services describe $Service --region $Region --format="value(status.url)"
Write-Host "Service URL: $url"

if ($SmokeTest) {
    Write-Host "Health check: $url/health"
    try {
        $resp = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing
        Write-Host "Health response: $($resp.Content)"
    } catch {
        Write-Host "Health check failed: $($_.Exception.Message)"
    }
}

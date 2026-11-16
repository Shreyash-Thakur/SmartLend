# Ship the CURRENT code to the running demo: rebuild the Docker image, push
# it to ECR, and replace the EC2 instance so first-boot pulls the new image.
# The Elastic IP moves to the replacement, so the URL does not change.
#
# Use after committing app changes; the stack must be up (run demo-up.bat
# first if it is stopped or not deployed).
#
# Run:  demo-publish.bat at the repo root.

$ErrorActionPreference = "Continue"
$Region = "ap-south-1"
$RepoName = "smartlend"
$Root = Split-Path $PSScriptRoot -Parent
$Tf = Join-Path $Root "terraform"

$aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
if (-not (Test-Path $aws)) { $aws = "aws" }

function Fail([string]$msg) { Write-Host "`nFAILED: $msg" -ForegroundColor Red; exit 1 }

$account = & $aws sts get-caller-identity --query Account --output text 2>$null
if (-not $account) { Fail "AWS CLI is not configured" }
$registry = "$account.dkr.ecr.$Region.amazonaws.com"
$imageUri = "${registry}/${RepoName}:latest"

docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "Docker Desktop is not running" }

Write-Host "[build] docker build (uses layer cache; code-only changes are quick)..."
docker build -t $imageUri $Root
if ($LASTEXITCODE -ne 0) { Fail "docker build failed" }

& $aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $registry
if ($LASTEXITCODE -ne 0) { Fail "docker login to ECR failed" }
docker push $imageUri
if ($LASTEXITCODE -ne 0) { Fail "docker push failed" }

# user_data pulls :latest only on FIRST boot, so a new image needs a new
# instance. -replace recreates just the EC2 instance; the database, Elastic
# IP and everything else stay put (~3-4 min of downtime while it boots).
Write-Host "[terraform] replacing the instance so it boots on the new image..."
Push-Location $Tf
terraform apply -auto-approve -input=false -replace=aws_instance.app -var "ecr_image=$imageUri"
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "terraform apply -replace failed" }
$url = terraform output -raw app_url
Pop-Location

Write-Host "[health] polling $url/health ..."
$deadline = (Get-Date).AddMinutes(12)
$up = $false
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 10
        if ($r.StatusCode -eq 200) { $up = $true; break }
    } catch { }
    Start-Sleep -Seconds 15
}
if (-not $up) { Fail "replacement instance did not come up - check /var/log/cloud-init-output.log" }

Write-Host "`nPublished: $url now runs the current code." -ForegroundColor Green

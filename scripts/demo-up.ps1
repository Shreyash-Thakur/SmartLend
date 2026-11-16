# One-click SmartLend demo: create-or-resume the AWS stack, wait until the
# app answers over HTTP, seed demo data when the database is fresh, open the
# browser. Safe to re-run from ANY state (nothing deployed / stopped / up).
#
# Run:  demo-up.bat at the repo root, or
#       powershell -NoProfile -ExecutionPolicy Bypass -File scripts\demo-up.ps1
#
# The app lives behind an Elastic IP with the container set to
# --restart unless-stopped, so once this script finishes the demo does NOT
# depend on this laptop or its network in any way.

$ErrorActionPreference = "Continue"
$Region = "ap-south-1"
$RepoName = "smartlend"
$DbId = "smartlend-db"
$Root = Split-Path $PSScriptRoot -Parent
$Tf = Join-Path $Root "terraform"

$aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
if (-not (Test-Path $aws)) { $aws = "aws" }

function Fail([string]$msg) { Write-Host "`nFAILED: $msg" -ForegroundColor Red; exit 1 }

Write-Host "== SmartLend demo-up ==" -ForegroundColor Cyan

# -- 1. The container image the instance pulls on boot -----------------------
$account = & $aws sts get-caller-identity --query Account --output text 2>$null
if (-not $account) { Fail "AWS CLI is not configured (sts get-caller-identity returned nothing)." }
$registry = "$account.dkr.ecr.$Region.amazonaws.com"
$imageUri = "${registry}/${RepoName}:latest"

& $aws ecr describe-repositories --repository-names $RepoName --region $Region --output text 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ecr] Repository '$RepoName' does not exist - creating it (storing one image costs ~3c/month)."
    & $aws ecr create-repository --repository-name $RepoName --region $Region --output text | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "could not create the ECR repository" }
}

& $aws ecr describe-images --repository-name $RepoName --image-ids imageTag=latest --region $Region --output text 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[image] No image in ECR yet - building and pushing (first run only, ~10 min)." -ForegroundColor Yellow
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "Docker Desktop is not running - start it, then re-run demo-up.bat" }
    docker build -t $imageUri $Root
    if ($LASTEXITCODE -ne 0) { Fail "docker build failed (output above)" }
    & $aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $registry
    if ($LASTEXITCODE -ne 0) { Fail "docker login to ECR failed" }
    docker push $imageUri
    if ($LASTEXITCODE -ne 0) { Fail "docker push failed" }
}

# -- 2. The stack: apply is a no-op when everything already matches ----------
Write-Host "[terraform] apply (no-op if the stack is already up)..."
Push-Location $Tf
terraform apply -auto-approve -input=false -var "ecr_image=$imageUri" 2>&1 | Tee-Object -Variable applyOut
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "terraform apply failed (output above)" }
# A freshly created database is empty, so it is the one case that needs seeding.
$freshDb = [bool]($applyOut | Select-String "aws_db_instance.smartlend: Creation complete")
$instanceId = terraform output -raw instance_id
$url = terraform output -raw app_url
Pop-Location
if (-not $instanceId -or -not $url) { Fail "terraform outputs missing - inspect terraform state" }

# -- 3. Resume anything demo-stop left stopped (no-ops when already running) --
& $aws rds start-db-instance --db-instance-identifier $DbId --region $Region --output text 2>$null | Out-Null
Write-Host "[rds] waiting until the database is available..."
& $aws rds wait db-instance-available --db-instance-identifier $DbId --region $Region
& $aws ec2 start-instances --instance-ids $instanceId --region $Region --output text 2>$null | Out-Null
& $aws ec2 wait instance-running --instance-ids $instanceId --region $Region

# -- 4. Wait for the app itself ----------------------------------------------
# Fresh instances run first-boot setup (docker install + image pull): minutes.
Write-Host "[health] polling $url/health (up to 20 min on a fresh build)..."
$deadline = (Get-Date).AddMinutes(20)
$up = $false
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 10
        if ($r.StatusCode -eq 200) { $up = $true; break }
    } catch { }
    Start-Sleep -Seconds 15
}
if (-not $up) { Fail "app did not answer within 20 min - check /var/log/cloud-init-output.log on the instance" }

# -- 5. Demo data, only on a brand-new database ------------------------------
if ($freshDb) {
    Write-Host "[seed] Fresh database - submitting demo applications through the real API..."
    & (Join-Path $Root ".venv\Scripts\python.exe") (Join-Path $Root "scripts\seed_demo_applications.py") --url $url
    if ($LASTEXITCODE -ne 0) { Write-Host "[seed] seeding failed - the app is still up, re-run the seed script by hand" -ForegroundColor Yellow }
} else {
    Write-Host "[seed] Existing database kept - skipping seed."
}

Start-Process $url
Write-Host "`nUP: $url" -ForegroundColor Green
Write-Host "This URL now lives on AWS - your wifi can change freely. Stop billing with demo-stop.bat (running ~`$1/day, stopped ~`$0.26/day)."

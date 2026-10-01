# Start the SmartLend AWS stack: RDS first (app needs the DB), then EC2.
# The container self-starts on boot (--restart unless-stopped) and the
# Elastic IP keeps its address, so the app returns at http://15.207.0.203
# a few minutes after this script finishes.
#
# Run:  powershell -File scripts\aws-start.ps1

$ErrorActionPreference = "Stop"
$aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
$region = "ap-south-1"
$instanceId = "i-0207079c8ce7801d1"
$dbId = "smartlend-db"

Write-Host "Starting RDS instance $dbId (takes ~3-5 min to become available)..."
& $aws rds start-db-instance --db-instance-identifier $dbId --region $region --output text | Out-Null

Write-Host "Waiting for RDS to be available..."
& $aws rds wait db-instance-available --db-instance-identifier $dbId --region $region

Write-Host "RDS available. Starting EC2 instance $instanceId ..."
& $aws ec2 start-instances --instance-ids $instanceId --region $region --output text | Out-Null
& $aws ec2 wait instance-running --instance-ids $instanceId --region $region

Write-Host "EC2 running. Waiting ~60s for Docker + container to come up..."
Start-Sleep -Seconds 60

try {
    $resp = Invoke-WebRequest -Uri "http://15.207.0.203/health" -UseBasicParsing -TimeoutSec 20
    Write-Host "App is UP: $($resp.StatusCode) -> http://15.207.0.203"
} catch {
    Write-Host "App not answering yet - give it another minute, then open http://15.207.0.203/health"
}

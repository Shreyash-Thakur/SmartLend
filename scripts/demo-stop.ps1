# Stop the SmartLend AWS stack (EC2 + RDS) without deleting anything, so
# demo-up.bat can resume it in ~5 minutes at the SAME URL.
#
# Billing while stopped: EBS storage + the Elastic IP, roughly $0.26/day.
# NOTE: AWS force-restarts a stopped RDS instance after 7 days.
# Full teardown (back to $0, next demo-up rebuilds in ~15 min):
#   cd terraform && terraform destroy
#
# Run:  demo-stop.bat at the repo root.

$ErrorActionPreference = "Continue"
$Region = "ap-south-1"
$DbId = "smartlend-db"
$Tf = Join-Path (Split-Path $PSScriptRoot -Parent) "terraform"

$aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
if (-not (Test-Path $aws)) { $aws = "aws" }

Push-Location $Tf
$instanceId = terraform output -raw instance_id 2>$null
Pop-Location
if (-not $instanceId) {
    Write-Host "No instance in the terraform state - the stack is not deployed, nothing to stop."
    exit 0
}

Write-Host "Stopping EC2 instance $instanceId ..."
& $aws ec2 stop-instances --instance-ids $instanceId --region $Region --output text 2>$null | Out-Null

Write-Host "Stopping RDS instance $DbId ..."
& $aws rds stop-db-instance --db-instance-identifier $DbId --region $Region --output text 2>$null | Out-Null

Write-Host "Current states:"
& $aws ec2 describe-instances --instance-ids $instanceId --region $Region --query "Reservations[0].Instances[0].State.Name" --output text
& $aws rds describe-db-instances --db-instance-identifier $DbId --region $Region --query "DBInstances[0].DBInstanceStatus" --output text

Write-Host "`nDone. The Elastic IP stays allocated so the URL survives; resume with demo-up.bat."

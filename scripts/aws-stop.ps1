# Stop the SmartLend AWS stack (EC2 + RDS) without deleting anything.
# Billing while stopped: storage + Elastic IP only (~$0.26/day vs ~$1.1/day
# running). NOTE: AWS force-restarts a stopped RDS instance after 7 days.
#
# Requires: AWS CLI configured as a user with the smartlend-start-stop
# inline policy (ec2/rds start+stop+describe).
#
# Run:  powershell -File scripts\aws-stop.ps1

$ErrorActionPreference = "Stop"
$aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
$region = "ap-south-1"
$instanceId = "i-0207079c8ce7801d1"
$dbId = "smartlend-db"

Write-Host "Stopping EC2 instance $instanceId ..."
& $aws ec2 stop-instances --instance-ids $instanceId --region $region --output text | Out-Null

Write-Host "Stopping RDS instance $dbId ..."
& $aws rds stop-db-instance --db-instance-identifier $dbId --region $region --output text | Out-Null

Write-Host "Stop requested for both. Current states:"
& $aws ec2 describe-instances --instance-ids $instanceId --region $region `
    --query "Reservations[0].Instances[0].State.Name" --output text
& $aws rds describe-db-instances --db-instance-identifier $dbId --region $region `
    --query "DBInstances[0].DBInstanceStatus" --output text

Write-Host "`nDone. Reminder: the Elastic IP (15.207.0.203) stays allocated and"
Write-Host "billing (~`$0.12/day) so the URL survives. RDS auto-restarts after 7 days."

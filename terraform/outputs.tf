# What apply prints at the end — the two values you otherwise dig out of
# the console. `terraform output` re-prints them any time.

output "app_url" {
  description = "The app, once the instance finishes first-boot setup (~3-4 min after apply)"
  value       = "http://${aws_eip.app.public_ip}"
}

output "ssh_command" {
  description = "Log into the app server"
  value       = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${aws_eip.app.public_ip}"
}

output "rds_endpoint" {
  description = "Database host (reachable only from inside the VPC)"
  value       = aws_db_instance.smartlend.address
}

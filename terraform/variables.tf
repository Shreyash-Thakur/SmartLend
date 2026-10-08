# Everything you changed by hand in the console during Phase 2 is a
# variable here. Values come from terraform.tfvars (gitignored) — see
# terraform.tfvars.example.

variable "region" {
  description = "AWS region (Phase 2 was built in Mumbai)"
  type        = string
  default     = "ap-south-1"
}

variable "instance_type" {
  description = "EC2 size for the app server"
  type        = string
  default     = "t3.micro"
}

variable "db_instance_class" {
  description = "RDS size"
  type        = string
  default     = "db.t4g.micro"
}

variable "db_username" {
  description = "RDS master username (also the app's DB user)"
  type        = string
  default     = "smartlend"
}

variable "db_password" {
  description = "RDS master password. NEVER in the repo: set it in terraform.tfvars (gitignored) or via the TF_VAR_db_password environment variable."
  type        = string
  sensitive   = true # keeps it out of plan output and console logs
}

variable "my_ip_cidr" {
  description = "Your public IP as a /32 for the SSH rule, e.g. 110.172.16.14/32. Re-apply with a new value when your home IP rotates."
  type        = string
}

variable "key_name" {
  description = "Existing EC2 key pair name (created in Phase 2; the .pem lives in ~/.ssh)"
  type        = string
  default     = "smartlend-key"
}

variable "ecr_image" {
  description = "Full image URI the instance pulls on boot"
  type        = string
  default     = "945323157703.dkr.ecr.ap-south-1.amazonaws.com/smartlend-app:latest"
}

variable "anthropic_api_key" {
  description = "Optional: enables the live agent-briefing panel. Leave empty to deploy without it; the service degrades to a calm 503 with setup notes."
  type        = string
  sensitive   = true
  default     = ""
}

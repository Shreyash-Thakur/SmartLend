# Provider + version pinning. The pin matters: provider majors change
# resource behavior, and "works on my machine" in IaC is usually an
# unpinned provider. ~> 6.0 means "any 6.x, never 7".
terraform {
  required_version = ">= 1.7"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region
  # Credentials come from the AWS CLI config (aws configure) — never from
  # this file. Every resource gets the project tag for cost attribution.
  default_tags {
    tags = {
      Project   = "smartlend"
      ManagedBy = "terraform"
    }
  }
}

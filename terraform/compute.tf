# EC2 + IAM role + Elastic IP. The biggest upgrade over the manual build:
# user_data makes the instance SELF-CONFIGURING — docker, swap, ECR pull,
# container run all happen on first boot, so `terraform apply` ends with a
# working app and no SSH session required.

# Find the current Ubuntu 24.04 AMI instead of hardcoding an ID that goes
# stale (AMI IDs are region-specific and rotate with patches).
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical's official account

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# The role the instance wears to pull from ECR — no registry credentials on
# disk, ever. In Terraform the console's hidden wrapper becomes visible:
# aws_iam_role and aws_iam_instance_profile are two separate resources.
resource "aws_iam_role" "ec2" {
  name = "smartlend-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "ec2" {
  name = "smartlend-ec2-role"
  role = aws_iam_role.ec2.name
}

resource "aws_instance" "app" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public_1a.id
  vpc_security_group_ids      = [aws_security_group.ec2.id]
  key_name                    = var.key_name
  iam_instance_profile        = aws_iam_instance_profile.ec2.name
  associate_public_ip_address = true

  # Without this, a user_data change is stored on the instance but the new
  # boot script never RUNS (the provider default is in-place attribute update)
  # - discovered live when a Gemini-key rollout silently no-opped.
  user_data_replace_on_change = true

  root_block_device {
    volume_size = 16 # 8 GiB default is too tight for OS + Docker + image
    volume_type = "gp3"
  }

  # Self-configuration script; the DB endpoint and password are interpolated
  # at apply time. Changing this script REPLACES the instance (user_data is
  # immutable per instance) — plan will say so, and that is correct behavior.
  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    region            = var.region
    ecr_image         = var.ecr_image
    db_url            = "postgresql+psycopg2://${var.db_username}:${var.db_password}@${aws_db_instance.smartlend.address}:5432/smartlend"
    anthropic_api_key = var.anthropic_api_key
    gemini_api_key    = var.gemini_api_key
  })

  # The app writes its seed to the DB on first boot, so the DB must exist
  # first. Terraform infers most ordering from references; this makes the
  # intent explicit anyway.
  depends_on = [aws_db_instance.smartlend]

  tags = { Name = "smartlend-ec2" }
}

# A fixed public address that survives stop/start. Billing note: ~$0.005/h
# while allocated — terraform destroy releases it (the classic leak, solved
# by making it code).
resource "aws_eip" "app" {
  domain = "vpc"

  tags = { Name = "smartlend-eip" }
}

resource "aws_eip_association" "app" {
  instance_id   = aws_instance.app.id
  allocation_id = aws_eip.app.id
}

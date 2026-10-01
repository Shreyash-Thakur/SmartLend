# Both security groups, including the Phase 2 centerpiece: the RDS group's
# source is the EC2 GROUP, not a CIDR — membership, not address, which is
# why instance replacement never breaks database access.

resource "aws_security_group" "ec2" {
  name        = "smartlend-ec2-sg"
  description = "SmartLend EC2 - SSH from me, HTTP from world"
  vpc_id      = aws_vpc.smartlend.id

  ingress {
    description = "SSH from my IP only"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }

  ingress {
    description = "HTTP from anywhere - it is a public website"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "smartlend-ec2-sg" }
}

resource "aws_security_group" "rds" {
  name        = "smartlend-rds-sg"
  description = "SmartLend RDS - Postgres from EC2 SG only"
  vpc_id      = aws_vpc.smartlend.id

  ingress {
    description     = "Postgres from members of the EC2 security group"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ec2.id] # the chain
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "smartlend-rds-sg" }
}

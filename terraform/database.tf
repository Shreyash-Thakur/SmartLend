# RDS — every hesitation screen from the wizard is an explicit argument
# here. This is why the console detour was worth it: you can now read this
# file and know exactly what each line cost you a pause to learn.

resource "aws_db_subnet_group" "smartlend" {
  name = "smartlend-db-subnets"
  # Two AZs because RDS insists on the OPTION of failover even when
  # multi_az is false below. Option != enabled.
  subnet_ids = [aws_subnet.private_1a.id, aws_subnet.private_1b.id]

  tags = { Name = "smartlend-db-subnets" }
}

resource "aws_db_instance" "smartlend" {
  identifier     = "smartlend-db"
  engine         = "postgres"
  engine_version = "16" # major pinned to match docker-compose; minors auto-patch
  instance_class = var.db_instance_class

  # The silent-gotcha field: without db_name, the server boots with no
  # database and the app's connection URL fails.
  db_name  = "smartlend"
  username = var.db_username
  password = var.db_password

  allocated_storage = 20
  storage_type      = "gp3"
  # No max_allocated_storage = storage autoscaling stays OFF (cost lever).

  db_subnet_group_name   = aws_db_subnet_group.smartlend.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  availability_zone      = "${var.region}a" # same AZ as EC2: free, faster
  publicly_accessible    = false            # the "answer no" question
  multi_az               = false            # the other "answer no" question

  backup_retention_period = 1
  skip_final_snapshot     = true  # lab: destroy must not strand a snapshot
  deletion_protection     = false # we practice teardown on purpose

  performance_insights_enabled = false
  auto_minor_version_upgrade   = true

  tags = { Name = "smartlend-db" }
}

# The network layer — everything you clicked through in Phase 2, step 1.
# Same CIDRs, same three-subnet asymmetry (one public for EC2, two private
# across AZs because RDS demands a two-AZ subnet group even for Single-AZ).

resource "aws_vpc" "smartlend" {
  cidr_block = "10.0.0.0/16"

  # The two DNS flags from the console hesitation log: resolution is on by
  # default, hostnames is NOT on custom VPCs — this is that checkbox.
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "smartlend-vpc" }
}

resource "aws_subnet" "public_1a" {
  vpc_id            = aws_vpc.smartlend.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "${var.region}a"

  tags = { Name = "smartlend-public-1a" }
}

resource "aws_subnet" "private_1a" {
  vpc_id            = aws_vpc.smartlend.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "${var.region}a"

  tags = { Name = "smartlend-private-1a" }
}

resource "aws_subnet" "private_1b" {
  vpc_id            = aws_vpc.smartlend.id
  cidr_block        = "10.0.3.0/24"
  availability_zone = "${var.region}b"

  tags = { Name = "smartlend-private-1b" }
}

resource "aws_internet_gateway" "igw" {
  # Attachment is implicit here — the console's forgettable "Attach to VPC"
  # step is just this argument.
  vpc_id = aws_vpc.smartlend.id

  tags = { Name = "smartlend-igw" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.smartlend.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = { Name = "smartlend-public-rt" }
}

# This association is the entire definition of "public subnet". The private
# subnets stay on the main route table (local-only) by doing nothing — no
# NAT gateway, by design and by budget.
resource "aws_route_table_association" "public_1a" {
  subnet_id      = aws_subnet.public_1a.id
  route_table_id = aws_route_table.public.id
}

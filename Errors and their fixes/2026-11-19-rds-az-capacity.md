# terraform apply failed: RDS could not be created in any configured AZ

- **Date:** 2026-11-19
- **Reported as:** `InvalidVPCNetworkStateFault: You can't create a db.t4g.micro database instance because no subnets exist in Availability Zones with sufficient capacity for VPC and storage type : gp3 for db.t4g.micro. Please first create at least one new subnet; choose from these Availability Zones: ap-south-1c.` (CreateDBInstance, StatusCode 400, RequestID 67357763-d5df-4ab2-88da-98405dc1be58)
- **Symptom:** `demo-up.bat` / `terraform apply` failed while creating `aws_db_instance.smartlend`; every other resource up to that point created fine. Deploy was from a fully destroyed stack on final-eval day.
- **Where:** terraform/database.tf:32 (the `availability_zone = "${var.region}a"` pin), terraform/database.tf:9 (subnet group spanning only 1a/1b), terraform/network.tf:24-38 (private subnets in 1a and 1b only)
- **Root cause:** AWS ap-south-1a and 1b temporarily had no db.t4g.micro + gp3 capacity, and our config both pinned the instance to 1a and gave the DB subnet group no subnet in 1c (the only AZ AWS offered), so RDS had nowhere it was allowed to place the instance.
- **Why it wasn't caught:** AZ capacity is an AWS-side, time-varying condition; `terraform validate`/`plan` cannot see it, and every previous apply happened while 1a had capacity.
- **Fix:** Added `aws_subnet.private_1c` (10.0.4.0/24), added it to `aws_db_subnet_group.smartlend`, and removed the `availability_zone` pin so RDS picks any AZ in the group with capacity. Shape of fix: widen the options rather than re-pin to 1c, because 1c can run dry just as 1a did. Cross-AZ EC2-to-RDS traffic costs $0.01/GB, negligible at demo scale; subnets are free.
- **Evidence:** `terraform validate` → "Success! The configuration is valid." Subsequent `demo-up` apply created `aws_db_instance.smartlend` without the fault (see demo-up output of 2026-11-19; stack reached a 200 on /health).
- **Blast radius:** terraform/network.tf, terraform/database.tf only. No application code; the Docker image is unaffected. State grows by one subnet. `demo-stop`/`demo-up`/`demo-publish` lifecycle unchanged.
- **Still open:** The EC2 instance stays in 1a while RDS may now land elsewhere; if demo latency ever matters, pin both to the same AZ *after checking capacity that day*.
- **Related:** Errors and their fixes/2026-11-16-demo-dies-on-wifi-switch.md (same demo lifecycle scripts)

# user_data change applied as in-place update - new boot script never ran

**Date:** 2026-10-09 - **Where:** terraform apply (instance swap for the Gemini agent key)

## Symptom
Apply reported `0 to add, 1 to change, 0 to destroy` and completed, but the
deployed app still ran the old image with no GEMINI_API_KEY: /api/agent/status
said not configured and the briefing error used the old service's wording.

## Root cause (one sentence)
`aws_instance.user_data` changes default to an in-place attribute update
(`user_data_replace_on_change = false`), so the new boot script was stored on
the instance but never executed - the comment in compute.tf asserting that a
user_data change replaces the instance was wrong about the provider default.

## Fix
- `terraform/compute.tf`: set `user_data_replace_on_change = true` so boot
  script changes genuinely re-create the instance from now on.
- Immediate repair without waiting for a replacement: SSH to the instance,
  pull :latest, re-run the container with GEMINI_API_KEY (DB URL recovered
  from the old container's own env, never printed).
- Bonus failure en route: SSH timed out because the SG's "My IP" rule had
  yesterday's home IP; fixed the right way - updated my_ip_cidr in
  terraform.tfvars and applied (in-place SG change, seconds).

## Verification (real output)
- `/api/agent/status` on http://35.154.151.137 -> `{"configured": true}`.
- POST /agent-briefing on a seeded DEFER -> model `gemini-2.5-flash`,
  grounded summary citing p_ml 0.753 vs CBES 0.440, suggestedReasonCodes
  ['REJ-THIN-FILE','REJ-EMI-BURDEN','GEN-MODEL-MISMATCH'].

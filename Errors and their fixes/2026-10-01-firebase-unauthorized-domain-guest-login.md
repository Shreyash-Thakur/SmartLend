# Firebase `auth/unauthorized-domain` on the AWS deployment → guest login

**Date:** 2026-10-01 · **Where:** deployed EC2 instance (http://15.207.0.203), `/auth` page

## Symptom

Google sign-in failed with `Firebase: Error (auth/unauthorized-domain)` on the
deployed site; the same flow worked on localhost. Earlier the page sat on
"Loading SmartLend..." before surfacing the error.

## Root cause (one sentence)

Firebase Auth only permits sign-in from domains on its per-project
*authorized domains* allowlist, which pre-includes `localhost` but **cannot
contain a raw IP address**, so a deployment served from `15.207.0.203` can
never pass the check — an environment constraint, not a code defect.

## Fix (cause, not symptom)

Make authentication adapt to the environment instead of assuming Firebase:

1. `frontend/src/hooks/useAuth.ts` — added `loginAsGuest()`: a stable
   browser-local identity (`guest-<random>` uid persisted in localStorage so
   "my applications" stays attached to the same applicant), restored on load
   whenever Firebase is disabled; logout clears it.
2. `frontend/src/pages/AuthPage.tsx` — Google button renders only when
   Firebase is configured; a "Continue as Guest" button is always available
   (primary styling when Firebase is absent). Removed the raw missing-keys
   warning from the login card.
3. `.dockerignore` — excluded `frontend/.env` (the `VITE_FIREBASE_*` client
   config) from the build context, so Docker/EC2 builds ship with Firebase
   disabled and guest login active, while local `vite` dev keeps Google
   sign-in. This also stops env files leaking into images generally.

Proper Firebase on AWS remains possible later via Phase 4 (real domain +
HTTPS + adding the domain to Firebase's authorized list); this fix removes
the hard dependency.

## Verification (real output)

- `npx tsc --noEmit` clean; `npm run build` ✓ (33.17s).
- Rebuilt image, pushed digest `sha256:b54f740a…` (single changed layer),
  redeployed on EC2 reusing the container's own DB URL.
- Container log: `Startup complete | DB initialized and model loaded`.
- `GET /health` → 200 with live artifact payload.
- Deployed chunk `assets/AuthPage-o502YwbZ.js` contains `Continue as Guest`
  (grep count 1), confirming the new auth path shipped.

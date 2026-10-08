# Frontend revamp — plan and task list

Status: **planning only, nothing implemented yet** (written 2026-10-08).
Any session picking this up: read this whole file first, then work the task
list top to bottom, one phase per commit, smoke test after every phase.

---

## 1. What was asked (the brief, kept as given)

1. **ShapeGrid background on every page.** React Bits `<ShapeGrid />` (canvas,
   animated squares, hover fill + trail) sits behind all content on every route.
   Reference usage: `speed=0.5, squareSize=40, direction='diagonal',
   borderColor='#fff', hoverFillColor='#222', shape='square', hoverTrailAmount=5`.
   Full source was pasted by the user (JS + CSS variant); it is also on
   reactbits.dev.
2. **Theme.** Two reference screenshots:
   - Dark: "Shadcn Fintech" dashboard — near-black background, left sidebar
     with grouped nav (Daily / Money / Insights / Auth), monochrome cards with
     thin borders and small pill labels on top edges, white primary buttons,
     green/red only for meaning (up/down, in/out), score gauge, progress bars.
   - Light: "Shadcn Dashboard" CRM — white/very light grey, left sidebar,
     breadcrumb top bar with mail + bell icons with count badges, KPI cards with
     pastel icon tiles and a "View all" footer row, single blue accent.
3. **Day/night toggle** — simple switch between the two themes above.
4. **Page transitions + a preloader/spinner** so switching pages never shows a
   blank screen.
5. **React Bits components the user wants to try:**
   - `BellToggle` — idea stage only, *do not build yet*. Suggest where a
     notification bell is genuinely useful; if nowhere, skip it.
   - `GlideSelect` — replace the select inputs on the loan application form.
   - `MaskedHeading` — intro screen showing the site name before login.
   - `TechText` — **skipped** (user decision).
6. **Voice chat** — check whether a voice feature exists, make sure it is
   enabled, add the Sarvam API key and connect it.
7. **Main dashboard should look good first** — dashboard is the priority page.
8. **Smoke test after every change** so nothing breaks by accident.
9. Answer: is `npx shadcn add @react-bits/...` or the copy-paste prompt the
   better way to bring React Bits in?

---

## 2. What the codebase looks like today (findings)

| Area | Current state |
| --- | --- |
| Stack | Vite 5 + React 18 + **TypeScript** (`allowJs: false`) + Tailwind 3.4 + framer-motion 11 + lucide-react + recharts + zustand |
| shadcn | **Not set up** — no `components.json`, no CSS-variable theme |
| Theme | Hard-coded light palette (green `primary`, sky `accent`) in `tailwind.config.js`; `globals.css` paints `:root` with gradients. **No dark mode** (`darkMode` not configured) |
| Layout | `components/layouts/DashboardLayout.tsx` — sticky top header nav, no sidebar. Used by all dashboard pages |
| Transitions | `components/layouts/PageTransition.tsx` — fade/slide on mount only, no exit animation; wraps `<main>` inside DashboardLayout |
| Loading | `App.tsx` routes are `lazy()`; Suspense fallback and auth-loading screen are plain text "Loading SmartLend..." on `bg-neutral-50` — this is the blank-ish flash |
| Routes | `/` Landing (logged out) · `/auth` · `/dashboard/customer` · `/dashboard/customer/new` · `/dashboard/org` · `/dashboard/models` · `/analytics/geo` · `/review` · `/review/:id` |
| Selects | 16 `<select>`s: 11 in `pages/CustomerNewApplication.tsx`, 5 in `components/forms/LoanApplicationForm.tsx` (+ table filter, models page, `common/Select.tsx`) |
| Tests | No frontend test runner. Checks available: `npx tsc --noEmit`, `npm run build`, backend `pytest` |

### Voice — it already exists, it is just switched off

- Backend: `backend/app/services/voice_service.py` + `routers/voice.py`
  (`/api/voice/status`, `/synthesize`, `/transcribe`), 24 tests. Documented in
  `docs/VOICE-MODULE.md`.
  - **Speech-to-text = Sarvam** (`saarika:v2`), Indian languages.
  - **Text-to-speech = ElevenLabs**.
- Frontend: `services/voice.ts`; used only in `pages/CustomerNewApplication.tsx`
  — a mic button to dictate into the purpose-details field, and a "listen"
  button that reads out the decision.
- **Why it is off:** there is **no root `.env` file at all**. Keys are read
  from `<repo root>/.env` (not `frontend/.env`). Without it, the backend
  correctly answers 503 and the UI hides voice controls.
- It is **dictation + readout, not a chat**. There is no conversational voice
  chat anywhere. If a real chat is wanted, that is new work (see §6, Phase 7).

### Problem found: `frontend/.env` is corrupted

The whole React Bits prompt (ShapeGrid source, props table, etc.) was pasted
into the bottom of `frontend/.env` (now 498 lines; only the first 6 Firebase
lines belong there). Vite still starts because the junk lines have no `VITE_`
prefix, but it is fragile. Fix in Phase 0 — remove everything after the
`VITE_FIREBASE_APP_ID` line. (File is gitignored, so nothing leaked to git.)

---

## 3. Answer: `npx shadcn add` vs the copy-paste prompt

They deliver **the same component code**; the difference is only how it
arrives.

- `npx shadcn@latest add @react-bits/...` needs shadcn initialised first
  (`npx shadcn init` → creates `components.json`, and **rewrites
  `tailwind.config.js` and `globals.css`** with its own theme variables). On
  this project that would clobber the existing palette in one step, the
  exact kind of silent breakage we want to avoid. It also needs the
  `@react-bits` registry configured in `components.json`.
- The **`-JS-CSS` variants are JavaScript**; this project is TypeScript with
  `allowJs: false`, so they would not even type-check without changing config.
- The copy-paste prompt is fully under our control: we see every line, put it
  where we want, and convert it to TypeScript.

**Recommendation:** use the copy-in approach, but take the **TypeScript
variants** from reactbits.dev (`TS-CSS` or `TS-TW`) rather than `JS-CSS`. If a
TS variant is missing, convert the JS one by hand (add prop types). Do not run
`shadcn init`. Our theme tokens (Phase 1) give us the shadcn *look* without
the shadcn CLI.

---

## 4. Where a notification bell actually helps (suggestions, not built)

The app already has natural "something changed that you're waiting for"
moments. Ranked by usefulness:

1. **Org reviewer — review queue.** Applications the engine **defers to a
   human** (the deferral arm) land in `/review`. A bell with a count of
   unreviewed deferred applications is the single most useful notification in
   the app.
2. **Customer — decision ready / status changed.** After submitting, the
   customer waits for approve/decline/deferred-to-review. Bell badge when an
   application's status changes since they last looked.
3. **Org — model/serving changes.** Serving model or `ENGINE_VERSION` changed
   (data already fetched for the header badge). Low frequency, nice-to-have.

How `BellToggle` fits: it is a *toggle* (on/off), so its best role is
**"Notify me when my decision is ready" / mute alerts**, while the bell in the
top bar (like screenshot 2) is a normal icon + count badge + dropdown.
First version can be **frontend-only**: poll the existing applications list,
compare with a "last seen" timestamp in localStorage. No backend change needed.
Decision: **park until the user says go.**

---

## 5. How we avoid breaking things (guard rails)

1. **Branch:** all work on `feat/frontend-revamp`; `main` stays as the safe
   copy. One commit per phase → any phase can be reverted alone.
2. **Never touch backend logic** for UI work. The only backend-adjacent change
   is adding keys to root `.env`. The uncommitted
   `backend/app/services/ml_service.py` and `requirements-api.txt` changes are
   the user's and are left alone.
3. **No `shadcn init`, no blanket config rewrites.** Tailwind config is
   *extended*, never replaced. Existing class names keep working.
4. **Theme through tokens.** Colours become CSS variables on `:root` /
   `.dark`; components switch to tokens gradually. A page not yet migrated
   still renders correctly in light mode.
5. **Logic untouched when restyling.** Form submission, validation (zod +
   react-hook-form), auth guards, API calls, the voice degrade-to-null logic
   are not edited — only markup/classes around them. GlideSelect is wired via
   react-hook-form `Controller` so the submitted values are identical.
6. **Accessibility/perf for the background:** ShapeGrid pauses when the tab is
   hidden (already built in), honours `prefers-reduced-motion` (we add this),
   sits at `z-index:-1` with `pointer-events:none`, and listens for mouse
   moves on `window` so hover still works through the content.

### Smoke test — run after **every** phase

```bash
cd frontend && npx tsc --noEmit          # 1. types
cd frontend && npm run build             # 2. production build
python -m pytest backend/tests -q        # 3. backend unaffected (only when backend/.env touched)
```

Then with backend + `npm run dev` running, click through (browser automation
or by hand), in **both light and dark**:

- [ ] `/` loads, intro/landing shows, no console errors
- [ ] `/auth` sign-in works and redirects by role
- [ ] Customer: dashboard → new application → fill every select → submit →
      decision shows
- [ ] Org: `/dashboard/org`, `/review`, `/review/:id`, `/dashboard/models`,
      `/analytics/geo` (Leaflet map renders, tiles visible in dark)
- [ ] Charts readable in both themes
- [ ] Navigating between pages shows the loader/transition, never blank
- [ ] Theme choice survives a page reload
- [ ] Logout works

A phase is "done" only when all of the above pass. Failures get reported, not
hidden.

---

## 6. Task list (in order)

### Phase 0 — housekeeping
- [ ] Clean `frontend/.env` (keep only the 6 `VITE_FIREBASE_*` lines)
- [x] Create branch `feat/frontend-revamp`
- [x] Baseline smoke test on untouched code, note anything already broken

### Phase 1 — theme foundation + day/night toggle  ✅ done 2026-10-08

Implemented as: `neutral`/`gray` scales on CSS variables (light values
unchanged) that flip under `.dark`; a dark-mode compatibility block in
`globals.css` re-maps `bg-white`, pastel tints (`bg-*-50/100`), dark coloured
text (`text-*-600…950`) and light gradient stops; `ink` fixed colour for the
4 panels that were `bg-neutral-900`; `store/themeStore.ts` + inline script in
`index.html`; `components/theme/ThemeToggle.tsx` in DashboardLayout, Landing
header and floating on `/auth`. Charts/map colours still hard-coded → Phase 4.
Known pre-existing quirk: `bg-white/88` (Card) and `primary-200/300/400/700/800`
are not real Tailwind classes, so they render nothing in light mode; untouched.

- [ ] `tailwind.config.js`: `darkMode: 'class'`; add token colours
      (`background`, `foreground`, `card`, `border`, `muted`, `accent`)
      backed by CSS variables
- [ ] `globals.css`: light tokens (screenshot 2: white, light grey, blue
      accent) and dark tokens (screenshot 1: near-black, zinc cards, white
      primary); keep green/red for semantic up/down
- [ ] `store/uiStore.ts`: `theme` state persisted to localStorage, defaults
      to **light**; apply class to `<html>` before first paint
      (inline script in `index.html` to avoid a flash)
- [ ] Sun/moon toggle component in the top bar
- [ ] Smoke test

### Phase 2 — ShapeGrid background on every page
- [ ] Add `components/background/ShapeGrid.tsx` (+ `.css`), TS variant
- [ ] Change hover listener from canvas → window (content covers the canvas)
- [ ] Add reduced-motion check and devicePixelRatio-aware sizing
- [ ] Mount once in `App.tsx` as fixed full-screen layer behind routes
- [ ] Colours from theme: dark = faint white lines / `#222` hover;
      light = faint grey lines / pale blue hover. Low opacity so text stays
      readable
- [ ] Remove the old gradient backgrounds that would fight it
- [ ] Smoke test (also check CPU in DevTools Performance — should idle low)

### Phase 3 — preloader + page transitions
- [ ] Branded loader (logo + spinner) replacing both "Loading SmartLend..."
      screens in `App.tsx`
- [ ] `AnimatePresence` keyed on route in `App.tsx` for enter + exit
      transitions; move `PageTransition` out of `DashboardLayout`
- [ ] Short top progress bar while lazy chunks load
- [ ] Smoke test

### Phase 4 — dashboard layout (priority page)
- [ ] Replace top-header layout with sidebar layout like the screenshots:
      collapsible left sidebar, grouped nav, user block at the bottom, top bar
      with breadcrumb + theme toggle (+ bell slot, empty for now)
- [ ] Mobile: sidebar becomes a drawer
- [ ] Restyle org dashboard (`Dashboard.org.tsx`) first: KPI cards (screenshot
      2 style), chart cards with pill labels (screenshot 1 style)
- [ ] Then customer dashboard, review, models, geo
- [ ] Recharts/Chart.js colours from tokens; Leaflet dark tiles in dark mode
- [ ] Smoke test after each page

### Phase 5 — intro screen with MaskedHeading
- [ ] Bring in MaskedHeading (TS variant)
- [ ] Intro on `/` for logged-out users: "SmartLend" masked reveal, then
      continue to landing/login; once per session, skippable, skipped entirely
      under reduced motion
- [ ] Smoke test

### Phase 6 — GlideSelect on the loan form
- [ ] Bring in GlideSelect (TS variant); check keyboard + screen-reader
      support before using it everywhere
- [ ] Wrap it in `common/Select.tsx`-compatible API, wire through
      react-hook-form `Controller`
- [ ] Replace the 11 + 5 form selects; verify submitted payload is
      byte-for-byte the same as before (compare network request)
- [ ] Smoke test (full application submit, both roles)

### Phase 7 — voice / Sarvam
- [ ] User creates root `.env` (copy `.env.example`) and adds
      `SARVAM_API_KEY` (and `ELEVENLABS_API_KEY` for readout) — the user
      pastes the key themselves; it never goes in chat or git
- [ ] Restart API, check `GET /api/voice/status` shows STT configured
- [ ] Test mic dictation on the new-application page end to end
- [ ] Optional: move TTS to Sarvam too (Sarvam has TTS with Indian voices) so
      one key covers both — needs a backend provider change + tests, ask first
- [ ] PARKED: voice commands that fill the loan form by speaking — wait for
      the user's go-ahead
- [ ] Smoke test + `pytest backend/tests/test_voice_service.py`

### Phase 8 — parked (needs a go-ahead)
- [ ] Notification bell (§4) with BellToggle as the "notify me" switch
- [ ] Voice commands to fill the form (see Phase 7)

---

## 7. Decisions (answered by the user, 2026-10-08)

1. **Day/night toggle** — simple sun/moon toggle. ✅
2. **TechText** — skipped. Use **MaskedHeading** (animated text with the
   background showing through the letters) instead. ✅
3. **Default theme** — always start in **light** mode; the user's choice is
   remembered after that. ✅
4. **Voice** — dictation + readout is enough for now. **Parked for later:**
   voice commands that fill in the loan form ("my income is 50,000…" fills
   the income field). Do not start until the user says go. ✅

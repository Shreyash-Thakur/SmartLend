# Voice module

Text-to-speech readout of loan decisions, and speech-to-text intake in Indian
languages. Built so the API runs identically with or without API keys — a
missing key degrades to HTTP 503 on the two voice endpoints and changes nothing
else.

| File | Role |
| --- | --- |
| `backend/app/services/voice_service.py` | Provider abstractions, implementations, all logic |
| `backend/app/routers/voice.py` | HTTP surface (3 endpoints) |
| `backend/app/config.py` | Secret loading (shared, pre-existing) |
| `backend/tests/test_voice_service.py` | 24 tests, all network calls mocked |

No new Python dependencies. HTTP uses `urllib` from the standard library.

---

## 1. Environment variables

All of these live in **`.env` at the project root, which is gitignored**.
`.env.example` is the tracked template and holds empty placeholders only —
never put a real key in it.

```dotenv
# Required for /api/voice/synthesize (text-to-speech)
ELEVENLABS_API_KEY=sk_...

# Required for /api/voice/transcribe (speech-to-text)
SARVAM_API_KEY=sk_...

# Optional — sensible defaults are used when blank
ELEVENLABS_VOICE_ID=       # default 21m00Tcm4TlvDq8ikWAM ("Rachel")
ELEVENLABS_MODEL_ID=       # default eleven_multilingual_v2
SARVAM_STT_MODEL=          # default saarika:v2
```

Where to get them:
* ElevenLabs — <https://elevenlabs.io/app/settings/api-keys>. **The key must
  have the `text_to_speech` permission**; a key without it authenticates fine
  and then fails per-call with HTTP 401 `missing_permissions`.
* Sarvam AI — <https://dashboard.sarvam.ai/>.

Keys are read lazily on every call, so adding one to `.env` and restarting the
API is the whole activation procedure. Nothing needs to be uncommented.

## 2. Endpoints

| Method | Path | Behaviour with no key |
| --- | --- | --- |
| `GET` | `/api/voice/status` | **200**, reports `available: false` plus a reason per provider |
| `POST` | `/api/voice/synthesize` | **503** `VOICE_UNAVAILABLE` with the exact env var to set |
| `POST` | `/api/voice/transcribe` | **503** `VOICE_UNAVAILABLE` with the exact env var to set |

`503`, never `500`: a missing credential is a configuration state, not a server
fault, and monitoring should not page anyone for it.

`POST /api/voice/synthesize` takes JSON `{"decision", "explanation", "language"}`
and returns `audio/mpeg` bytes.

`POST /api/voice/transcribe` takes a multipart upload (`file`, optional
`language` form field, 10 MB cap) and returns:

```json
{
  "available": true,
  "provider": "sarvam",
  "text": "mera naam",
  "language": "hi-IN",
  "confidence": null,
  "words": [
    {"word": "mera", "start": 0.0, "end": 0.4, "confidence": null},
    {"word": "naam", "start": 0.5, "end": 0.9, "confidence": null}
  ]
}
```

### Per-word confidence

`words` is **always** present, and every entry always has a `confidence` key.
`null` means *unknown*, not *certain* — do not coerce it to `1.0`. The field
exists to feed a downstream uncertainty layer: a low-confidence transcription of
a load-bearing field (income, dependants, employment) should trigger a
confirmation prompt or a deferral rather than being silently accepted.

Provider reality today: Sarvam `saarika` returns word timestamps but no acoustic
confidence, so we emit `null`. Local Whisper exposes token logprobs, from which
a genuine per-word confidence can be derived — one of the stronger arguments for
the open-source path below.

## 3. Verifying it works

**Without any key** (this is the state the module is designed for):

```bash
python -m pytest backend/tests/test_voice_service.py -q     # 24 passed
python -c "from backend.app.main import app; print('ok')"   # imports clean
uvicorn backend.app.main:app --reload
curl http://127.0.0.1:8000/api/voice/status                 # 200, available:false
```

**Once a key is set**, `/api/voice/status` flips the relevant provider to
`"configured": true` with `"reason": null`, and:

```bash
curl -X POST http://127.0.0.1:8000/api/voice/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"decision":"approved","explanation":"Your income is stable."}' \
  --output decision.mp3

curl -X POST http://127.0.0.1:8000/api/voice/transcribe \
  -F 'file=@sample.wav' -F 'language=hi'
```

The test suite never makes a real HTTP request: `voice_service._http_post` is
the single network chokepoint, and the autouse fixture replaces it with a stub
that raises if anything reaches it.

## 4. The provider-swap seam

Callers (`routers/voice.py`, anything else) only ever touch
`synthesize_decision()`, `transcribe()` and `voice_available()`. Those resolve a
provider through `get_tts_provider()` / `get_stt_provider()`, which return the
module-level `_tts_provider` / `_stt_provider`.

**That resolver pair is the seam.** To swap a provider:

1. Write a class implementing `SpeechToText` or `TextToSpeech`
   (`is_configured`, `unavailable_reason`, and `transcribe`/`synthesize`).
2. Return it from the resolver — either by changing the module default or by
   calling `set_providers(stt=MyProvider())` at startup.

Nothing else changes: no caller edits, no router edits, no schema changes. The
contract implementations must honour:

* `is_configured()` and `unavailable_reason()` never raise;
* `synthesize()` / `transcribe()` raise only `VoiceProviderError`;
* `transcribe()` returns `{"text", "language", "words", "confidence"}` with
  `words` always a list and `confidence: None` where unknown.

The public entry points wrap all of it defensively anyway — a provider that
violates the contract degrades to an "unavailable" result rather than a 500.

## 5. Open-source alternatives (documented, not implemented)

Hosted API credits are limited. These run locally, cost nothing per call, and
drop into the seam above. None are implemented yet — implementing one means
writing the class described in §4 and nothing more.

| Model | Side | Why | Licence |
| --- | --- | --- | --- |
| [`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | STT | 22 Indian languages in one model; the natural default for Indian-language intake | MIT |
| [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | STT | Strong Hindi and Indian English; word timestamps **and token logprobs**, i.e. real per-word confidence | Apache 2.0 |
| [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts) | TTS | Indian-language decision readout without ElevenLabs credits | Apache 2.0 |

Implementation sketch for the STT side (this is the whole diff):

```python
class IndicConformerSpeechToText(SpeechToText):
    name = "indic-conformer"

    def is_configured(self) -> bool:
        return _weights_present()          # no API key involved

    def unavailable_reason(self) -> str | None:
        return None if self.is_configured() else "Model weights not downloaded; see models/README.md"

    def transcribe(self, audio_bytes, language=None) -> dict:
        ...                                # return the documented shape
```

then `set_providers(stt=IndicConformerSpeechToText())`. Weights are gitignored
(`models/**/*.safetensors`), so they are fetched, not committed.

## 6. Accessibility caveat — worth measuring, not a footnote

Voice intake exists so that filling in a loan application is not a literacy
test. The intent is to widen access for low-literacy and rural applicants.

**The failure mode runs the same direction as the intent.** ASR word error rates
are systematically *higher* for exactly the speakers this feature is meant to
serve:

* non-standard accents and regional dialects, which are under-represented in
  training corpora relative to metropolitan standard speech;
* code-switching (Hindi-English and similar), which many models segment badly;
* noisy recording conditions and low-end handset microphones;
* older speakers and speakers with limited prior experience talking to devices,
  who produce more disfluency and hesitation.

If a bad transcription feeds a decision pipeline that defers on uncertainty, the
mechanism intended to *include* rural and low-literacy applicants can end up
raising *their* deferral rate — a fairness regression introduced by an
accessibility feature. That is a real, measurable effect, not a caveat to note
and move past.

What to actually do about it:

1. **Measure it.** Stratify deferral rate and decision outcomes by intake
   modality (voice vs typed) and, where consent allows, by language and region.
   A gap between voice and typed intake for the same applicant profile is the
   number that matters.
2. **Use the confidence channel.** This is why `words[].confidence` is preserved
   and why `None` must not be read as certainty. Low-confidence spans on
   critical fields should trigger a read-back-and-confirm step, not a silent
   deferral.
3. **Never let a transcription error alone cause a rejection.** Route it to
   human review or confirmation, so ASR error surfaces as friction rather than
   as an adverse outcome.
4. **Prefer models trained on Indian speech** (§5) over general-purpose hosted
   ASR for exactly this reason — the `ai4bharat` models exist to close this gap.

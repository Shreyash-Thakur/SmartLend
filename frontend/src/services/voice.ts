/**
 * Client for the optional `/api/voice/*` module.
 *
 * The whole point of this module is that it is *optional*. The backend answers
 * 503 (never 500) when no provider key is configured, and `/api/voice/status`
 * answers 200 in every configuration. Every function here therefore resolves to
 * `null` rather than throwing: a missing key, a network failure, a browser
 * without MediaRecorder — all of it must degrade to "no voice controls", never
 * to a broken page or a blocked form.
 */
import { apiClient } from '@/services/api.client'

export interface VoiceProviderStatus {
  kind: 'tts' | 'stt'
  provider: string
  configured: boolean
  reason: string | null
}

export interface VoiceStatus {
  available: boolean
  tts: VoiceProviderStatus
  stt: VoiceProviderStatus
}

export interface TranscriptionResult {
  text: string
  language?: string | null
}

/** Shape guard: a malformed/proxied response must not crash the caller. */
function isVoiceStatus(value: unknown): value is VoiceStatus {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return typeof candidate.tts === 'object' && typeof candidate.stt === 'object'
}

/**
 * Ask the backend which voice providers are configured.
 * Resolves to `null` when the endpoint is absent, unreachable or malformed —
 * the caller then renders no voice UI at all.
 */
export async function getVoiceStatus(): Promise<VoiceStatus | null> {
  try {
    const response = await apiClient.get<unknown>('/voice/status')
    return isVoiceStatus(response.data) ? response.data : null
  } catch {
    // Graceful degradation: voice is a bonus capability. Swallow the error.
    return null
  }
}

/**
 * Upload recorded audio and get a transcript back.
 * Resolves to `null` on 503 (no STT key), on any HTTP error, and on an empty
 * transcript, so the caller can simply leave the target field untouched.
 */
export async function transcribeAudio(
  audio: Blob,
  language?: string,
): Promise<TranscriptionResult | null> {
  try {
    const body = new FormData()
    // Give the part a filename with an extension — the backend sniffs the
    // container from it when the provider needs a hint.
    const extension = audio.type.includes('mp4') ? 'm4a' : audio.type.includes('ogg') ? 'ogg' : 'webm'
    body.append('file', audio, `recording.${extension}`)
    if (language) body.append('language', language)

    const response = await apiClient.post<{ text?: string; language?: string | null }>(
      '/voice/transcribe',
      body,
      { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 60_000 },
    )
    const text = typeof response.data?.text === 'string' ? response.data.text.trim() : ''
    if (!text) return null
    return { text, language: response.data?.language ?? null }
  } catch {
    // Graceful degradation: no key, no network, or provider failure.
    return null
  }
}

/**
 * Turn a decision + explanation into audio.
 * Resolves to `null` when TTS is unconfigured or the provider call fails, so
 * the "listen" button can simply stop offering itself.
 */
export async function synthesizeDecision(
  decision: string,
  explanation: string,
  language = 'en',
): Promise<Blob | null> {
  try {
    const response = await apiClient.post<Blob>(
      '/voice/synthesize',
      { decision, explanation, language },
      { responseType: 'blob', timeout: 60_000 },
    )
    const blob = response.data
    if (!(blob instanceof Blob) || blob.size === 0) return null
    return blob
  } catch {
    // Graceful degradation: silently drop back to the text-only decision.
    return null
  }
}

/** True when this browser can actually record audio at all (http:// hosts and
 *  older Safari cannot). Checked before any mic UI is offered. */
export function isRecordingSupported(): boolean {
  return (
    typeof window !== 'undefined'
    && typeof window.MediaRecorder !== 'undefined'
    && typeof navigator !== 'undefined'
    && Boolean(navigator.mediaDevices?.getUserMedia)
  )
}

import React, { useEffect, useRef, useState } from 'react'
import { Mic, Square } from 'lucide-react'
import { getVoiceStatus, isRecordingSupported, transcribeAudio } from '@/services/voice'

interface DictateButtonProps {
  /** Receives the transcript; the caller decides how to merge it into its field. */
  onText: (text: string) => void
}

/**
 * Small mic button that records, sends the audio to `/api/voice/transcribe`
 * (Sarvam) and hands back the text. Renders nothing when speech-to-text is not
 * configured or the browser cannot record, so it never leaves a dead button.
 */
export const DictateButton: React.FC<DictateButtonProps> = ({ onText }) => {
  const [enabled, setEnabled] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)

  useEffect(() => {
    let active = true
    void getVoiceStatus().then((status) => {
      if (active) setEnabled(Boolean(status?.stt.configured) && isRecordingSupported())
    })
    return () => {
      active = false
    }
  }, [])

  const start = async () => {
    setNotice(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      const chunks: Blob[] = []
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data)
      }
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop())
        setIsRecording(false)
        const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' })
        if (blob.size === 0) return
        setIsTranscribing(true)
        transcribeAudio(blob)
          .then((transcription) => {
            if (transcription) onText(transcription.text)
            else setNotice('Could not transcribe that — please type instead.')
          })
          .finally(() => setIsTranscribing(false))
      }
      recorderRef.current = recorder
      recorder.start()
      setIsRecording(true)
    } catch {
      setIsRecording(false)
      setNotice('Microphone unavailable — please type instead.')
    }
  }

  const stop = () => {
    try {
      recorderRef.current?.stop()
    } catch {
      setIsRecording(false)
    }
  }

  if (!enabled) return null

  return (
    <span className="inline-flex items-center gap-2">
      {notice && <span className="text-xs text-amber-700">{notice}</span>}
      <button
        type="button"
        onClick={() => (isRecording ? stop() : void start())}
        disabled={isTranscribing}
        className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors disabled:opacity-60 ${
          isRecording ? 'bg-red-100 text-red-700 hover:bg-red-200' : 'bg-primary-50 text-primary-700 hover:bg-primary-100'
        }`}
      >
        {isRecording ? <Square className="h-3 w-3 fill-current" /> : <Mic className="h-3 w-3" />}
        {isRecording ? 'Recording — tap to stop' : isTranscribing ? 'Transcribing…' : 'Dictate'}
      </button>
    </span>
  )
}

import { useEffect } from 'react'
import { motion } from 'framer-motion'
import MaskedHeading from '@/components/intro/MaskedHeading'

const SEEN_KEY = 'smartlend-intro-seen'
const HOLD_MS = 2800

// Brand gradient that drifts inside the letters (blue → sky → violet).
const FILL = [
  'radial-gradient(circle at 20% 30%, #60a5fa 0%, transparent 45%)',
  'radial-gradient(circle at 80% 70%, #8b5cf6 0%, transparent 45%)',
  'linear-gradient(120deg, #1d4ed8 0%, #0ea5e9 50%, #6366f1 100%)',
].join(', ')

/** True when the intro should play: first visit this session, motion allowed. */
export function shouldShowIntro(): boolean {
  try {
    if (sessionStorage.getItem(SEEN_KEY)) return false
  } catch {
    // Storage blocked: still fine to show it once per page load.
  }
  return !window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/**
 * Full-screen brand intro before the landing page: "SmartLend" revealed in a
 * masked gradient over the ShapeGrid background. Auto-continues after a short
 * hold; any click or key press skips it.
 */
export function IntroSplash({ onDone }: { onDone: () => void }) {
  useEffect(() => {
    try {
      sessionStorage.setItem(SEEN_KEY, '1')
    } catch {
      // Not remembered; it will simply play again next load.
    }
    const timer = window.setTimeout(onDone, HOLD_MS)
    const skip = () => onDone()
    window.addEventListener('keydown', skip)
    return () => {
      window.clearTimeout(timer)
      window.removeEventListener('keydown', skip)
    }
  }, [onDone])

  return (
    <motion.div
      role="presentation"
      onClick={onDone}
      initial={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 1.03 }}
      transition={{ duration: 0.5, ease: 'easeInOut' }}
      className="fixed inset-0 z-50 flex cursor-pointer flex-col items-center justify-center px-6"
    >
      <div className="w-full max-w-5xl">
        <MaskedHeading
          text="SmartLend"
          tag="h1"
          mediaType="gradient"
          background={FILL}
          reveal="rise"
          trigger="mount"
          duration={1.1}
          fillScale={1.6}
          drift={40}
          parallax={30}
          textScale={0.17}
          weight={800}
        />
      </div>
      <motion.p
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9, duration: 0.6 }}
        className="mt-4 text-center text-base text-neutral-500 sm:text-lg"
      >
        Smarter loan decisions. Backed by intelligence. Guided by trust.
      </motion.p>
      <motion.button
        type="button"
        onClick={(event) => {
          event.stopPropagation()
          onDone()
        }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
        className="absolute bottom-8 rounded-full border border-neutral-200 bg-white px-4 py-1.5 text-sm text-neutral-600 hover:text-neutral-900"
      >
        Skip intro
      </motion.button>
    </motion.div>
  )
}

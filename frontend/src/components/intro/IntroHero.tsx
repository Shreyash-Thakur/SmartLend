import { useRef } from 'react'
import { motion, useReducedMotion, useScroll, useTransform } from 'framer-motion'
import { ChevronDown } from 'lucide-react'
import MaskedHeading from '@/components/intro/MaskedHeading'
import { ThemeToggle } from '@/components/theme/ThemeToggle'

// Brand gradient that drifts inside the letters (blue → sky → violet).
const FILL = [
  'radial-gradient(circle at 20% 30%, #60a5fa 0%, transparent 45%)',
  'radial-gradient(circle at 80% 70%, #8b5cf6 0%, transparent 45%)',
  'linear-gradient(120deg, #1d4ed8 0%, #0ea5e9 50%, #6366f1 100%)',
].join(', ')

/**
 * Full-screen brand section at the top of the landing page: "SmartLend" in a
 * masked, drifting gradient over the ShapeGrid background. It stays until the
 * visitor scrolls; scrolling fades it out and brings the landing content up
 * from below (and scrolling back up brings it back).
 */
export function IntroHero({ continueTo }: { continueTo: string }) {
  const ref = useRef<HTMLElement>(null)
  const reduceMotion = useReducedMotion()
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end start'] })
  const opacity = useTransform(scrollYProgress, [0, 0.7], [1, 0])
  const scale = useTransform(scrollYProgress, [0, 1], [1, 0.92])
  const y = useTransform(scrollYProgress, [0, 1], [0, -60])

  const scrollOn = () => {
    document.getElementById(continueTo)?.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth' })
  }

  return (
    <section ref={ref} className="relative flex h-[100svh] flex-col items-center justify-center px-6">
      <div className="absolute right-4 top-4 sm:right-6">
        <ThemeToggle />
      </div>

      <motion.div style={reduceMotion ? undefined : { opacity, scale, y }} className="w-full max-w-5xl">
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
        <motion.p
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9, duration: 0.6 }}
          className="mt-4 text-center text-base text-neutral-500 sm:text-lg"
        >
          Smarter loan decisions. Backed by intelligence. Guided by trust.
        </motion.p>
      </motion.div>

      <motion.button
        type="button"
        onClick={scrollOn}
        aria-label="Scroll to continue"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.4 }}
        style={reduceMotion ? undefined : { opacity }}
        className="absolute bottom-8 flex flex-col items-center gap-1 text-sm text-neutral-500 hover:text-neutral-900"
      >
        <span>Scroll to explore</span>
        <motion.span
          animate={reduceMotion ? undefined : { y: [0, 6, 0] }}
          transition={{ repeat: Infinity, duration: 1.6, ease: 'easeInOut' }}
        >
          <ChevronDown className="h-5 w-5" />
        </motion.span>
      </motion.button>
    </section>
  )
}

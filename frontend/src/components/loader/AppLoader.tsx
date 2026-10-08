import { motion } from 'framer-motion'

/**
 * Full-screen branded loader for auth start-up and lazy route chunks.
 * Fades in after a short delay so fast loads never flash it. Transparent,
 * so the ShapeGrid background stays visible — never a blank screen.
 */
export function AppLoader({ label = 'Loading SmartLend…' }: { label?: string }) {
  return (
    <motion.div
      role="status"
      aria-live="polite"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.15, duration: 0.25 }}
      className="flex min-h-screen flex-col items-center justify-center gap-5"
    >
      <div className="relative h-14 w-14">
        <motion.span
          className="absolute inset-0 rounded-2xl border-2 border-neutral-200 border-t-primary-500"
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 0.9, ease: 'linear' }}
        />
        <span className="absolute inset-[7px] flex items-center justify-center rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 text-lg font-bold text-white shadow-md">
          S
        </span>
      </div>
      <p className="text-sm font-medium text-neutral-500">{label}</p>
    </motion.div>
  )
}

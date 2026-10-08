import type { PropsWithChildren } from 'react'
import { motion } from 'framer-motion'

/** Route-level enter/exit animation. Rendered once in App.tsx, keyed by path,
 *  inside <AnimatePresence mode="wait">. */
export function PageTransition({ children }: PropsWithChildren) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.22, ease: 'easeOut' }}
    >
      {children}
    </motion.div>
  )
}

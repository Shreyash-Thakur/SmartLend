import { useMemo } from 'react'
import type { ApplicationDecision } from '@/types/application'

export function useDecisionExplanation(decision?: ApplicationDecision) {
  return useMemo(() => {
    if (!decision) {
      return {
        confidenceLabel: 'Pending',
        confidencePercent: 0,
      }
    }

    return {
      confidenceLabel: decision.confidence.toUpperCase(),
      confidencePercent: Math.round((1 - decision.uncertainty) * 100),
    }
  }, [decision])
}

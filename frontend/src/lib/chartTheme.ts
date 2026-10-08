import { useThemeStore } from '@/store/themeStore'

/**
 * Colours for recharts / chart.js, which take literal colour strings (SVG
 * presentation attributes and canvas cannot read CSS variables). Semantic
 * series colours are identical in both themes; only chrome changes.
 */
export const SERIES = {
  approve: '#10b981',
  reject: '#ef4444',
  defer: '#f59e0b',
  brand: '#3b82f6',
  brandSoft: '#93c5fd',
  info: '#0ea5e9',
  violet: '#8b5cf6',
} as const

export function useChartTheme() {
  const isDark = useThemeStore((state) => state.theme) === 'dark'
  const grid = isDark ? '#262626' : '#e5e7eb'
  const axis = isDark ? '#a3a3a3' : '#6b7280'
  const surface = isDark ? '#141414' : '#ffffff'
  const text = isDark ? '#f5f5f5' : '#111827'

  return {
    isDark,
    grid,
    axis,
    surface,
    text,
    /** Spread onto <XAxis>/<YAxis>. */
    axisProps: { stroke: axis, tick: { fill: axis, fontSize: 12 }, tickLine: false, axisLine: { stroke: grid } },
    /** Spread onto <Tooltip>. */
    tooltipProps: {
      contentStyle: {
        background: surface,
        border: `1px solid ${grid}`,
        borderRadius: 10,
        color: text,
        boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
        fontSize: 13,
      },
      labelStyle: { color: text, fontWeight: 600 },
      itemStyle: { color: text },
      cursor: { fill: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(15,23,42,0.04)' },
    },
  }
}

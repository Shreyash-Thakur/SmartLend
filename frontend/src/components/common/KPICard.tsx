import React from 'react'

interface KPICardProps {
  label: string
  value: number | string
  format?: 'number' | 'currency' | 'percentage'
  trend?: {
    value: number
    direction: 'up' | 'down'
  }
  icon?: React.ComponentType<{ className?: string }>
  /** Colour of the icon tile. */
  tone?: 'blue' | 'green' | 'amber' | 'red' | 'violet' | 'neutral'
  /** Small line under the value. */
  hint?: string
  className?: string
}

const TONES: Record<NonNullable<KPICardProps['tone']>, string> = {
  blue: 'bg-blue-50 text-blue-600',
  green: 'bg-green-50 text-green-600',
  amber: 'bg-amber-50 text-amber-600',
  red: 'bg-red-50 text-red-600',
  violet: 'bg-violet-50 text-violet-600',
  neutral: 'bg-neutral-100 text-neutral-700',
}

export const KPICard: React.FC<KPICardProps> = ({
  label,
  value,
  format = 'number',
  trend,
  icon: Icon,
  tone = 'blue',
  hint,
  className = '',
}) => {
  const formatValue = (val: number | string): string => {
    if (typeof val === 'string') return val
    
    switch (format) {
      case 'currency':
        return `₹${val.toLocaleString('en-IN')}`
      case 'percentage':
        return `${val}%`
      default:
        return val.toLocaleString('en-IN')
    }
  }

  return (
    <div className={`rounded-2xl border border-neutral-200 bg-white p-5 shadow-[0_1px_2px_rgba(15,23,42,0.04)] ${className}`}>
      <div className="flex items-center gap-4">
        {Icon && (
          <span className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl ${TONES[tone]}`}>
            <Icon className="h-6 w-6" />
          </span>
        )}
        <div className="min-w-0">
          <p className="truncate text-sm text-neutral-500">{label}</p>
          <div className="mt-1 flex items-baseline gap-2">
            <p className="text-2xl font-semibold tracking-tight text-neutral-900">{formatValue(value)}</p>
            {trend && (
              <span className={`text-sm font-medium ${trend.direction === 'up' ? 'text-green-600' : 'text-red-600'}`}>
                {trend.direction === 'up' ? '↑' : '↓'} {Math.abs(trend.value)}%
              </span>
            )}
          </div>
          {hint && <p className="mt-0.5 truncate text-xs text-neutral-500">{hint}</p>}
        </div>
      </div>
    </div>
  )
}

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
  className?: string
}

export const KPICard: React.FC<KPICardProps> = ({
  label,
  value,
  format = 'number',
  trend,
  icon: Icon,
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
    <div className={`bg-white rounded-xl border border-neutral-200 p-6 shadow-md hover:shadow-lg transition-shadow ${className}`}>
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-sm font-medium text-neutral-600">{label}</p>
          <p className="text-3xl font-bold text-neutral-900 mt-2">{formatValue(value)}</p>
        </div>
        {Icon && <Icon className="w-12 h-12 text-primary-500 opacity-20" />}
      </div>

      {trend && (
        <div className={`flex items-center gap-2 text-sm font-medium ${trend.direction === 'up' ? 'text-green-600' : 'text-red-600'}`}>
          <span>{trend.direction === 'up' ? '↑' : '↓'}</span>
          <span>{Math.abs(trend.value)}%</span>
        </div>
      )}
    </div>
  )
}

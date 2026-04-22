import React from 'react'
import { BadgeProps } from '@/types/ui'

const statusConfig: Record<string, any> = {
  approved: {
    bgColor: 'bg-green-50',
    textColor: 'text-green-700',
    borderColor: 'border-green-200',
    icon: '✓',
  },
  rejected: {
    bgColor: 'bg-red-50',
    textColor: 'text-red-700',
    borderColor: 'border-red-200',
    icon: '✗',
  },
  deferred: {
    bgColor: 'bg-amber-50',
    textColor: 'text-amber-700',
    borderColor: 'border-amber-200',
    icon: '⏳',
  },
  pending: {
    bgColor: 'bg-blue-50',
    textColor: 'text-blue-700',
    borderColor: 'border-blue-200',
    icon: '⏵',
  },
  draft: {
    bgColor: 'bg-neutral-50',
    textColor: 'text-neutral-700',
    borderColor: 'border-neutral-200',
    icon: '✎',
  },
  submitted: {
    bgColor: 'bg-blue-50',
    textColor: 'text-blue-700',
    borderColor: 'border-blue-200',
    icon: '📤',
  },
  processing: {
    bgColor: 'bg-blue-50',
    textColor: 'text-blue-700',
    borderColor: 'border-blue-200',
    icon: '⏳',
  },
}

export const Badge: React.FC<BadgeProps> = ({ status, className = '' }) => {
  const config = statusConfig[status] || statusConfig['pending']
  const label = status.charAt(0).toUpperCase() + status.slice(1)

  return (
    <span
      className={`
        inline-flex items-center gap-1.5
        px-3 py-1.5 rounded-full
        border text-sm font-medium
        ${config.bgColor} ${config.textColor} ${config.borderColor}
        ${className}
      `}
    >
      <span className="text-xs">{config.icon}</span>
      {label}
    </span>
  )
}

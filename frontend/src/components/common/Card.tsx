import React from 'react'
import { CardProps } from '@/types/ui'

export const Card: React.FC<CardProps> = ({
  title,
  description,
  children,
  withGlass = true,
  className = '',
  footer,
}) => {
  return (
    <div
      className={`
        rounded-[28px] overflow-hidden
        border border-white/70
        bg-white/88
        ${withGlass ? 'backdrop-blur-xl' : ''}
        shadow-[0_20px_60px_rgba(15,23,42,0.08)] hover:shadow-[0_24px_70px_rgba(15,23,42,0.1)] transition-shadow duration-300
        ${className}
      `}
    >
      {(title || description) && (
        <div className="border-b border-neutral-200/80 bg-gradient-to-r from-white to-neutral-50 px-6 py-5">
          {title && <h3 className="text-lg font-semibold text-neutral-900">{title}</h3>}
          {description && <p className="mt-1 text-sm text-neutral-600">{description}</p>}
        </div>
      )}

      <div className="px-6 py-4">{children}</div>

      {footer && (
        <div className="px-6 py-4 border-t border-neutral-200 bg-neutral-50">
          {footer}
        </div>
      )}
    </div>
  )
}

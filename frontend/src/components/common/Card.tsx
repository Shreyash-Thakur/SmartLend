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
  // Callers often pass their own background (e.g. bg-amber-50 for warnings);
  // only apply the default surface when they don't, so the two never fight.
  const hasOwnBackground = /(^|\s)bg-(?!gradient|opacity|clip|blend)/.test(className)

  return (
    <div
      className={`
        overflow-hidden rounded-2xl
        border border-neutral-200
        ${hasOwnBackground ? '' : 'bg-white'}
        ${withGlass ? 'backdrop-blur-xl' : ''}
        shadow-[0_1px_2px_rgba(15,23,42,0.04)] transition-shadow duration-300 hover:shadow-[0_8px_30px_rgba(15,23,42,0.06)]
        ${className}
      `}
    >
      {(title || description) && (
        <div className="px-6 pt-5">
          {title && <h3 className="text-base font-semibold text-neutral-900">{title}</h3>}
          {description && <p className="mt-1 text-sm text-neutral-500">{description}</p>}
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

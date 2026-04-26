import React from 'react'
import { CardProps } from '@/types/ui'

export const Card: React.FC<CardProps> = ({
  title,
  description,
  children,
  withGlass = false,
  className = '',
  footer,
  level = 'medium',
}) => {
  const baseStyles = 'rounded-xl overflow-hidden bg-white transition-all duration-200'
  const levelStyles = level === 'medium'
    ? 'border-2 border-[#000000] shadow-[2px_2px_0px_#000000]'
    : 'border border-slate-200 shadow-none'

  return (
    <div className={`${baseStyles} ${levelStyles} ${className}`}>
      {(title || description) && (
        <div className="border-b border-slate-200 bg-slate-50 px-6 py-5">
          {title && <h3 className="text-lg font-bold text-[#0F172A]">{title}</h3>}
          {description && <p className="mt-1 text-sm text-slate-600">{description}</p>}
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

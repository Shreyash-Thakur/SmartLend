import React from 'react'
import { TextareaProps } from '@/types/ui'

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ label, error, hint, required, className = '', ...rest }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label className="block text-sm font-medium text-neutral-700 mb-2">
            {label}
            {required && <span className="text-red-500 ml-1">*</span>}
          </label>
        )}

        <textarea
          ref={ref}
          className={`
            w-full px-4 py-2
            border rounded-lg
            text-neutral-900 placeholder-neutral-400
            focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent
            disabled:bg-neutral-50 disabled:text-neutral-500 disabled:cursor-not-allowed
            transition-all duration-200 resize-vertical
            ${error ? 'border-red-500 focus:ring-red-500' : 'border-neutral-300'}
            ${className}
          `}
          {...rest}
        />

        {error && <p className="mt-1 text-sm text-red-500">{error}</p>}
        {hint && !error && <p className="mt-1 text-sm text-neutral-500">{hint}</p>}
      </div>
    )
  },
)
Textarea.displayName = 'Textarea'

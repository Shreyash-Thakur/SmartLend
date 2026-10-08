import React from 'react'
import { SelectProps } from '@/types/ui'
import GlideSelect from '@/components/common/GlideSelect'

// Theme-following colours: CSS variables flip with .dark, so no re-render is
// needed when the theme changes.
const COLORS = {
  surfaceColor: 'rgb(var(--surface))',
  highlightColor: 'rgb(var(--neutral-100))',
  textColor: 'rgb(var(--neutral-900))',
  accentColor: '#3b82f6',
}

/**
 * Form select backed by GlideSelect. Same props/contract as the old native
 * <select> wrapper: controlled `value`, `onChange(value: string)`. An option
 * with an empty value (e.g. "Select...") becomes the placeholder text rather
 * than a pickable row.
 */
export const Select: React.FC<SelectProps> = ({
  label,
  error,
  options = [],
  placeholder,
  required,
  className = '',
  onChange,
  value,
  disabled,
}) => {
  const emptyOption = options.find((option) => String(option.value) === '')
  const items = options
    .filter((option) => String(option.value) !== '')
    .map((option) => ({ value: String(option.value), label: option.label }))

  return (
    <div className="w-full">
      {label && (
        <span className="mb-2 block text-sm font-medium text-neutral-700">
          {label}
          {required && <span className="ml-1 text-red-500">*</span>}
        </span>
      )}

      <GlideSelect
        options={items}
        value={value === undefined || value === null ? '' : String(value)}
        onChange={(next) => onChange?.(next)}
        placeholder={placeholder ?? emptyOption?.label ?? 'Select…'}
        ariaLabel={label ?? placeholder ?? 'Select'}
        disabled={Boolean(disabled)}
        size="lg"
        radius={14}
        fullWidth
        showTags={false}
        className={className}
        triggerClassName={`border px-4 font-normal focus-visible:ring-2 focus-visible:ring-primary-500 ${
          error ? 'border-red-500' : 'border-neutral-300'
        }`}
        {...COLORS}
      />

      {error && <p className="mt-1 text-sm text-red-500">{error}</p>}
    </div>
  )
}

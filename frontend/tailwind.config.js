/** Grey scale backed by CSS variables (see globals.css) so it flips in dark mode. */
const varScale = (name) =>
  Object.fromEntries(
    [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950].map((shade) => [
      shade,
      `rgb(var(--${name}-${shade}) / <alpha-value>)`,
    ]),
  )

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Brand colour: blue, matching the reference light theme. Full scale
        // (the old green one only defined 50/100/500/600/900, so classes like
        // primary-700 silently rendered nothing).
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
          950: '#172554',
        },
        accent: {
          50: '#f0f9ff',
          500: '#0ea5e9',
        },
        neutral: varScale('neutral'),
        gray: varScale('gray'),
        // Fixed dark surface that does not flip with the theme.
        ink: '#111827',
        success: '#10b981',
        warning: '#f59e0b',
        error: '#ef4444',
        info: '#3b82f6',
        deferred: '#ec4899',
      },
      borderRadius: {
        sm: '8px',
        md: '12px',
        lg: '16px',
        xl: '24px',
      },
      boxShadow: {
        sm: '0 1px 2px rgba(0,0,0,0.05)',
        md: '0 4px 6px rgba(0,0,0,0.07)',
        lg: '0 10px 15px rgba(0,0,0,0.10)',
      },
      fontFamily: {
        sans: ['Inter', 'SF Pro Display', '-apple-system', 'sans-serif'],
      },
      backgroundImage: {
        'hero-grid':
          'radial-gradient(circle at top left, rgba(14,165,233,0.18), transparent 30%), radial-gradient(circle at bottom right, rgba(22,163,74,0.2), transparent 35%)',
      },
    },
  },
  plugins: [],
}

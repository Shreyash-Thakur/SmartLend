import { create } from 'zustand'

export type Theme = 'light' | 'dark'

// Same key the inline script in index.html reads before React mounts, so the
// saved theme is applied on first paint with no light→dark flash.
const STORAGE_KEY = 'smartlend-theme'

function readSavedTheme(): Theme {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'dark' ? 'dark' : 'light'
  } catch {
    // Storage blocked (private mode etc.) — default to light.
    return 'light'
  }
}

function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle('dark', theme === 'dark')
  try {
    localStorage.setItem(STORAGE_KEY, theme)
  } catch {
    // Theme still applies for this session; it just won't be remembered.
  }
}

interface ThemeStore {
  theme: Theme
  toggleTheme: () => void
}

export const useThemeStore = create<ThemeStore>((set, get) => ({
  theme: readSavedTheme(),
  toggleTheme: () => {
    const next: Theme = get().theme === 'dark' ? 'light' : 'dark'
    applyTheme(next)
    set({ theme: next })
  },
}))

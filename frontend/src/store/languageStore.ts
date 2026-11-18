import { create } from 'zustand'

/** Customer-side display language. Org pages always stay in English. */
export type Language = 'en' | 'hi' | 'mr'

const STORAGE_KEY = 'smartlend-lang'

function readSavedLanguage(): Language {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    return saved === 'hi' || saved === 'mr' ? saved : 'en'
  } catch {
    return 'en'
  }
}

interface LanguageStore {
  language: Language
  setLanguage: (language: Language) => void
}

export const useLanguageStore = create<LanguageStore>((set) => ({
  language: readSavedLanguage(),
  setLanguage: (language) => {
    try {
      localStorage.setItem(STORAGE_KEY, language)
    } catch {
      // Still applies for this session; it just won't be remembered.
    }
    set({ language })
  },
}))

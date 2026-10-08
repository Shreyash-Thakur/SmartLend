import { create } from 'zustand'

interface UiStore {
  activeTab: 'all' | 'deferred'
  statusFilter: string
  setActiveTab: (tab: 'all' | 'deferred') => void
  setStatusFilter: (status: string) => void
  /** Title of the current page, shown in the app shell's top bar. */
  pageTitle: string
  setPageTitle: (title: string) => void
}

export const useUiStore = create<UiStore>((set) => ({
  activeTab: 'all',
  statusFilter: 'all',
  setActiveTab: (activeTab) => set({ activeTab }),
  setStatusFilter: (statusFilter) => set({ statusFilter }),
  pageTitle: '',
  setPageTitle: (pageTitle) => set({ pageTitle }),
}))

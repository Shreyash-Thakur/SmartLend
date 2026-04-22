import { create } from 'zustand'

interface UiStore {
  activeTab: 'all' | 'deferred'
  statusFilter: string
  setActiveTab: (tab: 'all' | 'deferred') => void
  setStatusFilter: (status: string) => void
}

export const useUiStore = create<UiStore>((set) => ({
  activeTab: 'all',
  statusFilter: 'all',
  setActiveTab: (activeTab) => set({ activeTab }),
  setStatusFilter: (statusFilter) => set({ statusFilter }),
}))

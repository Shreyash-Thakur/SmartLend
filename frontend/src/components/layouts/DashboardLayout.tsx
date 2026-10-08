import React, { useEffect } from 'react'
import { useUiStore } from '@/store/uiStore'

interface DashboardLayoutProps {
  children: React.ReactNode
  title?: string
  /** Kept for existing call sites; the shell now derives navigation from the
   *  signed-in user's role. */
  role?: 'customer' | 'organization'
}

/**
 * Page wrapper inside the AppShell (sidebar + top bar live there, mounted once
 * in App.tsx). This only publishes the page title to the top-bar breadcrumb.
 */
export const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children, title }) => {
  const setPageTitle = useUiStore((state) => state.setPageTitle)

  useEffect(() => {
    setPageTitle(title ?? '')
  }, [title, setPageTitle])

  return <>{children}</>
}

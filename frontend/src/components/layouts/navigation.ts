import type { LucideIcon } from 'lucide-react'
import {
  BrainCircuit,
  ClipboardCheck,
  FilePlus2,
  History,
  LayoutDashboard,
  MapPinned,
} from 'lucide-react'

export interface NavItem {
  label: string
  /** Path plus optional query string. */
  to: string
  icon: LucideIcon
}

export interface NavGroup {
  heading: string
  items: NavItem[]
}

export const ORG_NAV: NavGroup[] = [
  { heading: 'Overview', items: [{ label: 'Dashboard', to: '/dashboard/org', icon: LayoutDashboard }] },
  { heading: 'Decisions', items: [{ label: 'Review Queue', to: '/review', icon: ClipboardCheck }] },
  {
    heading: 'Insights',
    items: [
      { label: 'Model Analysis', to: '/dashboard/models', icon: BrainCircuit },
      { label: 'Geo Analytics', to: '/analytics/geo', icon: MapPinned },
    ],
  },
]

export const CUSTOMER_NAV: NavGroup[] = [
  { heading: 'Overview', items: [{ label: 'Dashboard', to: '/dashboard/customer', icon: LayoutDashboard }] },
  {
    heading: 'Applications',
    items: [
      { label: 'New Application', to: '/dashboard/customer/new', icon: FilePlus2 },
      { label: 'My Applications', to: '/dashboard/customer?view=history', icon: History },
    ],
  },
]

/** Exact path + query match, so "/dashboard/customer" and
 *  "/dashboard/customer?view=history" are not both highlighted. */
export function isNavItemActive(item: NavItem, pathname: string, search: string): boolean {
  const [path, query = ''] = item.to.split('?')
  if (path !== pathname) return false
  const current = new URLSearchParams(search).get('view') ?? ''
  const wanted = new URLSearchParams(query).get('view') ?? ''
  return current === wanted
}

/** Paths rendered inside the app shell (sidebar + top bar). */
export function isShellPath(pathname: string): boolean {
  return /^\/(dashboard|review|analytics)(\/|$)/.test(pathname)
}

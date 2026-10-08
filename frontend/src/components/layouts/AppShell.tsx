import { Suspense, useEffect, useState } from 'react'
import { Link, useLocation, useNavigate, useOutlet } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { ChevronRight, LogOut, Menu, PanelLeft, X } from 'lucide-react'
import { ThemeToggle } from '@/components/theme/ThemeToggle'
import { AppLoader } from '@/components/loader/AppLoader'
import { useAuth } from '@/hooks/useAuth'
import { useAuthStore } from '@/store/authStore'
import { useUiStore } from '@/store/uiStore'
import { getServingModelInfo } from '@/services/applications'
import type { ServingModelInfo } from '@/services/applications'
import { CUSTOMER_NAV, ORG_NAV, isNavItemActive } from '@/components/layouts/navigation'
import type { NavGroup } from '@/components/layouts/navigation'

const COLLAPSED_KEY = 'smartlend-sidebar-collapsed'

function readCollapsed(): boolean {
  try {
    return localStorage.getItem(COLLAPSED_KEY) === '1'
  } catch {
    return false
  }
}

/**
 * Persistent app frame for every signed-in page: sidebar (drawer on mobile),
 * top bar with breadcrumb + theme toggle, and an animated outlet. Mounted once
 * as a layout route in App.tsx, so only the page content transitions.
 */
export function AppShell() {
  const location = useLocation()
  const outlet = useOutlet()
  const role = useAuthStore((state) => state.role)
  const pageTitle = useUiStore((state) => state.pageTitle)
  const isOrg = role === 'org'
  const nav = isOrg ? ORG_NAV : CUSTOMER_NAV

  const [collapsed, setCollapsed] = useState(readCollapsed)
  const [mobileOpen, setMobileOpen] = useState(false)

  const toggleCollapsed = () => {
    setCollapsed((value) => {
      try {
        localStorage.setItem(COLLAPSED_KEY, value ? '0' : '1')
      } catch {
        // Not remembered across reloads; harmless.
      }
      return !value
    })
  }

  // Close the mobile drawer whenever the route changes.
  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname, location.search])

  return (
    <div className="min-h-screen">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[60] focus:rounded-lg focus:bg-white focus:px-4 focus:py-2"
      >
        Skip to main content
      </a>

      {/* Desktop sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 hidden border-r border-neutral-200 bg-white/80 backdrop-blur-xl transition-[width] duration-200 lg:block ${
          collapsed ? 'w-[76px]' : 'w-64'
        }`}
      >
        <SidebarContent nav={nav} collapsed={collapsed} isOrg={isOrg} />
      </aside>

      {/* Mobile drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              className="fixed inset-0 z-40 bg-black/50 lg:hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMobileOpen(false)}
            />
            <motion.aside
              className="fixed inset-y-0 left-0 z-50 w-72 border-r border-neutral-200 bg-white lg:hidden"
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'tween', duration: 0.22 }}
            >
              <button
                type="button"
                onClick={() => setMobileOpen(false)}
                aria-label="Close menu"
                className="absolute right-3 top-4 rounded-lg p-2 text-neutral-500 hover:bg-neutral-100"
              >
                <X className="h-4 w-4" />
              </button>
              <SidebarContent nav={nav} collapsed={false} isOrg={isOrg} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      <div className={`transition-[padding] duration-200 ${collapsed ? 'lg:pl-[76px]' : 'lg:pl-64'}`}>
        {/* Top bar */}
        <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-neutral-200 bg-white/75 px-4 backdrop-blur-xl sm:px-6">
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            aria-label="Open menu"
            className="rounded-lg p-2 text-neutral-600 hover:bg-neutral-100 lg:hidden"
          >
            <Menu className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={toggleCollapsed}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="hidden rounded-lg p-2 text-neutral-600 hover:bg-neutral-100 lg:block"
          >
            <PanelLeft className="h-4 w-4" />
          </button>
          <span className="hidden h-5 w-px bg-neutral-200 sm:block" />
          <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-1.5 text-sm">
            <Link
              to={isOrg ? '/dashboard/org' : '/dashboard/customer'}
              className="shrink-0 text-neutral-500 hover:text-neutral-900"
            >
              Home
            </Link>
            {pageTitle && (
              <>
                <ChevronRight className="h-3.5 w-3.5 shrink-0 text-neutral-400" />
                <span className="truncate font-medium text-neutral-900">{pageTitle}</span>
              </>
            )}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <ThemeToggle />
          </div>
        </header>

        <main id="main-content" className="mx-auto max-w-[1400px] px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          <AnimatePresence mode="wait" onExitComplete={() => window.scrollTo(0, 0)}>
            <motion.div
              key={location.pathname + location.search}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
            >
              {/* Own Suspense boundary so a lazy page chunk shows the loader
                  inside the shell instead of replacing the whole frame. */}
              <Suspense fallback={<AppLoader />}>{outlet}</Suspense>
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  )
}

function SidebarContent({ nav, collapsed, isOrg }: { nav: NavGroup[]; collapsed: boolean; isOrg: boolean }) {
  const location = useLocation()

  return (
    <div className="flex h-full flex-col">
      <Link
        to={isOrg ? '/dashboard/org' : '/dashboard/customer'}
        className={`flex h-16 shrink-0 items-center gap-3 ${collapsed ? 'justify-center px-0' : 'px-5'}`}
      >
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-[10px] bg-gradient-to-br from-primary-500 to-accent-500 font-bold text-white shadow-md">
          S
        </span>
        {!collapsed && (
          <span className="min-w-0">
            <span className="block text-[15px] font-semibold leading-tight text-neutral-900">SmartLend</span>
            <span className="block text-xs text-neutral-500">{isOrg ? 'Lender Console' : 'Loan Applications'}</span>
          </span>
        )}
      </Link>

      <nav className="flex-1 space-y-6 overflow-y-auto px-3 py-4" aria-label="Main">
        {nav.map((group) => (
          <div key={group.heading}>
            {!collapsed && (
              <p className="mb-2 px-3 text-xs font-medium text-neutral-500">{group.heading}</p>
            )}
            <ul className="space-y-1">
              {group.items.map((item) => {
                const active = isNavItemActive(item, location.pathname, location.search)
                const Icon = item.icon
                return (
                  <li key={item.to}>
                    <Link
                      to={item.to}
                      title={collapsed ? item.label : undefined}
                      aria-current={active ? 'page' : undefined}
                      className={`flex items-center gap-3 rounded-[10px] py-2 text-sm transition-colors ${
                        collapsed ? 'justify-center px-0' : 'px-3'
                      } ${
                        active
                          ? 'bg-neutral-100 font-medium text-neutral-900'
                          : 'text-neutral-600 hover:bg-neutral-100 hover:text-neutral-900'
                      }`}
                    >
                      <Icon className="h-[18px] w-[18px] shrink-0" />
                      {!collapsed && <span className="truncate">{item.label}</span>}
                    </Link>
                  </li>
                )
              })}
            </ul>
          </div>
        ))}
      </nav>

      <div className="space-y-3 border-t border-neutral-200 p-3">
        {isOrg && !collapsed && <ServingModelBadge />}
        <UserBlock collapsed={collapsed} />
      </div>
    </div>
  )
}

/** Always-visible serving-model identity for org users. Hidden (null) when
 *  the backend is unreachable — never a stale placeholder value. */
function ServingModelBadge() {
  const [info, setInfo] = useState<ServingModelInfo | null>(null)
  useEffect(() => {
    let cancelled = false
    void getServingModelInfo().then((result) => {
      if (!cancelled) setInfo(result)
    })
    return () => {
      cancelled = true
    }
  }, [])

  if (!info || !info.artifact) return null
  return (
    <div title={info.engine_version} className="rounded-[10px] border border-neutral-200 px-3 py-2 text-xs">
      <p className="text-neutral-500">Serving model</p>
      <p className="mt-0.5 truncate font-medium text-neutral-900">
        {info.model_name} · {info.artifact}
      </p>
    </div>
  )
}

function UserBlock({ collapsed }: { collapsed: boolean }) {
  const navigate = useNavigate()
  const { logout } = useAuth()
  const user = useAuthStore((state) => state.user)
  const name = user?.displayName || 'Guest'
  const initial = name.charAt(0).toUpperCase()

  const handleLogout = async () => {
    await logout()
    navigate('/')
  }

  return (
    <div className={`flex items-center gap-3 ${collapsed ? 'flex-col' : ''}`}>
      {user?.photoURL ? (
        <img src={user.photoURL} alt="" referrerPolicy="no-referrer" className="h-9 w-9 shrink-0 rounded-full object-cover" />
      ) : (
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-neutral-200 text-sm font-semibold text-neutral-700">
          {initial}
        </span>
      )}
      {!collapsed && (
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-neutral-900">{name}</p>
          {user?.email && <p className="truncate text-xs text-neutral-500">{user.email}</p>}
        </div>
      )}
      <button
        type="button"
        onClick={() => void handleLogout()}
        title="Log out"
        aria-label="Log out"
        className="rounded-lg p-2 text-neutral-500 transition-colors hover:bg-red-50 hover:text-red-600"
      >
        <LogOut className="h-4 w-4" />
      </button>
    </div>
  )
}

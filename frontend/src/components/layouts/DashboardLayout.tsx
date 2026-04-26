import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { LogOut } from 'lucide-react'
import { Button } from '@/components/common'
import { PageTransition } from '@/components/layouts/PageTransition'
import { useAuth } from '@/hooks/useAuth'

interface DashboardLayoutProps {
  children: React.ReactNode
  title?: string
  role?: 'customer' | 'organization'
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  children,
  title,
  role = 'customer',
}) => {
  const navigate = useNavigate()
  const { logout } = useAuth()

  const handleLogout = async () => {
    await logout()
    navigate('/')
  }

  return (
    <div className="min-h-screen bg-slate-50 bg-hero-grid">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-white focus:px-4 focus:py-2"
      >
        Skip to main content
      </a>

      <header className="sticky top-0 z-40 border-b-4 border-black bg-[#F1F6F1]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => navigate('/')}
                className="flex items-center gap-3 text-left"
              >
                <div className="w-10 h-10 bg-[#6E61FF] border-2 border-black shadow-[2px_2px_0px_#000000] rounded flex items-center justify-center transition-transform hover:-translate-y-1">
                <span className="text-white font-black text-xl">S</span>
                </div>
                <div>
                  <h1 className="text-lg font-black text-black">SmartLend</h1>
                  {title && <p className="text-xs font-bold text-black opacity-80">{title}</p>}
                </div>
              </button>
            </div>

            <div className="flex items-center gap-3">
              <nav className="hidden items-center gap-2 md:flex">
                {role === 'customer' ? (
                  <DashboardNavLink to="/dashboard/customer" label="Dashboard" />
                ) : (
                  <>
                    <DashboardNavLink to="/dashboard/org" label="Dashboard" />
                    <DashboardNavLink to="/review" label="Review" />
                    <DashboardNavLink to="/dashboard/models" label="Model Analysis" />
                  </>
                )}
              </nav>
              {role !== 'customer' && (
                <Button variant="secondary" size="sm" onClick={() => navigate('/')}>
                  Back to Home
                </Button>
              )}
              <button
                type="button"
                onClick={() => void handleLogout()}
                title="Log out"
                className="flex items-center gap-2 rounded px-3 py-2 text-sm font-bold text-black bg-[#FD9745] border-2 border-black shadow-[2px_2px_0px_#000000] hover:-translate-y-0.5 hover:shadow-[4px_4px_0px_#000000] transition-all"
              >
                <LogOut className="h-4 w-4" />
                <span className="hidden sm:inline">Log Out</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <PageTransition>
        <main id="main-content" className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
          {children}
        </main>
      </PageTransition>

      <footer className="bg-neutral-900 text-neutral-400 py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center text-sm">
            <p>&copy; 2026 SmartLend. All rights reserved.</p>
            <p className="mt-2">Intelligent Loan Assessment Platform</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

function DashboardNavLink({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-4 py-2 text-sm font-bold transition-all border-2 rounded ${
          isActive
            ? 'bg-[#B0F0DA] text-black border-black shadow-[2px_2px_0px_#000000]'
            : 'bg-transparent text-black border-transparent hover:border-black hover:bg-white hover:shadow-[2px_2px_0px_#000000] hover:-translate-y-0.5'
        }`
      }
    >
      {label}
    </NavLink>
  )
}


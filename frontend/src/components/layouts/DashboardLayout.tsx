import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { Button } from '@/components/common'
import { PageTransition } from '@/components/layouts/PageTransition'

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

  return (
    <div className="min-h-screen bg-neutral-50 bg-hero-grid">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-white focus:px-4 focus:py-2"
      >
        Skip to main content
      </a>

      <header className="sticky top-0 z-40 border-b border-white/40 bg-white/80 shadow-sm backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => navigate('/')}
                className="flex items-center gap-3 text-left"
              >
                <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-accent-500 rounded-lg flex items-center justify-center shadow-md">
                <span className="text-white font-bold">S</span>
                </div>
                <div>
                  <h1 className="text-lg font-bold text-neutral-900">SmartLend</h1>
                  {title && <p className="text-xs text-neutral-500">{title}</p>}
                </div>
              </button>
            </div>

            <div className="flex items-center gap-4">
              <nav className="hidden items-center gap-2 md:flex">
                <NavLink
                  to="/dashboard/customer"
                  className={({ isActive }) =>
                    `rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-primary-100 text-primary-900'
                        : 'text-neutral-600 hover:bg-white hover:text-neutral-900'
                    }`
                  }
                >
                  Customer
                </NavLink>
                <NavLink
                  to="/dashboard/org"
                  className={({ isActive }) =>
                    `rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-primary-100 text-primary-900'
                        : 'text-neutral-600 hover:bg-white hover:text-neutral-900'
                    }`
                  }
                >
                  Organization
                </NavLink>
                <NavLink
                  to="/dashboard/models"
                  className={({ isActive }) =>
                    `rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-primary-100 text-primary-900'
                        : 'text-neutral-600 hover:bg-white hover:text-neutral-900'
                    }`
                  }
                >
                  Model Analysis
                </NavLink>
              </nav>
              <span className="hidden text-sm text-neutral-600 md:inline capitalize">{role} dashboard</span>
              <Button variant="secondary" size="sm" onClick={() => navigate('/')}>
                Back to Home
              </Button>
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

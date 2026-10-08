import { lazy, Suspense, useEffect } from 'react'
import { AnimatePresence } from 'framer-motion'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { ThemeToggle } from '@/components/theme/ThemeToggle'
import { AppBackground } from '@/components/background/AppBackground'
import { AppLoader } from '@/components/loader/AppLoader'
import { PageTransition } from '@/components/layouts/PageTransition'
import { AppShell } from '@/components/layouts/AppShell'
import { isShellPath } from '@/components/layouts/navigation'
import { useAuth } from '@/hooks/useAuth'
import { useAuthStore } from '@/store/authStore'

const AuthPage = lazy(async () => ({
  default: (await import('@/pages/AuthPage')).AuthPage,
}))
const Landing = lazy(async () => ({
  default: (await import('@/pages/Landing')).Landing,
}))
const CustomerDashboard = lazy(async () => ({
  default: (await import('@/pages/Dashboard.customer')).CustomerDashboard,
}))
const CustomerNewApplication = lazy(async () => ({
  default: (await import('@/pages/CustomerNewApplication')).CustomerNewApplication,
}))
const OrganizationDashboard = lazy(async () => ({
  default: (await import('@/pages/Dashboard.org')).OrganizationDashboard,
}))
const ModelAnalysisDashboard = lazy(async () => ({
  default: (await import('@/pages/Dashboard.models')).ModelAnalysisDashboard,
}))
const ApplicationReview = lazy(async () => ({
  default: (await import('@/pages/ApplicationReview')).ApplicationReview,
}))
const ReviewPage = lazy(async () => ({
  default: (await import('@/pages/ReviewPage')).ReviewPage,
}))
const GeoAnalytics = lazy(async () => ({
  default: (await import('@/pages/GeoAnalytics')).GeoAnalytics,
}))

const ProtectedRoute = ({ children, requiredRole }: { children: React.ReactNode; requiredRole?: 'customer' | 'org' }) => {
  const { isAuthenticated, role } = useAuth()

  if (!isAuthenticated) {
    return <Navigate to="/auth" replace />
  }

  if (!role) {
    return <Navigate to="/auth" replace />
  }

  if (requiredRole && role !== requiredRole) {
    return <Navigate to={role === 'customer' ? '/dashboard/customer' : '/dashboard/org'} replace />
  }

  return <>{children}</>
}

export default function App() {
  const { isAuthenticated, loading, role } = useAuth()
  const { initializeAuth } = useAuthStore()
  const location = useLocation()
  const { pathname } = location

  useEffect(() => {
    initializeAuth()
  }, [initializeAuth])

  // The background is mounted once, outside every branch, so the grid keeps
  // animating across loading states and route changes instead of restarting.
  return (
    <>
      <AppBackground />
      {/* /auth has no header of its own, so the theme toggle floats there. */}
      {pathname === '/auth' && (
        <div className="fixed right-4 top-4 z-50">
          <ThemeToggle />
        </div>
      )}
      {loading ? (
        <AppLoader />
      ) : (
        <AnimatePresence mode="wait" onExitComplete={() => window.scrollTo(0, 0)}>
          {/* Shell pages share one key so the shell persists across them and
              animates only its own content; other pages transition whole. */}
          <PageTransition key={isShellPath(pathname) ? 'shell' : pathname}>
            <Suspense fallback={<AppLoader />}>
              <Routes location={location}>
                <Route path="/auth" element={<AuthPage />} />
                <Route
                  path="/"
                  element={
                    isAuthenticated ? (
                      role ? (
                        <Navigate to={role === 'org' ? '/dashboard/org' : '/dashboard/customer'} />
                      ) : (
                        <Navigate to="/auth" replace />
                      )
                    ) : (
                      <Landing />
                    )
                  }
                />
                {/* Signed-in pages share one persistent shell (sidebar + top bar). */}
                <Route
                  element={
                    <ProtectedRoute>
                      <AppShell />
                    </ProtectedRoute>
                  }
                >
                  <Route
                    path="/dashboard/customer"
                    element={
                      <ProtectedRoute requiredRole="customer">
                        <CustomerDashboard />
                      </ProtectedRoute>
                    }
                  />
                  <Route
                    path="/dashboard/customer/new"
                    element={
                      <ProtectedRoute requiredRole="customer">
                        <CustomerNewApplication />
                      </ProtectedRoute>
                    }
                  />
                  <Route
                    path="/dashboard/org"
                    element={
                      <ProtectedRoute requiredRole="org">
                        <OrganizationDashboard />
                      </ProtectedRoute>
                    }
                  />
                  <Route
                    path="/dashboard/models"
                    element={
                      <ProtectedRoute requiredRole="org">
                        <ModelAnalysisDashboard />
                      </ProtectedRoute>
                    }
                  />
                  <Route
                    path="/analytics/geo"
                    element={
                      <ProtectedRoute requiredRole="org">
                        <GeoAnalytics />
                      </ProtectedRoute>
                    }
                  />
                  <Route
                    path="/review"
                    element={
                      <ProtectedRoute requiredRole="org">
                        <ReviewPage />
                      </ProtectedRoute>
                    }
                  />
                  <Route
                    path="/review/:applicationId"
                    element={
                      <ProtectedRoute>
                        <ApplicationReview />
                      </ProtectedRoute>
                    }
                  />
                </Route>
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Suspense>
          </PageTransition>
        </AnimatePresence>
      )}
    </>
  )
}

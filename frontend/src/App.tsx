import { lazy, Suspense, useEffect } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
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

  useEffect(() => {
    initializeAuth()
  }, [initializeAuth])

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-neutral-50 text-neutral-600">
        Loading SmartLend...
      </div>
    )
  }

  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-neutral-50 text-neutral-600">
          Loading SmartLend...
        </div>
      }
    >
      <Routes>
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
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  )
}

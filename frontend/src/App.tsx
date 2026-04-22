import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'

const Landing = lazy(async () => ({
  default: (await import('@/pages/Landing')).Landing,
}))
const CustomerDashboard = lazy(async () => ({
  default: (await import('@/pages/Dashboard.customer')).CustomerDashboard,
}))
const OrganizationDashboard = lazy(async () => ({
  default: (await import('@/pages/Dashboard.org')).OrganizationDashboard,
}))
const ApplicationReview = lazy(async () => ({
  default: (await import('@/pages/ApplicationReview')).ApplicationReview,
}))

export default function App() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-neutral-50 text-neutral-600">
          Loading SmartLend...
        </div>
      }
    >
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/dashboard/customer" element={<CustomerDashboard />} />
        <Route path="/dashboard/org" element={<OrganizationDashboard />} />
        <Route path="/review/:applicationId" element={<ApplicationReview />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  )
}

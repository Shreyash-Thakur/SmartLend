import React from 'react'
import { useNavigate } from 'react-router-dom'
import { ChevronLeft } from 'lucide-react'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Button, Card } from '@/components/common'
import { LoanApplicationForm } from '@/components/forms'
import { useApplicationData } from '@/hooks/useApplicationData'
import { useAuth } from '@/hooks/useAuth'
import { trackEvent } from '@/services/analytics'

export const CustomerNewApplication: React.FC = () => {
  const navigate = useNavigate()
  const { user } = useAuth()
  const { addApplication, uploadDocument, isLoading, error } = useApplicationData({ scope: 'customer', applicantId: user?.uid })

  const handleSubmitApplication = async (data: Parameters<typeof addApplication>[0], file?: File) => {
    const application = await addApplication({
      ...data,
      applicantId: user?.uid,
    })

    if (file) {
      await uploadDocument(application.id, file)
    }

    trackEvent('application_submitted', { applicationId: application.id })
    navigate('/dashboard/customer?view=history')
  }

  return (
    <DashboardLayout title="Customer Dashboard" role="customer">
      <section className="mb-6">
        <Button
          variant="ghost"
          leftIcon={<ChevronLeft className="h-4 w-4" />}
          onClick={() => navigate('/dashboard/customer')}
        >
          Back to dashboard
        </Button>
      </section>

      {error && (
        <section className="mb-6">
          <Card className="border-red-200 bg-red-50">
            <p className="text-red-700">{error}</p>
          </Card>
        </section>
      )}

      <section>
        <Card
          title="New Loan Application"
          description="Fill all details carefully. You will see a complete final review before submitting."
          className="rounded-[36px] border-white/80"
        >
          <LoanApplicationForm
            onSubmit={handleSubmitApplication}
            isLoading={isLoading}
            isMultiStep
          />
        </Card>
      </section>
    </DashboardLayout>
  )
}

import React from 'react'
import { KPICard } from '@/components/common'
import type { DashboardMetrics } from '@/types/api'

interface MetricsGridProps {
  metrics: DashboardMetrics
}

export const MetricsGrid: React.FC<MetricsGridProps> = ({ metrics }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <KPICard
        label="Total Applications"
        value={metrics.totalApplications}
        format="number"
        trend={{
          value: 12,
          direction: 'up',
        }}
      />
      <KPICard label="Approved" value={metrics.approved} format="number" />
      <KPICard label="Rejected" value={metrics.rejected} format="number" />
      <KPICard label="Deferred" value={metrics.deferred} format="number" />
    </div>
  )
}

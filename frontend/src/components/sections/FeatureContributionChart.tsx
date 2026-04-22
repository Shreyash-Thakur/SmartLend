import React from 'react'
import { Card } from '@/components/common'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { FeatureContribution } from '@/types/application'
import type { FeatureContributionChartProps } from '@/types/ui'

export const FeatureContributionChart: React.FC<FeatureContributionChartProps> = ({
  features,
  maxFeatures = 5,
}) => {
  const topFeatures = [...features]
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
    .slice(0, maxFeatures)
    .map((feature) => ({
      ...feature,
      displayImpact: Number(feature.impact.toFixed(2)),
    }))

  return (
    <Card title="Feature Importance Analysis" className="mt-6">
      <div className="space-y-6">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={topFeatures} layout="vertical" margin={{ left: 12, right: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis type="number" domain={[-1, 1]} />
              <YAxis dataKey="name" type="category" width={110} />
              <Tooltip
                formatter={(value: number) => [value, 'Impact']}
                labelFormatter={(label) => `Feature: ${label}`}
              />
              <Bar dataKey="displayImpact" radius={[6, 6, 6, 6]}>
                {topFeatures.map((feature: FeatureContribution) => (
                  <Cell
                    key={feature.name}
                    fill={feature.impact >= 0 ? '#16a34a' : '#ef4444'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          {topFeatures.map((feature) => (
            <div key={feature.name} className="rounded-lg border border-neutral-200 bg-neutral-50 p-3">
              <div className="flex items-center justify-between">
                <p className="font-medium text-neutral-900">{feature.name}</p>
                <span className={feature.impact >= 0 ? 'text-green-600' : 'text-red-600'}>
                  {feature.impact >= 0 ? '+' : ''}
                  {feature.impact.toFixed(2)}
                </span>
              </div>
              <p className="mt-1 text-sm text-neutral-600">
                Value: {typeof feature.value === 'number' ? feature.value.toLocaleString('en-IN') : feature.value}
              </p>
            </div>
          ))}
        </div>

        {features.length > maxFeatures && (
          <div className="text-sm text-neutral-500 pt-4 border-t border-neutral-200">
            +{features.length - maxFeatures} more factors analyzed
          </div>
        )}
      </div>
    </Card>
  )
}

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
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
              <XAxis type="number" domain={[-1, 1]} hide />
              <YAxis dataKey="name" type="category" width={110} tick={{ fill: '#64748B', fontSize: 12 }} axisLine={false} tickLine={false} />
              <Tooltip
                formatter={(value: number) => [value, 'Impact']}
                labelFormatter={(label) => `Feature: ${label}`}
                contentStyle={{ borderRadius: '8px', border: '1px solid #E2E8F0', boxShadow: 'none' }}
              />
              <Bar dataKey="displayImpact" radius={[0, 4, 4, 0]}>
                {topFeatures.map((feature: FeatureContribution) => (
                  <Cell
                    key={feature.name}
                    fill={feature.impact >= 0 ? '#B0F0DA' : '#FF6B6B'}
                    stroke="#000000"
                    strokeWidth={2}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          {topFeatures.map((feature) => (
            <div key={feature.name} className={`rounded border-2 border-[#000000] p-3 shadow-[2px_2px_0px_#000000] ${feature.impact >= 0 ? 'bg-[#B0F0DA]' : 'bg-[#FF6B6B]'} text-[#000000]`}>
              <div className="flex items-center justify-between">
                <p className="font-black uppercase text-sm truncate mr-2">{feature.name.replace(/_/g, ' ')}</p>
                <span className="font-bold bg-white px-2 py-0.5 rounded border-2 border-[#000000] whitespace-nowrap">
                  {feature.impact >= 0 ? '+' : ''}
                  {feature.impact.toFixed(2)}
                </span>
              </div>
              <p className="mt-2 text-xs font-bold opacity-80">
                VALUE: {typeof feature.value === 'number' ? feature.value.toLocaleString('en-IN') : feature.value}
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

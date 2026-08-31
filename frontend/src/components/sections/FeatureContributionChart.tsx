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

/** SHAP values here are log-odds contributions and are therefore unbounded — a
 *  fixed [-1, 1] domain silently clips the largest attributions, which are
 *  exactly the ones that matter. Derive a symmetric, padded domain from the data
 *  instead, with a small floor so an all-tiny-impact chart is not blown up. */
const MIN_AXIS_EXTENT = 0.5
const AXIS_PADDING = 1.15

const symmetricDomain = (values: number[]): [number, number] => {
  const maxAbs = values.reduce(
    (acc, value) => (Number.isFinite(value) ? Math.max(acc, Math.abs(value)) : acc),
    0,
  )
  const extent = Math.max(MIN_AXIS_EXTENT, maxAbs * AXIS_PADDING)
  const rounded = Number(extent.toFixed(2))
  return [-rounded, rounded]
}

export const FeatureContributionChart: React.FC<FeatureContributionChartProps> = ({
  features,
  maxFeatures = 5,
  source,
}) => {
  const topFeatures = [...features]
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
    .slice(0, maxFeatures)
    .map((feature) => ({
      ...feature,
      displayImpact: Number(feature.impact.toFixed(2)),
    }))

  const domain = symmetricDomain(topFeatures.map((feature) => feature.displayImpact))
  const resolvedSource = source ?? topFeatures.find((feature) => feature.source)?.source
  const isHeuristic = resolvedSource === 'heuristic'

  return (
    <Card title="Feature Importance Analysis" className="mt-6">
      <div className="space-y-6">
        {isHeuristic && (
          <div className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-800">
            <span className="font-medium">Not SHAP.</span> SHAP attributions were unavailable for
            this application, so these factors come from a hand-written rule fallback.
          </div>
        )}
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={topFeatures} layout="vertical" margin={{ left: 12, right: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis type="number" domain={domain} allowDataOverflow={false} />
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

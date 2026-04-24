import React from 'react'
import { ArrowLeft, MapPinned } from 'lucide-react'
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import { useNavigate } from 'react-router-dom'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Button, Card } from '@/components/common'

const densityData = [
  { region: 'North', applications: 420, fill: '#2563eb', x: 152, y: 82 },
  { region: 'West', applications: 360, fill: '#0891b2', x: 96, y: 190 },
  { region: 'Central', applications: 280, fill: '#16a34a', x: 170, y: 205 },
  { region: 'East', applications: 310, fill: '#f59e0b', x: 245, y: 178 },
  { region: 'South', applications: 500, fill: '#dc2626', x: 170, y: 318 },
]

const ruralUrbanSplit = [
  { name: 'Urban', value: 54, fill: '#2563eb' },
  { name: 'Semi-Urban', value: 28, fill: '#16a34a' },
  { name: 'Rural', value: 18, fill: '#f59e0b' },
]

export const GeoAnalytics: React.FC = () => {
  const navigate = useNavigate()
  const maxApplications = Math.max(...densityData.map((item) => item.applications))

  return (
    <DashboardLayout title="Geo Analytics" role="organization">
      <section className="mb-8 rounded-[32px] border border-[#d6e7e4] bg-white p-8 shadow-sm">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.28em] text-neutral-500">Geographic Intelligence</p>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight text-neutral-900">India application density</h2>
            <p className="mt-3 max-w-2xl text-neutral-600">
              Regional density and rural-urban portfolio mix for organization-side planning.
            </p>
          </div>
          <Button variant="secondary" leftIcon={<ArrowLeft className="h-4 w-4" />} onClick={() => navigate('/dashboard/org')}>
            Back to Dashboard
          </Button>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <Card title="Regional Density">
          <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
            <div className="relative mx-auto aspect-[0.72] w-full max-w-[340px]">
              <svg viewBox="0 0 340 460" className="h-full w-full drop-shadow-sm" role="img" aria-label="India map density visualization">
                <path
                  d="M155 28c22 6 34 23 46 40 14 20 33 23 55 29 21 5 29 19 22 38-5 14-15 25-16 40-2 27 32 41 29 68-2 22-25 30-34 48-9 17 1 39-13 54-12 13-34 8-49 20-18 14-14 48-38 56-23 8-38-20-42-40-5-24-5-49-20-69-15-21-44-28-53-52-8-22 5-44 8-67 3-24-14-44-13-68 1-20 18-35 37-42 19-8 36-9 48-27 10-15 13-34 37-28z"
                  fill="#f8fafc"
                  stroke="#94a3b8"
                  strokeWidth="3"
                />
                {densityData.map((item) => {
                  const radius = 18 + (item.applications / maxApplications) * 26
                  return (
                    <g key={item.region}>
                      <circle cx={item.x} cy={item.y} r={radius} fill={item.fill} opacity="0.18" />
                      <circle cx={item.x} cy={item.y} r={radius * 0.48} fill={item.fill} opacity="0.78" />
                      <text x={item.x} y={item.y + 4} textAnchor="middle" className="fill-white text-[14px] font-bold">
                        {item.applications}
                      </text>
                    </g>
                  )
                })}
              </svg>
            </div>

            <div className="space-y-3">
              {densityData.map((item) => (
                <div key={item.region} className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="h-3 w-3 rounded-full" style={{ backgroundColor: item.fill }} />
                      <p className="font-semibold text-neutral-900">{item.region}</p>
                    </div>
                    <p className="text-lg font-semibold text-neutral-900">{item.applications}</p>
                  </div>
                  <div className="mt-3 h-2 rounded-full bg-white">
                    <div
                      className="h-2 rounded-full"
                      style={{ width: `${(item.applications / maxApplications) * 100}%`, backgroundColor: item.fill }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>

        <div className="space-y-6">
          <Card title="Rural vs Urban Split">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart margin={{ top: 12, right: 12, bottom: 12, left: 12 }}>
                  <Pie data={ruralUrbanSplit} dataKey="value" nameKey="name" outerRadius={105} label>
                    {ruralUrbanSplit.map((entry) => (
                      <Cell key={entry.name} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${value}%`} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card>
            <div className="flex items-start gap-4">
              <div className="rounded-2xl bg-primary-50 p-3 text-primary-600">
                <MapPinned className="h-6 w-6" />
              </div>
              <div>
                <p className="text-lg font-semibold text-neutral-900">Density Insight</p>
                <p className="mt-2 text-sm leading-6 text-neutral-600">
                  South and North regions currently show the highest application density, while rural demand remains a smaller but visible share of the portfolio.
                </p>
              </div>
            </div>
          </Card>
        </div>
      </section>
    </DashboardLayout>
  )
}

import React, { useEffect, useMemo, useState } from 'react'
import { ArrowLeft, MapPinned } from 'lucide-react'
import type { Layer, LeafletMouseEvent, Path, PathOptions } from 'leaflet'
import { GeoJSON, MapContainer, TileLayer } from 'react-leaflet'
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'
import { useNavigate } from 'react-router-dom'
import type { Feature, FeatureCollection } from 'geojson'
import 'leaflet/dist/leaflet.css'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Button, Card } from '@/components/common'
import {
  getLocationMetrics,
  getRegionMetrics,
  type LocationMetricsResponse,
  type RegionMetricsResponse,
} from '@/services/applications'

const REGION_VISUALS: Record<string, { fill: string; x: number; y: number }> = {
  North: { fill: '#2563eb', x: 152, y: 82 },
  West: { fill: '#0891b2', x: 96, y: 190 },
  Central: { fill: '#16a34a', x: 170, y: 205 },
  East: { fill: '#f59e0b', x: 245, y: 178 },
  South: { fill: '#dc2626', x: 170, y: 318 },
  Unknown: { fill: '#64748b', x: 278, y: 392 },
}

const STATE_TO_REGION: Record<string, string> = {
  'Andaman and Nicobar': 'South',
  'Andhra Pradesh': 'South',
  ArunachalPradesh: 'East',
  Assam: 'East',
  Bihar: 'East',
  Chandigarh: 'North',
  Chhattisgarh: 'Central',
  'Dadra and Nagar Haveli': 'West',
  DamanandDiu: 'West',
  Delhi: 'North',
  Goa: 'West',
  Gujarat: 'West',
  Haryana: 'North',
  HimachalPradesh: 'North',
  JammuandKashmir: 'North',
  Jharkhand: 'East',
  Karnataka: 'South',
  Kerala: 'South',
  Lakshadweep: 'South',
  'Madhya Pradesh': 'Central',
  Maharashtra: 'West',
  Manipur: 'East',
  Meghalaya: 'East',
  Mizoram: 'East',
  Nagaland: 'East',
  Odisha: 'East',
  Orissa: 'East',
  Puducherry: 'South',
  Punjab: 'North',
  Rajasthan: 'North',
  Sikkim: 'East',
  'Tamil Nadu': 'South',
  Telangana: 'South',
  Tripura: 'East',
  UttarPradesh: 'North',
  Uttarakhand: 'North',
  'West Bengal': 'East',
}

type StateFeatureProperties = {
  NAME_1?: string
} & Record<string, unknown>

type IndiaGeoJson = FeatureCollection

function normalizeStateName(name: string): string {
  return name.replace(/\s+/g, ' ').replace(/&/g, 'and').trim()
}

function getChoroplethColor(approvalRate: number, applications: number): string {
  if (applications === 0) return '#e5e7eb'
  if (approvalRate >= 75) return '#14532d'
  if (approvalRate >= 60) return '#16a34a'
  if (approvalRate >= 45) return '#65a30d'
  if (approvalRate >= 30) return '#ea580c'
  return '#b91c1c'
}

export const GeoAnalytics: React.FC = () => {
  const navigate = useNavigate()
  const [metrics, setMetrics] = useState<RegionMetricsResponse | null>(null)
  const [locationMetrics, setLocationMetrics] = useState<LocationMetricsResponse | null>(null)
  const [indiaGeoJson, setIndiaGeoJson] = useState<IndiaGeoJson | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let mounted = true

    const loadMetrics = async () => {
      try {
        const [regionResponse, locationResponse] = await Promise.all([
          getRegionMetrics(),
          getLocationMetrics(),
        ])
        if (!mounted) return
        setMetrics(regionResponse)
        setLocationMetrics(locationResponse)
        setError(null)
      } catch (fetchError) {
        if (!mounted) return
        setError(fetchError instanceof Error ? fetchError.message : 'Failed to load geo metrics')
      } finally {
        if (mounted) {
          setLoading(false)
        }
      }
    }

    void loadMetrics()
    const intervalId = window.setInterval(() => {
      void loadMetrics()
    }, 5000)

    return () => {
      mounted = false
      window.clearInterval(intervalId)
    }
  }, [])

  useEffect(() => {
    let mounted = true

    const loadGeoJson = async () => {
      try {
        const response = await fetch('/data/india_state.geojson')
        if (!response.ok) {
          throw new Error('Failed to load India GeoJSON source')
        }
        const data = (await response.json()) as IndiaGeoJson
        if (mounted) {
          setIndiaGeoJson(data)
        }
      } catch (geoError) {
        if (mounted) {
          setError(geoError instanceof Error ? geoError.message : 'Failed to load map source')
        }
      }
    }

    void loadGeoJson()

    return () => {
      mounted = false
    }
  }, [])

  const densityData = useMemo(() => {
    const regionMap = metrics?.regions ?? {}
    const names = new Set([...Object.keys(REGION_VISUALS), ...Object.keys(regionMap)])

    return Array.from(names)
      .map((region) => {
        const visual = REGION_VISUALS[region] ?? REGION_VISUALS.Unknown
        const values = regionMap[region]
        return {
          region,
          applications: values?.applications ?? 0,
          approved: values?.approved ?? 0,
          rejected: values?.rejected ?? 0,
          approvalRate: values?.approvalRate ?? 0,
          fill: visual.fill,
          x: visual.x,
          y: visual.y,
        }
      })
      .filter((item) => item.applications > 0 || item.region in REGION_VISUALS)
      .sort((a, b) => b.applications - a.applications)
  }, [metrics])

  const decisionMix = useMemo(() => {
    const totals = densityData.reduce(
      (acc, item) => {
        acc.approved += item.approved
        acc.rejected += item.rejected
        acc.applications += item.applications
        return acc
      },
      { approved: 0, rejected: 0, applications: 0 },
    )

    const deferred = Math.max(totals.applications - totals.approved - totals.rejected, 0)
    return [
      { name: 'Approved', value: totals.approved, fill: '#16a34a' },
      { name: 'Rejected', value: totals.rejected, fill: '#dc2626' },
      { name: 'Deferred', value: deferred, fill: '#f59e0b' },
    ]
  }, [densityData])

  const maxApplications = Math.max(...densityData.map((item) => item.applications), 1)
  const topRegion = densityData[0]
  const regionMap = metrics?.regions ?? {}
  const topAreas = Object.entries(locationMetrics?.areas ?? {}).slice(0, 5)
  const topStates = Object.entries(locationMetrics?.states ?? {}).slice(0, 8)
  const topCities = Object.entries(locationMetrics?.cities ?? {}).slice(0, 8)

  const geoJsonStyle = (feature?: Feature): PathOptions => {
    const props = (feature?.properties ?? {}) as StateFeatureProperties
    const stateName = normalizeStateName(props.NAME_1 ?? 'Unknown')
    const region = STATE_TO_REGION[stateName] ?? 'Unknown'
    const stateMetrics = regionMap[region]

    return {
      fillColor: getChoroplethColor(stateMetrics?.approvalRate ?? 0, stateMetrics?.applications ?? 0),
      weight: 1,
      opacity: 1,
      color: '#334155',
      fillOpacity: 0.85,
    }
  }

  const onEachFeature = (feature: Feature, layer: Layer) => {
    const props = (feature.properties ?? {}) as StateFeatureProperties
    const stateName = normalizeStateName(props.NAME_1 ?? 'Unknown')
    const region = STATE_TO_REGION[stateName] ?? 'Unknown'
    const stateMetrics = regionMap[region]
    const applications = stateMetrics?.applications ?? 0
    const approvalRate = stateMetrics?.approvalRate ?? 0

    const tooltipContent = [
      `<strong>${stateName}</strong>`,
      `Region: ${region}`,
      `Applications: ${applications}`,
      `Approval Rate: ${approvalRate.toFixed(1)}%`,
    ].join('<br />')

    layer.bindTooltip(tooltipContent, { sticky: true })

    layer.on({
      mouseover: (event: LeafletMouseEvent) => {
        const target = event.target as Path
        target.setStyle({ weight: 2.5, color: '#0f172a' })
      },
      mouseout: (event: LeafletMouseEvent) => {
        const target = event.target as Path
        target.setStyle({ weight: 1, color: '#334155' })
      },
    })
  }

  return (
    <DashboardLayout title="Geo Analytics" role="organization">
      <section className="mb-8 rounded-[32px] border border-[#d6e7e4] bg-white p-8 shadow-sm">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.28em] text-neutral-500">Geographic Intelligence</p>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight text-neutral-900">India application density</h2>
            <p className="mt-3 max-w-2xl text-neutral-600">
              Live regional application density and outcomes from submitted applications.
            </p>
          </div>
          <Button variant="secondary" leftIcon={<ArrowLeft className="h-4 w-4" />} onClick={() => navigate('/dashboard/org')}>
            Back to Dashboard
          </Button>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <Card title="Regional Density">
          {loading && <p className="mb-4 text-sm text-neutral-500">Loading map metrics...</p>}
          {error && <p className="mb-4 text-sm text-red-600">{error}</p>}

          <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
            <div className="mx-auto w-full max-w-[560px] overflow-hidden rounded-2xl border border-neutral-200">
              {indiaGeoJson ? (
                <MapContainer
                  center={[22.9, 79.0]}
                  zoom={4.5}
                  minZoom={4}
                  maxZoom={7}
                  scrollWheelZoom={false}
                  className="h-[460px] w-full"
                >
                  <TileLayer
                    attribution='&copy; OpenStreetMap contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  <GeoJSON
                    key={metrics?.updatedAt ?? 'region-metrics'}
                    data={indiaGeoJson}
                    style={geoJsonStyle}
                    onEachFeature={onEachFeature}
                  />
                </MapContainer>
              ) : (
                <div className="flex h-[460px] items-center justify-center bg-neutral-50 text-sm text-neutral-500">
                  Loading map boundary...
                </div>
              )}
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
                  <p className="mt-2 text-sm text-neutral-600">Approval rate: {item.approvalRate.toFixed(1)}%</p>
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
          <Card title="Decision Mix">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart margin={{ top: 12, right: 12, bottom: 12, left: 12 }}>
                  <Pie data={decisionMix} dataKey="value" nameKey="name" outerRadius={105} label>
                    {decisionMix.map((entry) => (
                      <Cell key={entry.name} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => String(value)} />
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
                  {topRegion
                    ? `${topRegion.region} currently has the highest application volume with ${topRegion.applications} submissions and ${topRegion.approvalRate.toFixed(1)}% approval rate.`
                    : 'Region-level insights will appear as soon as applications are available.'}
                </p>
              </div>
            </div>
          </Card>
        </div>
      </section>

      <section className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card title="Top Areas (Region)">
          <div className="space-y-3">
            {topAreas.map(([name, values]) => (
              <div key={name} className="rounded-xl border border-neutral-200 bg-neutral-50 px-3 py-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium text-neutral-900">{name}</span>
                  <span className="text-neutral-700">{values.applications}</span>
                </div>
                <p className="mt-1 text-xs text-neutral-600">Approve {values.approvalRate.toFixed(1)}% | Reject {values.rejectionRate.toFixed(1)}%</p>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Top States">
          <div className="space-y-3">
            {topStates.map(([name, values]) => (
              <div key={name} className="rounded-xl border border-neutral-200 bg-neutral-50 px-3 py-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium text-neutral-900">{name}</span>
                  <span className="text-neutral-700">{values.applications}</span>
                </div>
                <p className="mt-1 text-xs text-neutral-600">Approve {values.approvalRate.toFixed(1)}% | Deferred {values.deferralRate.toFixed(1)}%</p>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Top Cities">
          <div className="space-y-3">
            {topCities.map(([name, values]) => (
              <div key={name} className="rounded-xl border border-neutral-200 bg-neutral-50 px-3 py-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium text-neutral-900">{name}</span>
                  <span className="text-neutral-700">{values.applications}</span>
                </div>
                <p className="mt-1 text-xs text-neutral-600">Approve {values.approvalRate.toFixed(1)}% | Reject {values.rejectionRate.toFixed(1)}%</p>
              </div>
            ))}
          </div>
        </Card>
      </section>
    </DashboardLayout>
  )
}

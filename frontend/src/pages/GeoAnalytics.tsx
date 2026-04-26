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
  North: { fill: '#6E61FF', x: 152, y: 82 },
  West: { fill: '#B0F0DA', x: 96, y: 190 },
  Central: { fill: '#FDE047', x: 170, y: 205 },
  East: { fill: '#FD9745', x: 245, y: 178 },
  South: { fill: '#FF6B6B', x: 170, y: 318 },
}

const STATE_TO_REGION: Record<string, string> = {
  'Andaman and Nicobar': 'South',
  'Andaman and Nicobar Islands': 'South',
  'Andhra Pradesh': 'South',
  ArunachalPradesh: 'East',
  'Arunachal Pradesh': 'East',
  Assam: 'East',
  Bihar: 'East',
  Chandigarh: 'North',
  Chhattisgarh: 'Central',
  'Dadra and Nagar Haveli': 'West',
  'Dadra and Nagar Haveli and Daman and Diu': 'West',
  DamanandDiu: 'West',
  'Daman and Diu': 'West',
  Delhi: 'North',
  Goa: 'West',
  Gujarat: 'West',
  Haryana: 'North',
  HimachalPradesh: 'North',
  'Himachal Pradesh': 'North',
  JammuandKashmir: 'North',
  'Jammu and Kashmir': 'North',
  Jharkhand: 'East',
  Karnataka: 'South',
  Kerala: 'South',
  Lakshadweep: 'South',
  'Madhya Pradesh': 'Central',
  Ladakh: 'North',
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
  'Uttar Pradesh': 'North',
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
  if (applications === 0) return '#ffffff'
  if (approvalRate >= 75) return '#B0F0DA'
  if (approvalRate >= 60) return '#6E61FF'
  if (approvalRate >= 45) return '#FDE047'
  if (approvalRate >= 30) return '#FD9745'
  return '#FF6B6B'
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
      .filter((region) => region !== 'Unknown')
      .map((region) => {
        const visual = REGION_VISUALS[region] || { fill: '#000', x: 0, y: 0 }
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
      .filter((item) => item.applications > 0 && item.region in REGION_VISUALS)
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
      { name: 'Approved', value: totals.approved, fill: '#B0F0DA' },
      { name: 'Rejected', value: totals.rejected, fill: '#FF6B6B' },
      { name: 'Deferred', value: deferred, fill: '#FD9745' },
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
      weight: 2,
      opacity: 1,
      color: '#000000',
      fillOpacity: 1,
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
        target.setStyle({ weight: 4, color: '#000000' })
      },
      mouseout: (event: LeafletMouseEvent) => {
        const target = event.target as Path
        target.setStyle({ weight: 2, color: '#000000' })
      },
    })
  }

  return (
    <DashboardLayout title="Geo Analytics" role="organization">
      <section className="mb-8 rounded border-4 border-black bg-white p-8 shadow-[8px_8px_0px_#000000]">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs font-black uppercase tracking-wider text-black opacity-60">Geographic Intelligence</p>
            <h2 className="mt-3 text-4xl font-black tracking-tight text-black">India application density</h2>
            <p className="mt-3 max-w-2xl font-bold text-black opacity-80">
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
            <div className="mx-auto w-full overflow-hidden rounded border-2 border-black shadow-[4px_4px_0px_#000000] z-10 flex flex-col">
              {indiaGeoJson ? (
                <MapContainer
                  center={[22.5, 82.0]}
                  zoom={4.2}
                  minZoom={4}
                  maxZoom={7}
                  scrollWheelZoom={false}
                  className="w-full"
                  style={{ height: '550px' }}
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
                <div className="flex h-[550px] items-center justify-center bg-neutral-50 text-sm font-bold text-black">
                  Loading map boundary...
                </div>
              )}
            </div>

            <div className="space-y-3">
              {densityData.map((item) => (
                <div key={item.region} className="rounded border-2 border-black bg-white p-4 shadow-[4px_4px_0px_#000000]">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="h-4 w-4 rounded border-2 border-black" style={{ backgroundColor: item.fill }} />
                      <p className="font-black text-black">{item.region}</p>
                    </div>
                    <p className="text-lg font-black text-black">{item.applications}</p>
                  </div>
                  <p className="mt-2 text-sm font-bold text-black opacity-80">Approval rate: {item.approvalRate.toFixed(1)}%</p>
                  <div className="mt-3 h-3 border-2 border-black bg-white w-full overflow-hidden">
                    <div
                      className="h-full border-r-2 border-black"
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

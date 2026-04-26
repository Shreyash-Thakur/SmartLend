import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity, BriefcaseBusiness, CheckCircle2, Clock3, MapPinned,
  Users, XCircle, AlertTriangle,
} from 'lucide-react'
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart,
  Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Button, Card, KPICard } from '@/components/common'
import { ApplicationTable } from '@/components/sections'
import { useApplicationData } from '@/hooks/useApplicationData'
import { getStats } from '@/services/applications'
import type { StatsResponse } from '@/types/api'
import { formatCurrency } from '@/lib/utils'
import type { LoanApplication } from '@/types/application'

type OrgTab = 'all' | 'deferred' | 'approved' | 'rejected' | 'confirmed'

const TAB_CONFIG: Array<{ id: OrgTab; label: string; color: string; activeColor: string }> = [
  { id: 'all', label: 'All Applications', color: 'text-neutral-600', activeColor: 'border-primary-600 text-primary-700 bg-primary-50' },
  { id: 'deferred', label: 'Needs Review', color: 'text-amber-600', activeColor: 'border-amber-500 text-amber-700 bg-amber-50' },
  { id: 'approved', label: 'Auto-Approved', color: 'text-green-600', activeColor: 'border-green-500 text-green-700 bg-green-50' },
  { id: 'rejected', label: 'Auto-Rejected', color: 'text-red-600', activeColor: 'border-red-500 text-red-700 bg-red-50' },
  { id: 'confirmed', label: 'Org-Confirmed', color: 'text-violet-600', activeColor: 'border-violet-500 text-violet-700 bg-violet-50' },
]

export const OrganizationDashboard: React.FC = () => {
  const navigate = useNavigate()
  const { applications, isLoading, error, bulkOverrideDecision } = useApplicationData({ scope: 'org' })
  const [activeTab, setActiveTab] = useState<OrgTab>('all')
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [dashboardError, setDashboardError] = useState<string | null>(null)
  const [statsLoading, setStatsLoading] = useState(true)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [bulkNotes, setBulkNotes] = useState('')
  const [bulkLoading, setBulkLoading] = useState(false)
  const [bulkError, setBulkError] = useState<string | null>(null)
  const [showBulkPanel, setShowBulkPanel] = useState(false)

  useEffect(() => {
    const load = async () => {
      setStatsLoading(true)
      try {
        const s = await getStats()
        setStats(s)
      } catch (e) {
        setDashboardError(e instanceof Error ? e.message : 'Failed to load stats')
      } finally {
        setStatsLoading(false)
      }
    }
    void load()
  }, [])

  // Filter logic for tabs
  const tabFiltered = useMemo<LoanApplication[]>(() => {
    if (activeTab === 'all') return applications
    if (activeTab === 'deferred') return applications.filter((a) => a.modelRecommendation === 'deferred' && !a.manualDecisionApplied)
    if (activeTab === 'approved') return applications.filter((a) => a.modelRecommendation === 'approved' && !a.manualDecisionApplied)
    if (activeTab === 'rejected') return applications.filter((a) => a.modelRecommendation === 'rejected' && !a.manualDecisionApplied)
    if (activeTab === 'confirmed') return applications.filter((a) => a.manualDecisionApplied)
    return applications
  }, [applications, activeTab])

  const tabCounts = useMemo(() => ({
    all: applications.length,
    deferred: applications.filter((a) => a.modelRecommendation === 'deferred' && !a.manualDecisionApplied).length,
    approved: applications.filter((a) => a.modelRecommendation === 'approved' && !a.manualDecisionApplied).length,
    rejected: applications.filter((a) => a.modelRecommendation === 'rejected' && !a.manualDecisionApplied).length,
    confirmed: applications.filter((a) => a.manualDecisionApplied).length,
  }), [applications])

  const uploadedCount = applications.filter((a) => a.source === 'seed').length
  const submittedCount = applications.filter((a) => a.source === 'customer').length

  const handleRowClick = (app: { id: string }) => navigate(`/review/${app.id}`)

  const toggleSelect = useCallback((id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const toggleSelectAll = () => {
    if (selected.size === tabFiltered.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(tabFiltered.map((a) => a.id)))
    }
  }

  const handleBulkAction = async (status: 'approved' | 'rejected') => {
    if (!bulkNotes.trim()) {
      setBulkError('Please provide notes for the bulk decision.')
      return
    }
    if (selected.size === 0) {
      setBulkError('No applications selected.')
      return
    }
    setBulkError(null)
    setBulkLoading(true)
    try {
      await bulkOverrideDecision(Array.from(selected), status, bulkNotes)
      setSelected(new Set())
      setBulkNotes('')
      setShowBulkPanel(false)
    } catch (e) {
      setBulkError(e instanceof Error ? e.message : 'Bulk action failed')
    } finally {
      setBulkLoading(false)
    }
  }

  const trends = useMemo(() => {
    const now = new Date()
    const weeks = [
      { label: 'Week 1', start: new Date(now.getTime() - 28 * 86400000), end: new Date(now.getTime() - 21 * 86400000) },
      { label: 'Week 2', start: new Date(now.getTime() - 21 * 86400000), end: new Date(now.getTime() - 14 * 86400000) },
      { label: 'Week 3', start: new Date(now.getTime() - 14 * 86400000), end: new Date(now.getTime() - 7 * 86400000) },
      { label: 'Week 4', start: new Date(now.getTime() - 7 * 86400000), end: new Date(now.getTime() + 86400000) },
    ]
    return weeks.map((w) => {
      const bucket = applications.filter((a) => {
        const d = new Date(a.createdAt)
        return d >= w.start && d < w.end
      })
      return {
        date: w.label,
        count: bucket.length,
        approved: bucket.filter((a) => a.finalDecision === 'APPROVE').length,
        rejected: bucket.filter((a) => a.finalDecision === 'REJECT').length,
        deferred: bucket.filter((a) => a.finalDecision === 'DEFER').length,
      }
    })
  }, [applications])

  const approvalDistribution = useMemo(() => [
    { label: 'Approved', value: stats?.approved ?? 0, fill: '#10b981' },
    { label: 'Rejected', value: stats?.rejected ?? 0, fill: '#ef4444' },
    { label: 'Deferred', value: stats?.deferred ?? 0, fill: '#f59e0b' },
  ], [stats])

  const categoryAnalysis = useMemo(() => {
    const counts = new Map<string, number>()
    for (const a of applications) counts.set(a.loanPurpose, (counts.get(a.loanPurpose) ?? 0) + 1)
    return Array.from(counts.entries()).map(([label, value]) => ({
      label: label.charAt(0).toUpperCase() + label.slice(1),
      value,
    }))
  }, [applications])

  return (
    <DashboardLayout title="Organization Dashboard" role="organization">
      {/* Hero */}
      <section className="mb-8 grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
        <div className="rounded border-4 border-black bg-white p-8 shadow-[8px_8px_0px_#000000]">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="max-w-2xl">
              <p className="text-xs font-black uppercase tracking-wider text-black opacity-60">Operations Dashboard</p>
              <h2 className="mt-3 text-4xl font-black tracking-tight text-black">Unified application pipeline</h2>
              <p className="mt-4 text-base font-bold leading-7 text-black opacity-80">
                All records in one queue. Review, override, and confirm ML decisions. Deferred cases require human action before the customer is notified.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <OrgChip icon={<Users className="h-4 w-4" />} label={`${uploadedCount} training records`} />
              <OrgChip icon={<BriefcaseBusiness className="h-4 w-4" />} label={`${submittedCount} live applications`} />
              <OrgChip icon={<Activity className="h-4 w-4" />} label="Analyst workflow active" />
              <OrgChip icon={<Clock3 className="h-4 w-4" />} label="Real-time backend sync" />
            </div>
          </div>
          <div className="mt-6 flex flex-wrap gap-3">
            <Button variant="secondary" onClick={() => navigate('/dashboard/models')}>Model Analysis Dashboard</Button>
            <Button variant="primary" leftIcon={<MapPinned className="h-4 w-4" />} onClick={() => navigate('/analytics/geo')}>Geo Analytics</Button>
          </div>
        </div>

        <div className="grid gap-4">
          <div className="rounded border-4 border-black bg-[#6E61FF] text-white p-6 shadow-[8px_8px_0px_#000000] flex flex-col justify-between">
            <div className="space-y-2">
              <p className="text-xs font-black uppercase tracking-wider text-white opacity-80">Application Pipeline</p>
              <p className="text-6xl font-black">{applications.length}</p>
            </div>
            <div className="grid gap-3 text-sm font-bold mt-6">
              <div className="flex items-center justify-between bg-white/10 p-2 rounded border border-black shadow-[2px_2px_0px_#000000]">
                <span className="flex items-center gap-2"><CheckCircle2 className="h-5 w-5 text-[#B0F0DA]" />Auto-Approved</span>
                <span className="bg-white text-black px-2 py-0.5 rounded border-2 border-black">{tabCounts.approved}</span>
              </div>
              <div className="flex items-center justify-between bg-white/10 p-2 rounded border border-black shadow-[2px_2px_0px_#000000]">
                <span className="flex items-center gap-2"><XCircle className="h-5 w-5 text-[#FF6B6B]" />Auto-Rejected</span>
                <span className="bg-white text-black px-2 py-0.5 rounded border-2 border-black">{tabCounts.rejected}</span>
              </div>
              <div className="flex items-center justify-between bg-white/10 p-2 rounded border border-black shadow-[2px_2px_0px_#000000]">
                <span className="flex items-center gap-2"><AlertTriangle className="h-5 w-5 text-[#FD9745]" />Needs Review</span>
                <span className="bg-white text-black px-2 py-0.5 rounded border-2 border-black">{tabCounts.deferred}</span>
              </div>
              <div className="flex items-center justify-between bg-white/10 p-2 rounded border border-black shadow-[2px_2px_0px_#000000] mt-2">
                <span className="flex items-center gap-2">✓ Org Confirmed</span>
                <span className="bg-[#B0F0DA] text-black px-2 py-0.5 rounded border-2 border-black">{tabCounts.confirmed}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {(error || dashboardError) && (
        <section className="mb-8">
          <Card className="border-red-200 bg-red-50">
            <p className="text-red-700">Connection issue: {error ?? dashboardError}</p>
          </Card>
        </section>
      )}

      {/* Live Stats */}
      <section className="mb-8">
        <Card title="Live Stats">
          {statsLoading ? (
            <p className="text-neutral-600">Loading stats…</p>
          ) : stats ? (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
              <KPICard label="Total Applications" value={stats.totalApplications} format="number" />
              <KPICard label="Approval Rate" value={stats.approvalRate} format="percentage" />
              <KPICard label="Rejection Rate" value={stats.rejectionRate} format="percentage" />
              <KPICard label="Deferral Rate" value={stats.deferralRate} format="percentage" />
              <KPICard label="Average CBES" value={stats.averageCBES} format="number" />
              <KPICard label="Average ML Score" value={stats.averageMLProbability} format="number" />
            </div>
          ) : <p className="text-neutral-600">Stats unavailable</p>}
        </Card>
      </section>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <Card title="Approval Distribution">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={approvalDistribution} dataKey="value" nameKey="label" outerRadius={90}>
                  {approvalDistribution.map((e) => <Cell key={e.label} fill={e.fill} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Applications Over Time">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trends} margin={{ top: 12, right: 24, bottom: 8, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="date" />
                <YAxis allowDecimals={false} width={36} />
                <Tooltip />
                <Line type="monotone" dataKey="approved" stroke="#10b981" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="rejected" stroke="#ef4444" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="deferred" stroke="#f59e0b" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Loan Purpose Mix">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryAnalysis} margin={{ top: 12, right: 24, bottom: 36, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="label" interval={0} angle={-18} textAnchor="end" height={60} />
                <YAxis allowDecimals={false} width={36} />
                <Tooltip />
                <Bar dataKey="value" fill="#0ea5e9" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Operations Snapshot">
          <div className="grid gap-4 mt-2">
            <Snapshot
              label="Avg Loan Amount"
              value={stats ? formatCurrency(Math.round(applications.reduce((s, a) => s + a.loanAmount, 0) / Math.max(applications.length, 1))) : '--'}
            />
            <Snapshot label="Deferred (needs human)" value={tabCounts.deferred.toString()} />
            <Snapshot label="Org-confirmed decisions" value={tabCounts.confirmed.toString()} />
          </div>
        </Card>
      </div>

      {/* Application Table with Tabs + Bulk Actions */}
      <div className="bg-white rounded-2xl border border-neutral-200 shadow-md overflow-hidden">
        {/* Tab Bar */}
        <div className="border-b border-neutral-200 flex overflow-x-auto">
          {TAB_CONFIG.map((tab) => (
            <button
              key={tab.id}
              onClick={() => { setActiveTab(tab.id); setSelected(new Set()) }}
              className={`flex-shrink-0 px-5 py-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? `border-b-2 ${tab.activeColor}`
                  : `border-transparent ${tab.color} hover:bg-neutral-50`
              }`}
            >
              {tab.label} ({tabCounts[tab.id]})
            </button>
          ))}
        </div>

        {/* Bulk Action Toolbar */}
        {(activeTab === 'approved' || activeTab === 'rejected' || activeTab === 'deferred') && (
          <div className="border-b border-neutral-100 bg-neutral-50 px-6 py-3 flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-neutral-600 cursor-pointer">
              <input
                type="checkbox"
                className="w-4 h-4 rounded accent-primary-600"
                checked={selected.size > 0 && selected.size === tabFiltered.length}
                onChange={toggleSelectAll}
              />
              {selected.size > 0 ? `${selected.size} selected` : 'Select all'}
            </label>
            {selected.size > 0 && (
              <button
                onClick={() => setShowBulkPanel(!showBulkPanel)}
                className="text-sm font-medium text-primary-600 hover:underline"
              >
                {showBulkPanel ? 'Hide bulk panel' : 'Bulk action →'}
              </button>
            )}
          </div>
        )}

        {/* Bulk Panel */}
        {showBulkPanel && selected.size > 0 && (
          <div className="bg-primary-50 border-b border-primary-100 px-6 py-4 flex flex-wrap gap-4 items-start">
            <div className="flex-1 min-w-[240px]">
              <label className="block text-xs font-semibold text-primary-900 mb-1">Decision Notes (required)</label>
              <input
                type="text"
                value={bulkNotes}
                onChange={(e) => setBulkNotes(e.target.value)}
                placeholder="e.g. Batch confirmed after committee review"
                className="w-full rounded-lg border border-primary-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-400"
              />
              {bulkError && <p className="mt-1 text-xs text-red-600">{bulkError}</p>}
            </div>
            <div className="flex gap-2 pt-5">
              <button
                onClick={() => void handleBulkAction('approved')}
                disabled={bulkLoading}
                className="flex items-center gap-1.5 rounded-lg bg-green-600 px-4 py-2 text-sm font-semibold text-white hover:bg-green-700 disabled:opacity-50"
              >
                <CheckCircle2 className="h-4 w-4" />
                {bulkLoading ? 'Processing…' : `Approve ${selected.size}`}
              </button>
              <button
                onClick={() => void handleBulkAction('rejected')}
                disabled={bulkLoading}
                className="flex items-center gap-1.5 rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700 disabled:opacity-50"
              >
                <XCircle className="h-4 w-4" />
                {bulkLoading ? 'Processing…' : `Reject ${selected.size}`}
              </button>
            </div>
          </div>
        )}

        {/* Table */}
        <div className="p-6">
          {activeTab === 'deferred' && tabCounts.deferred > 0 && (
            <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              These {tabCounts.deferred} cases require an analyst decision. Customers are in a pending state until action is taken.
            </div>
          )}
          <ApplicationTable
            data={tabFiltered}
            onRowClick={handleRowClick}
            isLoading={isLoading}
            pageSize={25}
            showApplicant
            selectedIds={selected}
            onToggleSelect={
              (activeTab === 'approved' || activeTab === 'rejected' || activeTab === 'deferred')
                ? toggleSelect
                : undefined
            }
          />
        </div>
      </div>
    </DashboardLayout>
  )
}

function OrgChip({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full bg-white/80 px-4 py-2 text-sm font-medium text-neutral-700 shadow-sm">
      {icon}{label}
    </div>
  )
}

function Snapshot({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-neutral-100 last:border-0">
      <p className="text-sm text-neutral-500">{label}</p>
      <p className="font-semibold text-neutral-900">{value}</p>
    </div>
  )
}

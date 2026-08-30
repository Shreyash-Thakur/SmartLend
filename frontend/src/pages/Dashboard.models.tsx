import React, { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { DashboardLayout } from '@/components/layouts/DashboardLayout'
import { Card, KPICard } from '@/components/common'
import { getModelAnalysis, getStats, getActiveModel, setActiveModel } from '@/services/applications'
import type { ModelAnalysisResponse, ModelCaseItem, ModelMetricItem, StatsResponse } from '@/types/api'

const DECISION_COLORS: Record<string, string> = {
  APPROVE: '#10b981',
  REJECT: '#ef4444',
  DEFER: '#f59e0b',
}

/**
 * Real Home Credit evaluation artifacts hold tens of thousands of rows.
 * Aggregates (models / confusionByModel / probabilityBands / summary) are computed
 * server-side over the FULL dataset regardless of this limit, so we only pull a
 * bounded slice of case-level rows for the drill-down table and scatter plot.
 */
const CASE_LIMIT = 4000
/** Max SVG points we are willing to paint in the scatter chart. */
const SCATTER_MAX_POINTS = 1500
const CASE_PAGE_SIZE = 25

/** Coerce anything into a finite number, or null when the value is missing/NaN. */
function safeNum(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

/** 0..1 ratio -> 0..100 percentage number, or null if unusable. */
function asPct(value: unknown): number | null {
  const num = safeNum(value)
  if (num === null) return null
  return Number((num * 100).toFixed(2))
}

/** Percentage number (already 0..100) -> display string, dash when missing. */
function fmtPct(value: unknown, digits = 2): string {
  const num = safeNum(value)
  if (num === null) return '—'
  return `${num.toFixed(digits)}%`
}

/** 0..1 ratio -> display string, dash when missing. */
function fmtRatioPct(value: unknown, digits = 2): string {
  return fmtPct(asPct(value), digits)
}

function fmtInt(value: unknown): string {
  const num = safeNum(value)
  if (num === null) return '—'
  return Math.round(num).toLocaleString('en-IN')
}

/** Recharts silently drops NaN but chokes on undefined domains; null is safe. */
function chartValue(value: unknown): number | null {
  return asPct(value)
}

type MetricKey = 'accuracy' | 'precision' | 'recall' | 'f1' | 'auc'

const METRIC_COLUMNS: { key: MetricKey; label: string; hint: string }[] = [
  { key: 'auc', label: 'AUC', hint: 'Ranking quality across all thresholds' },
  { key: 'accuracy', label: 'Accuracy', hint: 'Overall correct decisions' },
  { key: 'precision', label: 'Precision', hint: 'TP / (TP + FP) — controls false approvals' },
  { key: 'recall', label: 'Recall', hint: 'TP / (TP + FN) — controls missed good applicants' },
  { key: 'f1', label: 'F1', hint: 'Harmonic mean of precision and recall' },
]

type SortKey = 'model' | MetricKey
type SortDir = 'asc' | 'desc'

function bestValueFor(models: ModelMetricItem[], key: MetricKey): number | null {
  let best: number | null = null
  for (const model of models) {
    const value = safeNum(model[key])
    if (value === null) continue
    if (best === null || value > best) best = value
  }
  return best
}

function buildInsightLines(models: ModelMetricItem[], bestModel: string) {
  if (!models.length) return []
  const lines: string[] = []

  const withAuc = models.filter((m) => safeNum(m.auc) !== null)
  if (withAuc.length) {
    const sortedByAuc = [...withAuc].sort((a, b) => (safeNum(b.auc) ?? 0) - (safeNum(a.auc) ?? 0))
    const first = sortedByAuc[0]
    const second = sortedByAuc[1]
    const gap = second ? (safeNum(first.auc) ?? 0) - (safeNum(second.auc) ?? 0) : null
    lines.push(
      `${bestModel || first.model} leads on AUC with ${fmtRatioPct(first.auc)}.` +
        (gap !== null ? ` Margin vs ${second.model}: ${(gap * 100).toFixed(2)}pp.` : ''),
    )
  }

  const withRecall = models.filter((m) => safeNum(m.recall) !== null)
  if (withRecall.length) {
    const top = [...withRecall].sort((a, b) => (safeNum(b.recall) ?? 0) - (safeNum(a.recall) ?? 0))[0]
    lines.push(
      `${top.model} gives highest recall at ${fmtRatioPct(top.recall)}, useful when minimizing missed good applicants.`,
    )
  }

  const withPrecision = models.filter((m) => safeNum(m.precision) !== null)
  if (withPrecision.length) {
    const top = [...withPrecision].sort((a, b) => (safeNum(b.precision) ?? 0) - (safeNum(a.precision) ?? 0))[0]
    lines.push(
      `${top.model} has strongest precision at ${fmtRatioPct(top.precision)}, useful when controlling false approvals.`,
    )
  }

  const missing = models.filter((m) =>
    METRIC_COLUMNS.some((column) => safeNum(m[column.key]) === null),
  )
  if (missing.length) {
    lines.push(
      `Incomplete metrics for: ${missing.map((m) => m.model).join(', ')}. Those cells show a dash instead of a value.`,
    )
  }

  return lines
}

export const ModelAnalysisDashboard: React.FC = () => {
  const [analysis, setAnalysis] = useState<ModelAnalysisResponse | null>(null)
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [activeModel, setActiveModelState] = useState<string>('LogisticRegression')
  const [isChangingModel, setIsChangingModel] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [sortKey, setSortKey] = useState<SortKey>('auc')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [confusionModel, setConfusionModel] = useState<string>('')
  const [casePage, setCasePage] = useState(0)

  useEffect(() => {
    const loadAnalysis = async () => {
      setIsLoading(true)
      setError(null)
      try {
        const [analysisResponse, statsResponse, currentActiveModel] = await Promise.all([
          getModelAnalysis(CASE_LIMIT),
          getStats(),
          getActiveModel(),
        ])
        setAnalysis(analysisResponse)
        setStats(statsResponse)
        setActiveModelState(currentActiveModel)
      } catch (fetchError) {
        setError(fetchError instanceof Error ? fetchError.message : 'Failed to load model analysis')
      } finally {
        setIsLoading(false)
      }
    }

    void loadAnalysis()
  }, [])

  const models = useMemo(() => analysis?.models ?? [], [analysis])
  const cases = useMemo(() => analysis?.cases ?? [], [analysis])
  const confusionRows = useMemo(() => analysis?.confusionByModel ?? [], [analysis])

  // Default the confusion-matrix selector to the best model once data lands.
  useEffect(() => {
    if (!confusionRows.length) return
    setConfusionModel((current) =>
      current && confusionRows.some((row) => row.model === current) ? current : confusionRows[0].model,
    )
  }, [confusionRows])

  const bestValues = useMemo(() => {
    const map = {} as Record<MetricKey, number | null>
    for (const column of METRIC_COLUMNS) {
      map[column.key] = bestValueFor(models, column.key)
    }
    return map
  }, [models])

  const sortedModels = useMemo(() => {
    const copy = [...models]
    copy.sort((a, b) => {
      if (sortKey === 'model') {
        return sortDir === 'asc' ? a.model.localeCompare(b.model) : b.model.localeCompare(a.model)
      }
      const av = safeNum(a[sortKey])
      const bv = safeNum(b[sortKey])
      // Missing metrics always sink to the bottom regardless of direction.
      if (av === null && bv === null) return 0
      if (av === null) return 1
      if (bv === null) return -1
      return sortDir === 'asc' ? av - bv : bv - av
    })
    return copy
  }, [models, sortKey, sortDir])

  const modelMetricsData = useMemo(
    () =>
      models.map((model) => ({
        model: model.model,
        accuracy: chartValue(model.accuracy),
        precision: chartValue(model.precision),
        recall: chartValue(model.recall),
        auc: chartValue(model.auc),
        f1: chartValue(model.f1),
      })),
    [models],
  )

  const aucChartData = useMemo(() => {
    const rows = models
      .map((model) => ({ model: model.model, auc: asPct(model.auc) }))
      .filter((row): row is { model: string; auc: number } => row.auc !== null)
    rows.sort((a, b) => b.auc - a.auc)
    return rows
  }, [models])

  const bestAuc = aucChartData.length ? aucChartData[0].auc : null

  // Down-sample: real artifacts carry tens of thousands of rows; painting them
  // all as SVG circles locks the browser. Deterministic stride keeps the shape.
  const scatterData = useMemo(() => {
    const stride = Math.max(1, Math.ceil(cases.length / SCATTER_MAX_POINTS))
    const sampled: ModelCaseItem[] = []
    for (let i = 0; i < cases.length; i += stride) sampled.push(cases[i])
    return sampled
      .map((item) => ({
        applicantId: item.applicantId,
        x: asPct(item.bestModelProb),
        y: asPct(item.cbesProb),
        confidence: asPct(item.hybridConfidence),
        decision: item.hybridDecision,
      }))
      .filter((point) => point.x !== null && point.y !== null)
  }, [cases])

  const precisionRecallData = useMemo(
    () =>
      models
        .map((model) => ({
          model: model.model,
          precision: asPct(model.precision),
          recall: asPct(model.recall),
          auc: asPct(model.auc),
        }))
        .filter(
          (point): point is { model: string; precision: number; recall: number; auc: number | null } =>
            point.precision !== null && point.recall !== null,
        ),
    [models],
  )

  const probabilityBandsData = useMemo(
    () =>
      (analysis?.probabilityBands ?? []).map((band) => {
        const total = safeNum(band.total) ?? 0
        const share = (value: unknown) => {
          const num = safeNum(value)
          if (num === null || !total) return 0
          return Number(((num / total) * 100).toFixed(2))
        }
        return {
          band: band.band,
          total,
          approvePct: share(band.approve),
          rejectPct: share(band.reject),
          deferPct: share(band.defer),
        }
      }),
    [analysis],
  )

  /** Derived per-model quality stats straight from the confusion counts. */
  const confusionDerived = useMemo(
    () =>
      confusionRows.map((row) => {
        const tp = safeNum(row.tp) ?? 0
        const fp = safeNum(row.fp) ?? 0
        const tn = safeNum(row.tn) ?? 0
        const fn = safeNum(row.fn) ?? 0
        const total = tp + fp + tn + fn
        const ratio = (numerator: number, denominator: number) =>
          denominator > 0 ? numerator / denominator : null
        return {
          model: row.model,
          tp,
          fp,
          tn,
          fn,
          total,
          accuracy: ratio(tp + tn, total),
          precision: ratio(tp, tp + fp),
          recall: ratio(tp, tp + fn),
          specificity: ratio(tn, tn + fp),
          falsePositiveRate: ratio(fp, fp + tn),
          f1FromCases: safeNum(row.f1FromCases),
        }
      }),
    [confusionRows],
  )

  const selectedConfusion = useMemo(
    () => confusionDerived.find((row) => row.model === confusionModel) ?? confusionDerived[0] ?? null,
    [confusionDerived, confusionModel],
  )

  const insightLines = useMemo(
    () => buildInsightLines(models, analysis?.summary?.bestModel ?? ''),
    [models, analysis],
  )

  const topAucModel = useMemo(() => {
    const ranked = models
      .filter((m) => safeNum(m.auc) !== null)
      .sort((a, b) => (safeNum(b.auc) ?? 0) - (safeNum(a.auc) ?? 0))
    return ranked[0]?.model ?? ''
  }, [models])

  const topRecallModel = useMemo(() => {
    const ranked = models
      .filter((m) => safeNum(m.recall) !== null)
      .sort((a, b) => (safeNum(b.recall) ?? 0) - (safeNum(a.recall) ?? 0))
    return ranked[0]?.model ?? ''
  }, [models])

  const topPrecisionModel = useMemo(() => {
    const ranked = models
      .filter((m) => safeNum(m.precision) !== null)
      .sort((a, b) => (safeNum(b.precision) ?? 0) - (safeNum(a.precision) ?? 0))
    return ranked[0]?.model ?? ''
  }, [models])

  const probColumns = analysis?.modelsByProbabilityColumns ?? []
  const casePageCount = Math.max(1, Math.ceil(cases.length / CASE_PAGE_SIZE))
  const clampedPage = Math.min(casePage, casePageCount - 1)
  const visibleCases = useMemo(
    () => cases.slice(clampedPage * CASE_PAGE_SIZE, clampedPage * CASE_PAGE_SIZE + CASE_PAGE_SIZE),
    [cases, clampedPage],
  )

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((dir) => (dir === 'asc' ? 'desc' : 'asc'))
      return
    }
    setSortKey(key)
    setSortDir(key === 'model' ? 'asc' : 'desc')
  }

  const handleModelChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = e.target.value
    if (!newModel) return

    setIsChangingModel(true)
    try {
      const result = await setActiveModel(newModel)
      setActiveModelState(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to switch model')
    } finally {
      setIsChangingModel(false)
    }
  }

  const sortIndicator = (key: SortKey) => (sortKey === key ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '')

  return (
    <DashboardLayout title="Model Analysis Dashboard" role="organization">
      {(error || (!analysis && !isLoading)) && (
        <section className="mb-6">
          <Card className="border-red-200 bg-red-50">
            <p className="text-red-700">{error ?? 'Model analysis unavailable'}</p>
          </Card>
        </section>
      )}

      <section className="mb-6 grid gap-6 lg:grid-cols-3">
        <Card title="Active Production Model" description="Select the core ML architecture for future loan decisions." className="col-span-1 border-primary-200 bg-primary-50/50">
          <div className="flex flex-col gap-4">
            <p className="text-sm text-neutral-600">
              The model selected here will process all new applications. Changing this does not affect historical applications.
            </p>
            <div className="flex flex-col gap-2">
              <label htmlFor="model-select" className="text-sm font-semibold text-neutral-900">
                Core ML Model
              </label>
              <select
                id="model-select"
                value={activeModel}
                onChange={handleModelChange}
                disabled={isChangingModel || isLoading || !models.length}
                className="w-full rounded-lg border border-neutral-300 bg-white px-4 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500 disabled:opacity-50"
              >
                {models.map((m) => (
                  <option key={m.model} value={m.model}>
                    {m.model} (AUC: {fmtRatioPct(m.auc)})
                  </option>
                ))}
              </select>
              {isChangingModel && <p className="text-xs text-primary-600 animate-pulse">Switching active model...</p>}
            </div>
            {models.length > 0 && (
              <div className="mt-2 rounded-lg bg-white p-3 text-sm shadow-sm">
                <span className="font-semibold text-primary-900">Business Impact:</span>{' '}
                {activeModel === topAucModel
                  ? 'This model provides the best overall balance of approvals and rejections based on historical performance (Highest AUC).'
                  : activeModel === topRecallModel
                  ? 'This model minimizes missing risky applicants (Highest Recall), making it the most conservative choice.'
                  : activeModel === topPrecisionModel
                  ? 'This model minimizes false approvals (Highest Precision), ensuring high confidence when approving.'
                  : 'This model offers a balanced approach, though not maximizing any single metric.'}
              </div>
            )}
          </div>
        </Card>

        <Card title="Hybrid Deferral Summary" description="Live totals plus saved evaluation artifact metrics." className="col-span-2">
          {isLoading || !analysis ? (
            <p className="text-neutral-600">Loading model analysis...</p>
          ) : (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
              <KPICard label="Live Total Cases" value={fmtInt(stats?.totalApplications)} />
              <KPICard label="Evaluation Artifact Cases" value={fmtInt(analysis.summary?.totalCases)} />
              <KPICard label="Deferred Cases" value={fmtInt(analysis.summary?.deferredCases)} />
              <KPICard label="Deferral Rate" value={fmtPct(analysis.summary?.deferralRate)} />
              <KPICard label="Automated Coverage" value={fmtPct(analysis.summary?.automatedCoverage)} />
              <KPICard label="Automated Accuracy" value={fmtPct(analysis.summary?.automatedAccuracy)} />
              <KPICard label="Hybrid Overall Accuracy" value={fmtPct(analysis.summary?.overallHybridAccuracy)} />
              <KPICard label="Best Model (Offline)" value={analysis.summary?.bestModel || '—'} />
            </div>
          )}
        </Card>
      </section>

      <section className="mb-6">
        <Card
          title="Model Leaderboard"
          description="Click any column to sort. Green cells are the best value for that metric; a dash means the metric was not produced for that model."
        >
          {isLoading ? (
            <p className="text-neutral-600">Loading leaderboard...</p>
          ) : !sortedModels.length ? (
            <p className="text-neutral-600">No model metrics available in the evaluation artifacts.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-neutral-200 text-left text-neutral-500">
                    <th className="px-3 py-2">
                      <button
                        type="button"
                        onClick={() => handleSort('model')}
                        className="font-medium hover:text-primary-700"
                      >
                        Model{sortIndicator('model')}
                      </button>
                    </th>
                    {METRIC_COLUMNS.map((column) => (
                      <th key={column.key} className="px-3 py-2" title={column.hint}>
                        <button
                          type="button"
                          onClick={() => handleSort(column.key)}
                          className="font-medium hover:text-primary-700"
                        >
                          {column.label}
                          {sortIndicator(column.key)}
                        </button>
                      </th>
                    ))}
                    <th className="px-3 py-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedModels.map((model) => {
                    const isActive = model.model === activeModel
                    return (
                      <tr
                        key={model.model}
                        className={`border-b border-neutral-100 ${isActive ? 'bg-primary-50/60' : ''}`}
                      >
                        <td className="px-3 py-2 font-medium text-neutral-800">{model.model}</td>
                        {METRIC_COLUMNS.map((column) => {
                          const value = safeNum(model[column.key])
                          const isBest =
                            value !== null &&
                            bestValues[column.key] !== null &&
                            value === bestValues[column.key]
                          return (
                            <td
                              key={column.key}
                              className={`px-3 py-2 tabular-nums ${
                                isBest
                                  ? 'font-semibold text-green-700 bg-green-50'
                                  : value === null
                                  ? 'text-neutral-400'
                                  : 'text-neutral-700'
                              }`}
                            >
                              {fmtRatioPct(value)}
                            </td>
                          )
                        })}
                        <td className="px-3 py-2 text-xs">
                          <span className="text-neutral-500">
                            {isActive ? 'Active' : `Rank ${fmtInt(model.rank)}`}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </section>

      <section className="mb-6 grid gap-6 lg:grid-cols-2">
        <Card title="AUC by Model" description="Primary ranking metric. Best model highlighted in green.">
          <div className="h-96">
            {aucChartData.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={aucChartData}
                  layout="vertical"
                  margin={{ top: 8, right: 32, left: 24, bottom: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" domain={[0, 100]} unit="%" />
                  <YAxis type="category" dataKey="model" width={130} />
                  <Tooltip formatter={(value: number) => [`${value}%`, 'AUC']} />
                  <Bar dataKey="auc" radius={[0, 4, 4, 0]}>
                    {aucChartData.map((row) => (
                      <Cell
                        key={row.model}
                        fill={bestAuc !== null && row.auc === bestAuc ? '#16a34a' : '#2563eb'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-neutral-600">No AUC values available.</p>
            )}
          </div>
        </Card>

        <Card title="Model Performance Comparison" description="All five metrics side by side per model.">
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelMetricsData} margin={{ top: 8, right: 16, left: 0, bottom: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="model" angle={-18} textAnchor="end" height={60} />
                <YAxis domain={[0, 100]} />
                <Tooltip formatter={(value: number) => [`${value}%`, '']} />
                <Legend />
                <Bar dataKey="accuracy" fill="#2563eb" radius={[4, 4, 0, 0]} />
                <Bar dataKey="precision" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                <Bar dataKey="recall" fill="#14b8a6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="auc" fill="#f97316" radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </section>

      <section className="mb-6 grid gap-6 lg:grid-cols-2">
        <Card
          title="Confusion Matrix"
          description="Per-model true/false decisions computed over every evaluation case."
        >
          {isLoading ? (
            <p className="text-neutral-600">Loading confusion summary...</p>
          ) : !selectedConfusion ? (
            <p className="text-neutral-600">No confusion counts available.</p>
          ) : (
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {confusionDerived.map((row) => (
                  <button
                    key={row.model}
                    type="button"
                    onClick={() => setConfusionModel(row.model)}
                    className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                      row.model === selectedConfusion.model
                        ? 'border-primary-500 bg-primary-50 text-primary-700'
                        : 'border-neutral-200 bg-white text-neutral-600 hover:border-primary-300'
                    }`}
                  >
                    {row.model}
                  </button>
                ))}
              </div>

              <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-sm">
                <div />
                <div className="text-center text-xs font-semibold uppercase text-neutral-500">Predicted approve</div>
                <div className="text-center text-xs font-semibold uppercase text-neutral-500">Predicted reject</div>

                <div className="flex items-center text-xs font-semibold uppercase text-neutral-500">Actual approve</div>
                <div className="rounded-lg border border-green-200 bg-green-50 px-3 py-4 text-center">
                  <p className="text-xs text-green-700">True Positive</p>
                  <p className="text-2xl font-bold text-green-800">{fmtInt(selectedConfusion.tp)}</p>
                </div>
                <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-4 text-center">
                  <p className="text-xs text-amber-700">False Negative</p>
                  <p className="text-2xl font-bold text-amber-800">{fmtInt(selectedConfusion.fn)}</p>
                </div>

                <div className="flex items-center text-xs font-semibold uppercase text-neutral-500">Actual reject</div>
                <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-4 text-center">
                  <p className="text-xs text-red-700">False Positive</p>
                  <p className="text-2xl font-bold text-red-800">{fmtInt(selectedConfusion.fp)}</p>
                </div>
                <div className="rounded-lg border border-neutral-200 bg-neutral-50 px-3 py-4 text-center">
                  <p className="text-xs text-neutral-600">True Negative</p>
                  <p className="text-2xl font-bold text-neutral-800">{fmtInt(selectedConfusion.tn)}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-3">
                {[
                  { label: 'Accuracy', value: selectedConfusion.accuracy },
                  { label: 'Precision', value: selectedConfusion.precision },
                  { label: 'Recall', value: selectedConfusion.recall },
                  { label: 'Specificity', value: selectedConfusion.specificity },
                  { label: 'False Positive Rate', value: selectedConfusion.falsePositiveRate },
                ].map((item) => (
                  <div key={item.label} className="rounded-lg border border-neutral-200 bg-neutral-50 px-3 py-2">
                    <p className="text-xs text-neutral-500">{item.label}</p>
                    <p className="font-semibold text-neutral-900">{fmtRatioPct(item.value)}</p>
                  </div>
                ))}
                <div className="rounded-lg border border-neutral-200 bg-neutral-50 px-3 py-2">
                  <p className="text-xs text-neutral-500">Cases scored</p>
                  <p className="font-semibold text-neutral-900">{fmtInt(selectedConfusion.total)}</p>
                </div>
              </div>
            </div>
          )}
        </Card>

        <Card
          title="Error Profile Across Models"
          description="Stacked false positives vs false negatives — the two failures that cost money."
        >
          <div className="h-80">
            {confusionDerived.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={confusionDerived} margin={{ top: 8, right: 16, left: 0, bottom: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="model" angle={-18} textAnchor="end" height={60} />
                  <YAxis />
                  <Tooltip formatter={(value: number, name: string) => [value.toLocaleString('en-IN'), name]} />
                  <Legend />
                  <Bar dataKey="fp" stackId="err" fill="#ef4444" name="False Positives (bad approvals)" />
                  <Bar dataKey="fn" stackId="err" fill="#f59e0b" name="False Negatives (missed good)" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-neutral-600">No confusion counts available.</p>
            )}
          </div>
          <div className="mt-3 overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-neutral-200 text-left text-neutral-500">
                  <th className="px-2 py-2">Model</th>
                  <th className="px-2 py-2">TP</th>
                  <th className="px-2 py-2">FP</th>
                  <th className="px-2 py-2">TN</th>
                  <th className="px-2 py-2">FN</th>
                  <th className="px-2 py-2">F1 (Cases)</th>
                </tr>
              </thead>
              <tbody>
                {confusionDerived.map((item) => (
                  <tr key={item.model} className="border-b border-neutral-100">
                    <td className="px-2 py-2 font-medium text-neutral-800">{item.model}</td>
                    <td className="px-2 py-2 tabular-nums">{fmtInt(item.tp)}</td>
                    <td className="px-2 py-2 tabular-nums">{fmtInt(item.fp)}</td>
                    <td className="px-2 py-2 tabular-nums">{fmtInt(item.tn)}</td>
                    <td className="px-2 py-2 tabular-nums">{fmtInt(item.fn)}</td>
                    <td className="px-2 py-2 tabular-nums">{fmtPct(item.f1FromCases)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </section>

      <section className="mb-6 grid gap-6 lg:grid-cols-2">
        <Card
          title="ML vs CBES Decision Landscape"
          description={
            cases.length > scatterData.length
              ? `Sampled ${scatterData.length.toLocaleString('en-IN')} of ${cases.length.toLocaleString('en-IN')} loaded cases to keep the chart responsive.`
              : 'Scatter view to inspect confidence and deferral zones.'
          }
        >
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis type="number" dataKey="x" name="ML Prob" unit="%" domain={[0, 100]} />
                <YAxis type="number" dataKey="y" name="CBES Prob" unit="%" domain={[0, 100]} />
                <ReferenceLine x={50} stroke="#94a3b8" strokeDasharray="4 4" />
                <ReferenceLine y={50} stroke="#94a3b8" strokeDasharray="4 4" />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  formatter={(value: number, name: string) => [`${value}%`, name]}
                  labelFormatter={() => 'Model point'}
                />
                <Legend />
                {(['APPROVE', 'REJECT', 'DEFER'] as const).map((decision) => (
                  <Scatter
                    key={decision}
                    name={decision}
                    data={scatterData.filter((point) => point.decision === decision)}
                    fill={DECISION_COLORS[decision]}
                    fillOpacity={0.55}
                    isAnimationActive={false}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Precision vs Recall Frontier" description="Model-selection tradeoff curve from tuned artifacts.">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis type="number" dataKey="recall" name="Recall" unit="%" domain={[0, 100]} />
                <YAxis type="number" dataKey="precision" name="Precision" unit="%" domain={[0, 100]} />
                <Tooltip
                  formatter={(value: number, name: string) => [`${value}%`, name]}
                  labelFormatter={() => 'Model Point'}
                />
                <Scatter data={precisionRecallData} fill="#2563eb" isAnimationActive={false}>
                  {precisionRecallData.map((point) => (
                    <Cell key={point.model} fill={point.auc !== null && point.auc >= 90 ? '#16a34a' : '#2563eb'} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {precisionRecallData.map((point) => (
              <div key={point.model} className="rounded-lg border border-neutral-200 bg-neutral-50 px-3 py-2 text-sm">
                <p className="font-medium text-neutral-900">{point.model}</p>
                <p className="text-neutral-700">
                  Precision {fmtPct(point.precision)} | Recall {fmtPct(point.recall)} | AUC {fmtPct(point.auc)}
                </p>
              </div>
            ))}
          </div>
        </Card>
      </section>

      <section className="mb-6 grid gap-6 lg:grid-cols-2">
        <Card title="Decision Mix by Probability Band" description="Shows how hybrid outcomes shift across model confidence bands.">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={probabilityBandsData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="band" />
                <YAxis domain={[0, 100]} />
                <Tooltip formatter={(value: number) => [`${value}%`, '']} />
                <Legend />
                <Bar dataKey="approvePct" stackId="a" fill="#10b981" name="Approve %" />
                <Bar dataKey="rejectPct" stackId="a" fill="#ef4444" name="Reject %" />
                <Bar dataKey="deferPct" stackId="a" fill="#f59e0b" name="Defer %" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Core ML Theory (Audit Notes)">
          <div className="space-y-3 text-sm leading-6 text-neutral-700">
            <p>
              Hybrid decisioning combines <strong>probabilistic classification</strong> (ML probability) and
              <strong> rule-based explainability</strong> (CBES score). Cases near decision boundaries are deferred to analysts to reduce wrong auto-decisions.
            </p>
            <p>
              Precision = TP / (TP + FP), Recall = TP / (TP + FN), and F1 = 2PR / (P + R). AUC measures ranking quality across thresholds.
            </p>
            <p>
              Deferral policy improves reliability by routing low-confidence cases for manual adjudication, maximizing automation where confidence is high and minimizing risk where uncertainty is high.
            </p>
            <p>
              Charts and the leaderboard use aggregates computed over the full evaluation set. The case table below shows a bounded
              sample so the browser stays responsive on the 307k-application Home Credit dataset.
            </p>
          </div>
        </Card>
      </section>

      <section className="mb-6">
        <Card
          title="Case-Level Drill Down"
          description={`Page ${clampedPage + 1} of ${casePageCount} — ${cases.length.toLocaleString('en-IN')} cases loaded (capped at ${CASE_LIMIT.toLocaleString('en-IN')}).`}
        >
          {isLoading ? (
            <p className="text-neutral-600">Loading cases...</p>
          ) : !cases.length ? (
            <p className="text-neutral-600">No case-level rows available.</p>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-neutral-200 text-left text-neutral-500">
                      <th className="px-2 py-2">Applicant</th>
                      <th className="px-2 py-2">Expected</th>
                      <th className="px-2 py-2">Hybrid</th>
                      <th className="px-2 py-2">Confidence</th>
                      <th className="px-2 py-2">Best model P</th>
                      <th className="px-2 py-2">CBES P</th>
                      {probColumns.map((column) => (
                        <th key={column} className="px-2 py-2">{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {visibleCases.map((item) => (
                      <tr key={item.applicantId} className="border-b border-neutral-100">
                        <td className="px-2 py-2 font-medium text-neutral-800">{item.applicantId || '—'}</td>
                        <td className="px-2 py-2 text-neutral-600">{item.expectedDecision ?? '—'}</td>
                        <td className="px-2 py-2">
                          <span
                            className="rounded-full px-2 py-0.5 text-xs font-semibold text-white"
                            style={{ backgroundColor: DECISION_COLORS[item.hybridDecision] ?? '#94a3b8' }}
                          >
                            {item.hybridDecision ?? '—'}
                          </span>
                        </td>
                        <td className="px-2 py-2 tabular-nums">{fmtRatioPct(item.hybridConfidence)}</td>
                        <td className="px-2 py-2 tabular-nums">{fmtRatioPct(item.bestModelProb)}</td>
                        <td className="px-2 py-2 tabular-nums">{fmtRatioPct(item.cbesProb)}</td>
                        {probColumns.map((column) => (
                          <td key={column} className="px-2 py-2 tabular-nums text-neutral-600">
                            {fmtRatioPct(item.modelProbabilities?.[column])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-4 flex items-center justify-between text-sm">
                <button
                  type="button"
                  onClick={() => setCasePage((page) => Math.max(0, page - 1))}
                  disabled={clampedPage === 0}
                  className="rounded-lg border border-neutral-300 bg-white px-3 py-1.5 font-medium text-neutral-700 disabled:opacity-40"
                >
                  Previous
                </button>
                <span className="text-neutral-600">
                  Showing {(clampedPage * CASE_PAGE_SIZE + 1).toLocaleString('en-IN')}–
                  {Math.min((clampedPage + 1) * CASE_PAGE_SIZE, cases.length).toLocaleString('en-IN')} of{' '}
                  {cases.length.toLocaleString('en-IN')}
                </span>
                <button
                  type="button"
                  onClick={() => setCasePage((page) => Math.min(casePageCount - 1, page + 1))}
                  disabled={clampedPage >= casePageCount - 1}
                  className="rounded-lg border border-neutral-300 bg-white px-3 py-1.5 font-medium text-neutral-700 disabled:opacity-40"
                >
                  Next
                </button>
              </div>
            </>
          )}
        </Card>
      </section>

      <section className="mb-6">
        <Card title="Tuning Insights">
          {isLoading || !analysis ? (
            <p className="text-neutral-600">Loading insights...</p>
          ) : !insightLines.length ? (
            <p className="text-neutral-600">Not enough model metrics to derive insights.</p>
          ) : (
            <ul className="space-y-3 text-sm text-neutral-700">
              {insightLines.map((line) => (
                <li key={line} className="rounded-lg border border-neutral-200 bg-neutral-50 p-3">
                  {line}
                </li>
              ))}
            </ul>
          )}
        </Card>
      </section>
    </DashboardLayout>
  )
}

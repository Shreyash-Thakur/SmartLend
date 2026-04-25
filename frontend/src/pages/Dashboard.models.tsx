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
import { getModelAnalysis, getStats } from '@/services/applications'
import type { ModelAnalysisResponse, ModelMetricItem, StatsResponse } from '@/types/api'

const DECISION_COLORS: Record<string, string> = {
  APPROVE: '#10b981',
  REJECT: '#ef4444',
  DEFER: '#f59e0b',
}

function asPct(value: number) {
  return Number((value * 100).toFixed(2))
}

function buildInsightLines(models: ModelMetricItem[], bestModel: string) {
  if (!models.length) {
    return []
  }
  const sortedByAuc = [...models].sort((a, b) => b.auc - a.auc)
  const first = sortedByAuc[0]
  const second = sortedByAuc[1]
  const aucGap = second ? asPct(first.auc - second.auc) : asPct(first.auc)

  const sortedByRecall = [...models].sort((a, b) => b.recall - a.recall)
  const sortedByPrecision = [...models].sort((a, b) => b.precision - a.precision)

  return [
    `${bestModel || first.model} leads on AUC with ${asPct(first.auc)}%. Margin vs next model: ${aucGap}pp.`,
    `${sortedByRecall[0].model} gives highest recall at ${asPct(sortedByRecall[0].recall)}%, useful when minimizing missed good applicants.`,
    `${sortedByPrecision[0].model} has strongest precision at ${asPct(sortedByPrecision[0].precision)}%, useful when controlling false approvals.`,
  ]
}

export const ModelAnalysisDashboard: React.FC = () => {
  const [analysis, setAnalysis] = useState<ModelAnalysisResponse | null>(null)
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loadAnalysis = async () => {
      setIsLoading(true)
      setError(null)
      try {
        const [analysisResponse, statsResponse] = await Promise.all([
          getModelAnalysis(50000),
          getStats(),
        ])
        setAnalysis(analysisResponse)
        setStats(statsResponse)
      } catch (fetchError) {
        setError(fetchError instanceof Error ? fetchError.message : 'Failed to load model analysis')
      } finally {
        setIsLoading(false)
      }
    }

    void loadAnalysis()
  }, [])

  const modelMetricsData = useMemo(
    () =>
      (analysis?.models ?? []).map((model) => ({
        model: model.model,
        accuracy: asPct(model.accuracy),
        precision: asPct(model.precision),
        recall: asPct(model.recall),
        auc: asPct(model.auc),
        f1: asPct(model.f1),
      })),
    [analysis],
  )

  const scatterData = useMemo(
    () =>
      (analysis?.cases ?? []).map((item) => ({
        applicantId: item.applicantId,
        x: Number((item.bestModelProb * 100).toFixed(2)),
        y: Number((item.cbesProb * 100).toFixed(2)),
        confidence: Number((item.hybridConfidence * 100).toFixed(2)),
        decision: item.hybridDecision,
      })),
    [analysis],
  )

  const precisionRecallData = useMemo(
    () =>
      (analysis?.models ?? []).map((model) => ({
        model: model.model,
        precision: asPct(model.precision),
        recall: asPct(model.recall),
        auc: asPct(model.auc),
      })),
    [analysis],
  )

  const probabilityBandsData = useMemo(
    () =>
      (analysis?.probabilityBands ?? []).map((band) => ({
        ...band,
        approvePct: band.total ? Number(((band.approve / band.total) * 100).toFixed(2)) : 0,
        rejectPct: band.total ? Number(((band.reject / band.total) * 100).toFixed(2)) : 0,
        deferPct: band.total ? Number(((band.defer / band.total) * 100).toFixed(2)) : 0,
      })),
    [analysis],
  )

  const insightLines = useMemo(
    () => buildInsightLines(analysis?.models ?? [], analysis?.summary.bestModel ?? ''),
    [analysis],
  )

  return (
    <DashboardLayout title="Model Analysis Dashboard" role="organization">
      {(error || (!analysis && !isLoading)) && (
        <section className="mb-6">
          <Card className="border-red-200 bg-red-50">
            <p className="text-red-700">{error ?? 'Model analysis unavailable'}</p>
          </Card>
        </section>
      )}

      <section className="mb-6">
        <Card title="Hybrid Deferral Summary" description="Live totals plus model-evaluation artifact metrics.">
          {isLoading || !analysis ? (
            <p className="text-neutral-600">Loading model analysis...</p>
          ) : (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
              <KPICard label="Live Total Cases" value={stats?.totalApplications ?? 0} />
              <KPICard label="Evaluation Sample Cases" value={analysis.summary.totalCases} />
              <KPICard label="Deferred Cases" value={analysis.summary.deferredCases} />
              <KPICard label="Deferral Rate" value={analysis.summary.deferralRate} format="percentage" />
              <KPICard label="Automated Coverage" value={analysis.summary.automatedCoverage} format="percentage" />
              <KPICard label="Automated Accuracy" value={analysis.summary.automatedAccuracy} format="percentage" />
              <KPICard label="Hybrid Overall Accuracy" value={analysis.summary.overallHybridAccuracy} format="percentage" />
              <KPICard label="Selected Best Model" value={analysis.summary.bestModel || 'N/A'} />
              <KPICard label="Selected Alpha" value={analysis.summary.selectedAlpha} />
            </div>
          )}
        </Card>
      </section>

      <section className="mb-6 grid gap-6 lg:grid-cols-2">
        <Card title="Model Performance Comparison" description="Similar to model_performance_comparison with additional F1 insight.">
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

        <Card title="ML vs CBES Decision Landscape" description="Scatter view to inspect confidence and deferral zones.">
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
                  >
                    {scatterData
                      .filter((point) => point.decision === decision)
                      .map((point) => (
                        <Cell key={`${decision}-${point.applicantId}`} fill={DECISION_COLORS[decision]} />
                      ))}
                  </Scatter>
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </section>

      <section className="mb-6 grid gap-6 lg:grid-cols-2">
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
                <Scatter data={precisionRecallData} fill="#2563eb">
                  {precisionRecallData.map((point) => (
                    <Cell key={point.model} fill={point.auc >= 90 ? '#16a34a' : '#2563eb'} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {precisionRecallData.map((point) => (
              <div key={point.model} className="rounded-lg border border-neutral-200 bg-neutral-50 px-3 py-2 text-sm">
                <p className="font-medium text-neutral-900">{point.model}</p>
                <p className="text-neutral-700">Precision {point.precision}% | Recall {point.recall}% | AUC {point.auc}%</p>
              </div>
            ))}
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
              This dashboard uses full artifact cases plus live application stats, so metrics are both theoretically grounded and operationally actionable.
            </p>
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

        <Card title="Confusion Matrix Summary" description="Per-model true/false decisions from case-level outputs.">
          {isLoading || !analysis ? (
            <p className="text-neutral-600">Loading confusion summary...</p>
          ) : (
            <div className="overflow-x-auto">
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
                  {analysis.confusionByModel.map((item) => (
                    <tr key={item.model} className="border-b border-neutral-100">
                      <td className="px-2 py-2 font-medium text-neutral-800">{item.model}</td>
                      <td className="px-2 py-2">{item.tp}</td>
                      <td className="px-2 py-2">{item.fp}</td>
                      <td className="px-2 py-2">{item.tn}</td>
                      <td className="px-2 py-2">{item.fn}</td>
                      <td className="px-2 py-2">{item.f1FromCases.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </section>

      <section className="mb-6">
        <Card title="Tuning Insights">
          {isLoading || !analysis ? (
            <p className="text-neutral-600">Loading insights...</p>
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

import React from 'react'
import { Card } from '@/components/common'
import { ApplicationDecision } from '@/types/application'

interface DecisionExplanationProps {
  decision: ApplicationDecision
}

export const DecisionExplanation: React.FC<DecisionExplanationProps> = ({ decision }) => {
  return (
    <Card title="Decision Analysis" className="mt-6">
      <div className="space-y-6">
        {/* Main Explanation */}
        <div>
          <h4 className="font-semibold text-neutral-900 mb-2">Why this decision?</h4>
          <p className="text-neutral-700 leading-relaxed">{decision.explanation}</p>
        </div>

        {/* Factors Supporting Decision */}
        {decision.positiveFactors.length > 0 && (
          <div>
            <h4 className={`font-semibold mb-3 flex items-center gap-2 ${
              decision.status === 'rejected' ? 'text-red-700' : 'text-green-700'
            }`}>
              <span>{decision.status === 'rejected' ? '⚠' : '✓'}</span> 
              {decision.status === 'rejected' ? 'Key Reasons for Rejection' : decision.status === 'approved' ? 'Key Reasons for Approval' : 'Positive Factors'}
            </h4>
            <ul className="space-y-2">
              {decision.positiveFactors.map((factor, idx) => (
                <li key={idx} className="text-sm text-neutral-700 flex items-start gap-3">
                  <span className={`mt-0.5 flex-shrink-0 ${
                    decision.status === 'rejected' ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {decision.status === 'rejected' ? '⚠' : '✓'}
                  </span>
                  <span>{factor}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Factors Opposing Decision */}
        {decision.negativeFactors.length > 0 && (
          <div>
            <h4 className={`font-semibold mb-3 flex items-center gap-2 ${
              decision.status === 'rejected' ? 'text-green-700' : 'text-red-700'
            }`}>
              <span>{decision.status === 'rejected' ? '✓' : '⚠'}</span> 
              {decision.status === 'rejected' ? 'Mitigating Factors' : 'Risk Factors'}
            </h4>
            <ul className="space-y-2">
              {decision.negativeFactors.map((factor, idx) => (
                <li key={idx} className="text-sm text-neutral-700 flex items-start gap-3">
                  <span className={`mt-0.5 flex-shrink-0 ${
                    decision.status === 'rejected' ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {decision.status === 'rejected' ? '✓' : '⚠'}
                  </span>
                  <span>{factor}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Metadata */}
        <div className="pt-4 border-t border-neutral-200">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-neutral-600 mb-1">Decided By</p>
              <p className="font-semibold text-neutral-900">{decision.decidedBy === 'model' ? '🤖 AI Model' : '👤 Analyst'}</p>
            </div>
            <div>
              <p className="text-neutral-600 mb-1">Decision Time</p>
              <p className="font-semibold text-neutral-900">{new Date(decision.decidedAt).toLocaleDateString()}</p>
            </div>
            {decision.modelVersion && (
              <div>
                <p className="text-neutral-600 mb-1">Model Version</p>
                <p className="font-semibold text-neutral-900">{decision.modelVersion}</p>
              </div>
            )}
            {decision.analystId && (
              <div>
                <p className="text-neutral-600 mb-1">Analyst ID</p>
                <p className="font-semibold text-neutral-900">{decision.analystId}</p>
              </div>
            )}
          </div>
        </div>

        {/* Analyst Notes */}
        {decision.analystNotes && (
          <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
            <p className="text-sm font-semibold text-blue-900 mb-2">Analyst Notes</p>
            <p className="text-sm text-blue-800">{decision.analystNotes}</p>
          </div>
        )}
      </div>
    </Card>
  )
}

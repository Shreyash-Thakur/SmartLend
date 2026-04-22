import React from 'react'
import { Card } from '@/components/common'
import { DecisionType, ConfidenceLevel } from '@/types/application'

interface DecisionBannerProps {
  status: DecisionType
  riskScore: number
  confidence: ConfidenceLevel
  timestamp: string
  decidedBy: 'model' | 'human'
}

const statusConfig = {
  approved: {
    bgColor: 'bg-gradient-to-r from-green-50 to-green-100',
    textColor: 'text-green-900',
    icon: '✓',
    title: 'Loan Approved',
    description: 'Your loan application has been approved',
  },
  rejected: {
    bgColor: 'bg-gradient-to-r from-red-50 to-red-100',
    textColor: 'text-red-900',
    icon: '✗',
    title: 'Application Rejected',
    description: 'Unfortunately, your application could not be approved',
  },
  deferred: {
    bgColor: 'bg-gradient-to-r from-amber-50 to-amber-100',
    textColor: 'text-amber-900',
    icon: '⏳',
    title: 'Under Review',
    description: 'Your application is under review by our team',
  },
}

export const DecisionBanner: React.FC<DecisionBannerProps> = ({
  status,
  riskScore,
  confidence,
  timestamp,
  decidedBy,
}) => {
  const config = statusConfig[status]

  return (
    <div className={`${config.bgColor} rounded-xl p-8 border-2 ${config.textColor === 'text-green-900' ? 'border-green-200' : config.textColor === 'text-red-900' ? 'border-red-200' : 'border-amber-200'}`}>
      <div className="flex items-start gap-4">
        <div className="text-4xl">{config.icon}</div>
        
        <div className="flex-1">
          <h2 className={`text-2xl font-bold ${config.textColor} mb-2`}>{config.title}</h2>
          <p className={`${config.textColor} opacity-80 mb-4`}>{config.description}</p>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            <div>
              <p className={`text-sm ${config.textColor} opacity-70`}>Risk Score</p>
              <p className="text-lg font-semibold text-neutral-900">{(riskScore * 100).toFixed(1)}%</p>
            </div>
            
            <div>
              <p className={`text-sm ${config.textColor} opacity-70`}>Confidence</p>
              <p className="text-lg font-semibold text-neutral-900 capitalize">{confidence}</p>
            </div>
            
            <div>
              <p className={`text-sm ${config.textColor} opacity-70`}>Decided By</p>
              <p className="text-lg font-semibold text-neutral-900">{decidedBy === 'model' ? '🤖 AI' : '👤 Analyst'}</p>
            </div>
            
            <div>
              <p className={`text-sm ${config.textColor} opacity-70`}>Decision Time</p>
              <p className="text-lg font-semibold text-neutral-900">{new Date(timestamp).toLocaleDateString()}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

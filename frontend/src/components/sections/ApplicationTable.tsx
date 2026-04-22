import React, { useMemo, useState } from 'react'
import { Badge, Card, Select } from '@/components/common'
import type { ApplicationTableProps } from '@/types/ui'
import { formatCurrency, formatDate } from '@/lib/utils'

export const ApplicationTable: React.FC<ApplicationTableProps> = ({
  data,
  onRowClick,
  isLoading = false,
  pageSize = 10,
  showApplicant = false,
}) => {
  const [page, setPage] = useState(1)
  const [sortBy, setSortBy] = useState<'date' | 'amount' | 'status'>('date')
  const [filterStatus, setFilterStatus] = useState<string>('all')

  const sortedApps = useMemo(() => {
    const filteredApps =
      filterStatus === 'all' ? data : data.filter((app) => app.status === filterStatus)

    return [...filteredApps].sort((a, b) => {
      switch (sortBy) {
        case 'amount':
          return b.loanAmount - a.loanAmount
        case 'status':
          return a.status.localeCompare(b.status)
        default:
          return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      }
    })
  }, [data, filterStatus, sortBy])

  const totalPages = Math.max(1, Math.ceil(sortedApps.length / pageSize))
  const paginatedApps = sortedApps.slice((page - 1) * pageSize, page * pageSize)

  return (
    <Card title="Application History">
      <div className="space-y-4">
        <div className="flex flex-col gap-4 border-b border-neutral-200 pb-4 sm:flex-row">
          <Select
            value={filterStatus}
            onChange={(value) => {
              setFilterStatus(String(value))
              setPage(1)
            }}
            options={[
              { value: 'all', label: 'All Status' },
              { value: 'draft', label: 'Draft' },
              { value: 'submitted', label: 'Submitted' },
              { value: 'processing', label: 'Processing' },
              { value: 'approved', label: 'Approved' },
              { value: 'rejected', label: 'Rejected' },
              { value: 'deferred', label: 'Deferred' },
            ]}
          />
          <Select
            value={sortBy}
            onChange={(value) => setSortBy(value as 'date' | 'amount' | 'status')}
            options={[
              { value: 'date', label: 'Sort by Date' },
              { value: 'amount', label: 'Sort by Amount' },
              { value: 'status', label: 'Sort by Status' },
            ]}
          />
        </div>

        {isLoading ? (
          <div className="text-center py-8">
            <svg className="animate-spin h-8 w-8 text-primary-500 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <p className="text-neutral-600 mt-2">Loading applications...</p>
          </div>
        ) : paginatedApps.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-neutral-600">No applications found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-neutral-200 bg-neutral-50">
                  <th className="px-4 py-3 text-left text-sm font-semibold text-neutral-900">Application ID</th>
                  {showApplicant && (
                    <th className="px-4 py-3 text-left text-sm font-semibold text-neutral-900">Applicant</th>
                  )}
                  <th className="px-4 py-3 text-left text-sm font-semibold text-neutral-900">Loan Amount</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-neutral-900">Status</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-neutral-900">Submitted</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-neutral-900">Purpose</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-neutral-900">Decision</th>
                </tr>
              </thead>
              <tbody>
                {paginatedApps.map((app) => (
                  <tr
                    key={app.id}
                    onClick={() => onRowClick?.(app)}
                    className="border-b border-neutral-200 hover:bg-neutral-50 transition-colors cursor-pointer"
                  >
                    <td className="px-4 py-3 text-sm font-medium text-primary-600">{app.id.slice(0, 8)}...</td>
                    {showApplicant && (
                      <td className="px-4 py-3 text-sm text-neutral-900">{app.applicantName}</td>
                    )}
                    <td className="px-4 py-3 text-sm text-neutral-900">{formatCurrency(app.loanAmount)}</td>
                    <td className="px-4 py-3 text-sm">
                      <Badge status={app.status === 'processing' ? 'pending' : app.status} />
                    </td>
                    <td className="px-4 py-3 text-sm text-neutral-600">{formatDate(app.createdAt)}</td>
                    <td className="px-4 py-3 text-sm capitalize text-neutral-600">{app.loanPurpose}</td>
                    <td className="px-4 py-3 text-sm text-neutral-600 capitalize">
                      {app.decision?.status ?? 'Pending'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {!isLoading && totalPages > 1 && (
          <div className="flex items-center justify-between border-t border-neutral-200 pt-4 text-sm text-neutral-600">
            <span>
              Page {page} of {totalPages}
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                className="rounded-lg border border-neutral-300 px-3 py-2 disabled:opacity-50"
                onClick={() => setPage((value) => Math.max(1, value - 1))}
                disabled={page === 1}
              >
                Previous
              </button>
              <button
                type="button"
                className="rounded-lg border border-neutral-300 px-3 py-2 disabled:opacity-50"
                onClick={() => setPage((value) => Math.min(totalPages, value + 1))}
                disabled={page === totalPages}
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}

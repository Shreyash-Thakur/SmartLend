import React from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Clock3, ShieldCheck, Sparkles, Users, Workflow, BarChart3 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Button, Card } from '@/components/common'
import { getPublicMetrics } from '@/services/applications'
import type { PublicMetrics } from '@/types/api'
import { PageTransition } from '@/components/layouts'

export const Landing: React.FC = () => {
  const navigate = useNavigate()
  const [metrics, setMetrics] = useState<PublicMetrics | null>(null)

  const features = [
    {
      id: 'speed',
      icon: Clock3,
      title: 'Faster Approvals',
      description: 'AI-powered decisions in seconds, not days',
    },
    {
      id: 'risk',
      icon: BarChart3,
      title: 'AI Risk Prediction',
      description: 'ML model trained on 100k+ historical decisions',
    },
    {
      id: 'explainability',
      icon: Sparkles,
      title: 'CBES Score (Explainability)',
      description: 'Understand why decisions are made',
    },
    {
      id: 'uncertainty',
      icon: Workflow,
      title: 'Uncertainty Detection',
      description: 'Confidence-aware decisions',
    },
    {
      id: 'review',
      icon: Users,
      title: 'Human-in-the-Loop',
      description: 'Human analysts review deferred cases',
    },
    {
      id: 'compliance',
      icon: ShieldCheck,
      title: 'Compliance Ready',
      description: 'Audit trails for regulatory requirements',
    },
  ]

  useEffect(() => {
    void getPublicMetrics().then(setMetrics)
  }, [])

  return (
    <PageTransition>
    <div className="min-h-screen bg-gradient-to-b from-white to-neutral-50">
      <header className="sticky top-0 z-40 bg-white/80 backdrop-blur-md border-b border-neutral-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-accent-500 rounded-lg" />
              <span className="font-bold text-lg text-neutral-900">SmartLend</span>
            </div>
            <div className="flex items-center gap-4">
              <Button variant="ghost" onClick={() => navigate('/dashboard/customer')}>
                Dashboard
              </Button>
              <Button variant="primary" onClick={() => navigate('/dashboard/customer')}>
                Get Started
              </Button>
            </div>
          </div>
        </div>
      </header>

      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center space-y-6"
        >
          <h1 className="text-5xl md:text-6xl font-bold text-neutral-900 leading-tight">
            Smarter Loan Decisions. Backed by Intelligence. Guided by Trust.
          </h1>
          <p className="text-xl text-neutral-600 max-w-2xl mx-auto">
            Instant risk assessment with human oversight. Approve loans 2-3x faster while reducing decision errors.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Button
              variant="primary"
              size="lg"
              onClick={() => navigate('/dashboard/customer')}
            >
              Start Application
            </Button>
            <Button
              variant="secondary"
              size="lg"
              onClick={() => navigate('/dashboard/org')}
            >
              View Dashboard
            </Button>
          </div>
        </motion.div>
      </section>

      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <h2 className="text-4xl font-bold text-center text-neutral-900 mb-12">
          Why SmartLend?
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
            <motion.div
              key={feature.id}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: index * 0.1 }}
              viewport={{ once: true, amount: 0.2 }}
            >
            <Card withGlass={false}>
              <div className="space-y-4">
                <div className="inline-flex rounded-2xl bg-primary-50 p-3 text-primary-600">
                  <Icon className="h-7 w-7" />
                </div>
                <h3 className="text-xl font-semibold text-neutral-900">
                  {feature.title}
                </h3>
                <p className="text-neutral-600">{feature.description}</p>
              </div>
            </Card>
            </motion.div>
          )})}
        </div>
      </section>

      <section className="bg-gradient-to-r from-primary-500 to-accent-500 py-20 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center text-white">
            <div>
              <p className="text-4xl font-bold">{metrics?.applicationsProcessed ?? 0}+</p>
              <p className="text-lg mt-2">Applications Processed</p>
            </div>
            <div>
              <p className="text-4xl font-bold">{metrics?.accuracy ?? 0}%</p>
              <p className="text-lg mt-2">Accuracy Rate</p>
            </div>
            <div>
              <p className="text-4xl font-bold">{metrics?.approvalSpeedup ?? 0}x</p>
              <p className="text-lg mt-2">Faster Approvals</p>
            </div>
            <div>
              <p className="text-4xl font-bold">{metrics?.automationRate ?? 0}%</p>
              <p className="text-lg mt-2">Automation Rate</p>
            </div>
          </div>
        </div>
      </section>

      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid gap-6 lg:grid-cols-2">
          <Card title="Traditional Process" description="Application -> Manual Review (days) -> Binary Decision -> Closed">
            <p className="text-neutral-600">
              Legacy lending workflows create long wait times, limited transparency, and a hard
              yes-or-no outcome with little explainability.
            </p>
          </Card>
          <Card title="Smart Process" description="Application -> AI Assessment -> Confidence Check -> Instant Decision or Expert Review">
            <p className="text-neutral-600">
              SmartLend adds explainability, uncertainty detection, and expert review only when the
              model needs human support.
            </p>
          </Card>
        </div>
      </section>

      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 text-center">
        <h2 className="text-3xl font-bold text-neutral-900 mb-6">
          Ready to Transform Your Lending?
        </h2>
        <p className="text-lg text-neutral-600 mb-8 max-w-2xl mx-auto">
          Join hundreds of lenders already using SmartLend to make smarter, faster loan decisions.
        </p>
        <Button
          variant="primary"
          size="lg"
          onClick={() => navigate('/dashboard/customer')}
        >
          Get Started Today
        </Button>
      </section>

      {/* Footer */}
      <footer className="bg-neutral-900 text-neutral-400 py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center text-sm">
            <p>&copy; 2026 SmartLend. All rights reserved.</p>
            <p className="mt-2">Intelligent Loan Assessment Platform</p>
          </div>
        </div>
      </footer>
    </div>
    </PageTransition>
  )
}

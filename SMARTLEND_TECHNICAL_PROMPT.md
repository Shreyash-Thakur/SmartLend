# 🏗️ SMARTLEND / CREDDFER FINTECH UI — TECHNICAL SPECIFICATION

**Version:** 1.0 Production  
**Target Stack:** React 18+, TypeScript, Tailwind CSS 3.x, Framer Motion  
**Design System:** Fintech-grade (Stripe / Revolut inspired)

---

## 📋 TABLE OF CONTENTS

1. [Architecture Overview](#architecture-overview)
2. [Design System Specification](#design-system-specification)
3. [Component Architecture](#component-architecture)
4. [Page Specifications](#page-specifications)
5. [Data Contracts & API Integration](#data-contracts--api-integration)
6. [Backend Integration Points](#backend-integration-points)
7. [State Management](#state-management)
8. [Performance & Accessibility](#performance--accessibility)

---

## 🏗️ ARCHITECTURE OVERVIEW

### Tech Stack (Non-Negotiable)

```json
{
  "framework": "React 18+ with TypeScript",
  "styling": "Tailwind CSS 3.x (utility-first)",
  "motion": "Framer Motion (page transitions, scroll triggers)",
  "forms": "React Hook Form + Zod (validation)",
  "state": "Zustand (lightweight) or Context API",
  "http": "Axios with retry logic",
  "charting": "Recharts or Chart.js (not custom SVG)",
  "bundler": "Vite (not Create React App)"
}
```

### Project Structure

```
src/
├── components/
│   ├── common/           # Reusable UI primitives
│   ├── layouts/          # Page wrappers
│   ├── sections/         # Page-specific sections
│   └── forms/            # Loan forms, validation
├── pages/
│   ├── Landing.tsx
│   ├── Dashboard.customer.tsx
│   ├── Dashboard.org.tsx
│   └── ApplicationReview.tsx
├── hooks/
│   ├── useFormValidation.ts
│   ├── useApplicationData.ts
│   └── useDecisionExplanation.ts
├── types/
│   ├── application.ts    # Data models
│   ├── api.ts            # API response shapes
│   └── ui.ts             # Component props
├── services/
│   ├── api.client.ts     # HTTP client
│   ├── fileParser.ts     # PDF/CSV parsing
│   └── analytics.ts      # Logging
├── store/
│   ├── applicationStore.ts
│   ├── authStore.ts
│   └── uiStore.ts
├── styles/
│   └── globals.css       # Tailwind + design tokens
├── lib/
│   └── utils.ts          # Helper functions
└── App.tsx
```

---

## 🎨 DESIGN SYSTEM SPECIFICATION

### Color Palette (Tokens)

```css
/* Primary - Trust & Finance */
--color-primary-50: #f0fdf4    /* Mint very light */
--color-primary-100: #dcfce7
--color-primary-500: #16a34a    /* Main brand green */
--color-primary-600: #15803d
--color-primary-900: #14532d

/* Secondary - Accent */
--color-accent-50: #f0f9ff    /* Teal very light */
--color-accent-500: #0ea5e9   /* Teal accent */

/* Neutral - Structure */
--color-neutral-50: #f9fafb
--color-neutral-100: #f3f4f6
--color-neutral-200: #e5e7eb
--color-neutral-500: #6b7280
--color-neutral-900: #111827

/* Semantic */
--color-success: #10b981
--color-warning: #f59e0b
--color-error: #ef4444
--color-info: #3b82f6
--color-deferred: #ec4899  /* Pink for deferral */
```

### Typography Scale (Tailwind + Custom)

```css
/* Font Family */
--font-sans: 'Inter', 'SF Pro Display', -apple-system, sans-serif

/* Scale */
h1: 2.5rem (40px), font-weight: 700, line-height: 1.2
h2: 2rem (32px), font-weight: 600, line-height: 1.3
h3: 1.5rem (24px), font-weight: 600, line-height: 1.4
body-lg: 1.125rem (18px), font-weight: 400, line-height: 1.6
body: 1rem (16px), font-weight: 400, line-height: 1.6
body-sm: 0.875rem (14px), font-weight: 400, line-height: 1.5
label: 0.875rem (14px), font-weight: 500, line-height: 1.5
caption: 0.75rem (12px), font-weight: 400, line-height: 1.5
```

### Spacing System (Tailwind Standard)

```
4px (0.25rem) / 8px (0.5rem) / 12px (0.75rem) / 16px / 24px / 32px / 48px / 64px
```

### Border Radius (Fintech Standard)

```css
--radius-sm: 8px
--radius-md: 12px
--radius-lg: 16px
--radius-xl: 24px
--radius-full: 9999px
```

### Shadow System (Glassmorphism)

```css
/* Subtle elevation for premium feel */
--shadow-sm: 0 1px 2px rgba(0,0,0,0.05)
--shadow-md: 0 4px 6px rgba(0,0,0,0.07)
--shadow-lg: 0 10px 15px rgba(0,0,0,0.10)

/* Glassmorphism backgrounds */
--glass-bg: rgba(255, 255, 255, 0.8)
--glass-backdrop: blur(12px)
```

### Motion Tokens (Framer Motion)

```ts
// Standard transitions
transition.fast = 0.15s     // Micro interactions
transition.normal = 0.3s    // Button hover, card entrance
transition.slow = 0.6s      // Page transitions, scroll triggers

// Easing (cubic-bezier)
easing.easeIn = [0.4, 0, 1, 1]
easing.easeOut = [0, 0, 0.2, 1]
easing.easeInOut = [0.4, 0, 0.2, 1]
```

---

## 🧩 COMPONENT ARCHITECTURE

### Component Hierarchy & Responsibility

#### **Level 1: Primitives (UI Atoms)**

```tsx
// components/common/Button.tsx
interface ButtonProps {
  variant: 'primary' | 'secondary' | 'ghost' | 'danger'
  size: 'sm' | 'md' | 'lg'
  isLoading?: boolean
  disabled?: boolean
  fullWidth?: boolean
}

// components/common/Card.tsx
interface CardProps {
  className?: string
  withGlass?: boolean  // Glassmorphism effect
}

// components/common/Input.tsx
interface InputProps {
  label?: string
  error?: string
  hint?: string
  required?: boolean
}

// components/common/Badge.tsx
interface BadgeProps {
  status: 'approved' | 'rejected' | 'deferred' | 'pending'
}

// components/common/Modal.tsx
interface ModalProps {
  isOpen: boolean
  onClose: () => void
  size: 'sm' | 'md' | 'lg'
}
```

#### **Level 2: Composite Components**

```tsx
// components/forms/LoanApplicationForm.tsx
// Handles: validation, field binding, error display, file upload integration

// components/sections/MetricsGrid.tsx
// Handles: 4-column metric cards, animation on load

// components/sections/ApplicationTable.tsx
// Handles: sorting, filtering, pagination, inline actions

// components/sections/DecisionExplanation.tsx
// Handles: CBES score visualization, feature contribution display
```

#### **Level 3: Page Layouts**

```tsx
// components/layouts/DashboardLayout.tsx
// - Top navigation
// - Sidebar (if applicable)
// - Breadcrumbs
// - Content area

// components/layouts/PageTransition.tsx
// - Framer Motion wrapper for page entrance/exit
```

---

## 📄 PAGE SPECIFICATIONS

### **PAGE 1: LANDING PAGE** (`pages/Landing.tsx`)

#### **1.1 Hero Section**

```tsx
interface HeroSection {
  title: "Smarter Loan Decisions. Backed by Intelligence. Guided by Trust."
  subtitle: "Instant risk assessment with human oversight. Approve loans 2-3x faster while reducing decision errors."
  cta: [
    { label: "Start Application", href: "/dashboard/customer", variant: "primary" },
    { label: "View Dashboard", href: "/login", variant: "secondary" }
  ]
  backgroundAnimation: "soft gradient parallax" // Framer Motion
  heroImage?: "optional - fintech dashboard mockup"
}
```

**Technical Implementation:**

```tsx
// Hero uses motion.div for parallax on scroll
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.6 }}
  className="min-h-screen flex flex-col justify-center"
>
```

#### **1.2 Feature Cards Section** (3-column grid)

```tsx
interface FeatureCard {
  id: string
  icon: React.ComponentType   // SVG icon
  title: string              // "Faster Approvals"
  description: string        // "2-3x speed improvement over manual review"
  metrics?: {
    value: string           // "2.5x"
    label: string           // "Average speedup"
  }
}

const features: FeatureCard[] = [
  {
    id: "speed",
    title: "Faster Approvals",
    description: "AI-powered decisions in seconds, not days"
  },
  {
    id: "risk",
    title: "AI Risk Prediction",
    description: "ML model trained on 100k+ historical decisions"
  },
  {
    id: "explainability",
    title: "CBES Score (Explainability)",
    description: "Understand why decisions are made"
  },
  {
    id: "uncertainty",
    title: "Uncertainty Detection",
    description: "Confidence-aware decisions"
  },
  {
    id: "review",
    title: "Human-in-the-Loop",
    description: "Human analysts review deferred cases"
  },
  {
    id: "compliance",
    title: "Compliance Ready",
    description: "Audit trails for regulatory requirements"
  }
]
```

**Technical Details:**

- Scroll trigger: Cards fade in on scroll (use Framer Motion `useViewportScroll` or `whileInView`)
- Stagger animation: Each card enters 100ms after previous
- Responsive: 1 column (mobile), 2 columns (tablet), 3 columns (desktop)

#### **1.3 Metrics Section** (Call-to-Action with Placeholders)

```tsx
interface MetricsDisplay {
  data: [
    { label: "Applications Processed", value: 0, unit: "K+", status: "placeholder" },
    { label: "Approval Speedup", value: 0, unit: "x", status: "placeholder" },
    { label: "Decision Accuracy", value: 0, unit: "%", status: "placeholder" },
    { label: "Automation Rate", value: 0, unit: "%", status: "placeholder" }
  ]
}
```

**Backend Contract:**

```
GET /api/metrics/public
Response:
{
  "applicationsProcessed": 1250,
  "approvalSpeedup": 2.3,
  "accuracy": 94.2,
  "automationRate": 78
}
```

#### **1.4 How It Works Section** (Flow diagram text-based)

```
Traditional Process:
Application → Manual Review (days) → Binary Decision → Closed

Smart Process:
Application → AI Assessment (seconds) → Confidence Check → 
  If High Confidence → Instant Decision
  If Low Confidence → Expert Review → Final Decision
```

#### **1.5 CTA Section**

```tsx
<section className="bg-gradient-to-r from-primary-500 to-accent-500 py-16">
  <h2>Ready to Transform Your Lending?</h2>
  <Button variant="primary" size="lg">Get Started Today</Button>
</section>
```

---

### **PAGE 2: CUSTOMER DASHBOARD** (`pages/Dashboard.customer.tsx`)

#### **2.1 Data Model**

```ts
interface LoanApplication {
  id: string                    // UUID
  createdAt: ISO8601
  status: 'draft' | 'submitted' | 'processing' | 'approved' | 'rejected' | 'deferred'
  loanAmount: number            // In rupees
  applicationData: {
    income: number
    emi: number
    assets: number
    creditHistory?: string
    employment?: string
    age?: number
    dependents?: number
    loanPurpose?: string
  }
  decision?: {
    status: 'approved' | 'rejected' | 'deferred'
    riskScore: number           // 0-1 confidence
    cbessScore: number          // 0-100 explainability
    uncertainty: number         // 0-1 model uncertainty
    explanation: string         // Human-readable reason
    timestamp: ISO8601
  }
  uploadedFiles?: {
    documentType: 'pdf' | 'csv'
    fileName: string
    uploadedAt: ISO8601
  }[]
}
```

#### **2.2 Layout Structure**

```tsx
<DashboardLayout>
  {/* Header with user greeting + logout */}
  <header>
    <h1>Welcome, {userName}</h1>
    <button onClick={logout}>Logout</button>
  </header>

  {/* Two-column layout */}
  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
    
    {/* Left: New Application Form (2/3) */}
    <section className="lg:col-span-2">
      <Card title="New Loan Application">
        <LoanApplicationForm onSubmit={handleSubmit} />
      </Card>
    </section>

    {/* Right: Quick Stats (1/3) */}
    <aside>
      <Card title="Your Applications">
        <ApplicationSummary applications={applications} />
      </Card>
    </aside>
  </div>

  {/* Full-width: Application History Table */}
  <section className="mt-8">
    <ApplicationTable 
      applications={applications}
      onRowClick={viewDetails}
    />
  </section>
</DashboardLayout>
```

#### **2.3 Loan Application Form**

**Fields (Must Match ML Input Layer):**

```tsx
interface ApplicationFormData {
  // Income & Employment
  monthlyIncome: number         // Required
  emi: number                   // Existing EMI obligations
  
  // Assets & Liabilities
  assets: number                // Total assets
  liabilities?: number
  
  // Credit Profile
  creditScore?: number          // CIBIL score
  creditHistory?: 'good' | 'average' | 'poor'
  
  // Demographics
  age: number
  dependents?: number
  employmentType?: 'salaried' | 'self-employed' | 'business'
  
  // Loan Details
  loanAmount: number            // Requested amount
  loanPurpose?: 'home' | 'auto' | 'personal' | 'business'
  loanTenure?: number           // In months
  
  // Documents
  documents?: File[]
}
```

**Validation Rules:**

```ts
const validationSchema = {
  monthlyIncome: 'required | number | min:15000 | max:500000',
  emi: 'required | number | min:0 | max:{monthlyIncome * 0.4}',
  assets: 'required | number | min:0',
  loanAmount: 'required | number | min:100000 | max:10000000',
  age: 'required | number | min:21 | max:65'
}
```

**Form UX Requirements:**

- Multi-step form (optional): Step 1 (Income), Step 2 (Assets), Step 3 (Documents)
- Auto-save to localStorage after each field change
- Progress indicator if multi-step
- File upload area with drag-drop support
- Real-time validation feedback (no submit if invalid)
- Submit button disabled until valid

#### **2.4 File Upload Integration**

```tsx
interface FileUploadProps {
  onFileSelect: (file: File) => void
  acceptedFormats: ['pdf', 'csv']
  maxSize: 5 * 1024 * 1024  // 5MB
}

// On file upload:
const handleFileUpload = async (file: File) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('applicationId', currentAppId)
  
  const response = await axios.post('/api/applications/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  
  // Returns: { extractedData: {...}, fileName, uploadedAt }
}
```

#### **2.5 Application History Table**

```tsx
interface ApplicationTableProps {
  columns: [
    { key: 'id', label: 'Application ID', sortable: true },
    { key: 'loanAmount', label: 'Loan Amount', sortable: true, format: 'currency' },
    { key: 'status', label: 'Status', render: <Badge /> },
    { key: 'submittedAt', label: 'Date', sortable: true, format: 'date' },
    { key: 'decision', label: 'Decision', render: <DecisionBadge /> }
  ]
  rows: LoanApplication[]
  onRowClick: (id) => void
}
```

**Table Features:**

- Sortable columns (default: newest first)
- Filterable by status (dropdown)
- Pagination (10 rows per page)
- Row hover effect (subtle highlight)
- Click row → Navigate to detail view

---

### **PAGE 3: ORGANIZATION DASHBOARD** (`pages/Dashboard.org.tsx`)

#### **3.1 Layout**

```tsx
<DashboardLayout role="organization">
  {/* Top KPI Cards */}
  <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
    <KPICard 
      label="Total Applications" 
      value={dashboardMetrics.totalApplications}
      trend="+12%" 
    />
    <KPICard label="Approved" value={dashboardMetrics.approved} />
    <KPICard label="Rejected" value={dashboardMetrics.rejected} />
    <KPICard label="Deferred" value={dashboardMetrics.deferred} />
  </section>

  {/* Charts Section */}
  <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
    <Chart type="pie" data={approvalDistribution} />
    <Chart type="line" data={applicationsOverTime} />
    <Chart type="bar" data={categoryAnalysis} />
    <Chart type="bar" data={riskScoreDistribution} />
  </section>

  {/* Tabs: View All / Review Deferred */}
  <section className="mt-8">
    <Tabs defaultValue="all">
      <TabsContent value="all">
        <ApplicationTable showAllApplications={true} />
      </TabsContent>
      <TabsContent value="deferred">
        <ApplicationTable 
          filterStatus="deferred"
          description="Applications awaiting human review"
        />
      </TabsContent>
    </Tabs>
  </section>
</DashboardLayout>
```

#### **3.2 KPI Card Component**

```tsx
interface KPICard {
  label: string
  value: number
  trend?: string     // e.g., "+12%" or "-3%"
  format?: 'number' | 'currency' | 'percentage'
  icon?: React.ComponentType
}
```

#### **3.3 Chart Specifications**

**Chart 1: Approval Distribution (Pie)**

```ts
{
  labels: ['Approved', 'Rejected', 'Deferred', 'Processing'],
  datasets: [{
    data: [65, 20, 10, 5],
    backgroundColor: [
      '#10b981',  // green
      '#ef4444',  // red
      '#ec4899',  // pink
      '#f59e0b'   // amber
    ]
  }]
}
```

**Chart 2: Applications Over Time (Line)**

```ts
{
  labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
  datasets: [{
    label: 'Applications Received',
    data: [120, 145, 160, 210],
    borderColor: '#16a34a',
    fill: true,
    backgroundColor: 'rgba(22, 163, 74, 0.1)'
  }]
}
```

**Chart 3: Category Analysis (Bar)**

```ts
{
  labels: ['Home Loan', 'Auto Loan', 'Personal Loan', 'Business Loan'],
  datasets: [{
    label: 'Loan Count',
    data: [450, 320, 280, 150],
    backgroundColor: '#0ea5e9'
  }]
}
```

#### **3.4 Application Table (Org View)**

Enhanced version with:

```tsx
interface OrgApplicationTable {
  columns: [
    { key: 'id', label: 'Application ID' },
    { key: 'applicantName', label: 'Applicant' },
    { key: 'loanAmount', label: 'Loan Amount', format: 'currency' },
    { key: 'riskScore', label: 'Risk Score', render: <RiskScoreBar /> },
    { key: 'status', label: 'Status', render: <Badge /> },
    { key: 'submittedAt', label: 'Submitted' },
    { key: 'actions', label: 'Actions', render: <ActionMenu /> }
  ]
  features: {
    filtering: true         // Status, risk level, date range
    sorting: true           // All columns
    pagination: true        // 25 rows per page
    rowExpand: true         // Click to expand details inline
  }
}
```

**Action Menu Options:**

```tsx
<ActionMenu>
  <MenuItem onClick={() => navigate(`/review/${id}`)}>
    👁 Review
  </MenuItem>
  <MenuItem onClick={() => downloadPDF(id)}>
    ⬇️ Download PDF
  </MenuItem>
  <MenuItem onClick={() => exportToCSV()}>
    📊 Export
  </MenuItem>
</ActionMenu>
```

---

### **PAGE 4: APPLICATION REVIEW (XAI PAGE)** (`pages/ApplicationReview.tsx`)

#### **4.1 Two-Column Layout**

```tsx
<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
  
  {/* Left: Applicant Details (1/3) */}
  <aside className="lg:col-span-1">
    <Card title="Applicant Information">
      <div className="space-y-4">
        <DetailRow label="Name" value={application.applicantName} />
        <DetailRow label="Age" value={application.age} />
        <DetailRow label="Monthly Income" value={application.income} format="currency" />
        <DetailRow label="EMI Obligations" value={application.emi} format="currency" />
        <DetailRow label="Total Assets" value={application.assets} format="currency" />
        <DetailRow label="Credit Score" value={application.creditScore} />
        <DetailRow label="Requested Amount" value={application.loanAmount} format="currency" />
      </div>
    </Card>
  </aside>

  {/* Right: Decision & Explanation (2/3) */}
  <section className="lg:col-span-2">
    {/* Decision Banner */}
    <DecisionBanner decision={application.decision} />
    
    {/* Model Outputs */}
    <ModelOutputs decision={application.decision} />
    
    {/* Feature Contributions (XAI) */}
    <FeatureContributionChart decision={application.decision} />
    
    {/* Decision Explanation */}
    <DecisionExplanation decision={application.decision} />
    
    {/* If Deferred: Override Controls */}
    {application.status === 'deferred' && (
      <HumanDecisionOverride 
        applicationId={application.id}
        onDecision={handleManualDecision}
      />
    )}
  </section>
</div>
```

#### **4.2 Decision Banner Component**

```tsx
interface DecisionBanner {
  status: 'approved' | 'rejected' | 'deferred'
  timestamp: ISO8601
  decidedBy: 'model' | 'human'
}

// Visuals:
// Approved: Green background, checkmark, "Loan Approved"
// Rejected: Red background, X mark, "Application Rejected"
// Deferred: Pink background, clock icon, "Under Review"
```

#### **4.3 Model Outputs Component**

```tsx
interface ModelOutputs {
  riskScore: number           // 0-1, displayed as percentage
  cbessScore: number          // 0-100 (explainability score)
  uncertainty: number         // 0-1, displayed as percentage
  confidence: number          // 1 - uncertainty
}

// Visual representation:
// - Risk Score: Horizontal progress bar (red gradient for high risk)
// - CBES Score: Gauge chart (green for high explainability)
// - Uncertainty: Confidence meter
// - Confidence Level: Text badge ("High", "Medium", "Low")
```

**Example Output:**

```
┌─────────────────────────────┐
│ Risk Score: 0.35 (Low Risk) │
│ ████░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────────────────┤
│ CBES Score: 78/100 (Good)   │
│ 🎯 [78° gauge]              │
├─────────────────────────────┤
│ Uncertainty: 15%            │
│ Confidence: HIGH ✓          │
└─────────────────────────────┘
```

#### **4.4 Feature Contribution Chart (SHAP / LIME visualization)**

```tsx
interface FeatureContribution {
  features: [
    { name: 'Income', impact: 0.25, direction: 'positive' },
    { name: 'EMI', impact: -0.18, direction: 'negative' },
    { name: 'Assets', impact: 0.15, direction: 'positive' },
    { name: 'Credit Score', impact: 0.12, direction: 'positive' },
    { name: 'Age', impact: 0.05, direction: 'neutral' }
  ]
}

// Visualized as horizontal bar chart:
// Income       ████████████████████████  +0.25 (Most influential)
// EMI          ████████████████░░░░░░░░  -0.18 (Negative impact)
// Assets       ███████████████░░░░░░░░░  +0.15
// Credit Score ██████████░░░░░░░░░░░░░░  +0.12
// Age          ███░░░░░░░░░░░░░░░░░░░░░  +0.05
```

#### **4.5 Decision Explanation Section**

```tsx
interface DecisionExplanation {
  text: string  // e.g., "Low income combined with high EMI obligations indicates elevated risk. Assets provide some mitigation. Model deferred for human review."
  factors: {
    positive: string[]   // e.g., ["Strong credit history", "Adequate assets"]
    negative: string[]   // e.g., ["High debt-to-income ratio", "Recent employment"]
    neutral: string[]
  }
}
```

#### **4.6 Human Decision Override (If Status = Deferred)**

```tsx
<Card title="Analyst Decision" className="border-2 border-warning">
  <form>
    <fieldset>
      <legend>Override Decision</legend>
      
      <RadioGroup>
        <Radio value="approve">
          <Label>✓ Approve</Label>
          <Caption>Loan approved based on manual review</Caption>
        </Radio>
        <Radio value="reject">
          <Label>✗ Reject</Label>
          <Caption>Application rejected based on manual review</Caption>
        </Radio>
      </RadioGroup>
    </fieldset>

    <TextArea 
      label="Review Notes" 
      placeholder="Document your decision reasoning..."
      required
    />

    <Button 
      variant="primary" 
      onClick={submitManualDecision}
    >
      Submit Decision
    </Button>
  </form>
</Card>
```

---

## 📊 DATA CONTRACTS & API INTEGRATION

### Backend API Routes (Required)

```
# Authentication
POST   /api/auth/login          → { email, password } → { token, role }
POST   /api/auth/logout         → {}
GET    /api/auth/me             → { user }

# Applications (Customer)
POST   /api/applications        → { formData } → { id, status }
GET    /api/applications/:id    → { application }
GET    /api/applications        → { applications[] }
PUT    /api/applications/:id    → { formData } → { application }

# File Upload
POST   /api/applications/:id/upload        → { file } → { extractedData, fileName }
GET    /api/applications/:id/documents     → { documents[] }

# Decision & ML
POST   /api/applications/:id/decide       → { } → { decision: {...} }
GET    /api/applications/:id/explanation  → { explanation: {...} }

# Organization Dashboard
GET    /api/org/metrics                    → { totalApplications, approved, rejected, deferred }
GET    /api/org/applications               → { applications[], filters }
GET    /api/org/charts/approval-dist       → { chartData }
GET    /api/org/charts/applications-trend  → { chartData }

# Manual Review
POST   /api/applications/:id/manual-decision  → { decision, notes } → { updatedApplication }

# Metrics (Public)
GET    /api/metrics/public                 → { applicationsProcessed, speedup, accuracy, automationRate }
```

### Response Data Schemas

```ts
// Standard Response Envelope
interface ApiResponse<T> {
  success: boolean
  data: T
  error?: {
    code: string
    message: string
    details?: Record<string, any>
  }
  timestamp: ISO8601
}

// Application Decision Object
interface ApplicationDecision {
  id: string
  status: 'approved' | 'rejected' | 'deferred'
  riskScore: number              // 0-1
  cbessScore: number             // 0-100
  uncertainty: number            // 0-1
  confidence: 'high' | 'medium' | 'low'
  explanation: string
  features: {
    name: string
    impact: number              // -1 to +1
    value: number               // Actual value from data
  }[]
  decidedBy: 'model' | 'human'
  timestamp: ISO8601
  analystId?: string            // If decided by human
  analystNotes?: string
}
```

---

## 🔌 BACKEND INTEGRATION POINTS

### 1. File Upload Pipeline

```
Flow:
Frontend: User selects PDF/CSV
  ↓
POST /api/applications/:id/upload { file }
  ↓
Backend: Parse file
  - PDF: Extract text → Parse fields
  - CSV: Read rows → Map to schema
  ↓
Return: { extractedData: {...}, fileName, uploadedAt }
  ↓
Frontend: Auto-fill form with extracted data
```

### 2. ML Decision Pipeline

```
Flow:
Frontend: User submits form
  ↓
POST /api/applications { formData }
  ↓
Backend: Store application
  ↓
Trigger ML Model:
  - Validate inputs
  - Scale/normalize features
  - Get model predictions
  - Calculate CBES score
  - Determine uncertainty
  ↓
Decision Logic:
  IF confidence > threshold:
    → Approve/Reject
  ELSE:
    → Defer for human review
  ↓
Return decision to frontend
  ↓
Frontend: Display decision + explanation
```

### 3. Human Decision Override

```
Flow:
Analyst views deferred application
  ↓
Analyst makes decision (Approve/Reject)
  ↓
POST /api/applications/:id/manual-decision { decision, notes }
  ↓
Backend: Update application status
  ↓
Update decision timestamp + analyst ID
  ↓
Return updated application
  ↓
Frontend: Refresh decision view
```

---

## 🎛️ STATE MANAGEMENT (Zustand)

```ts
// store/applicationStore.ts
export const useApplicationStore = create((set) => ({
  // State
  applications: [] as LoanApplication[],
  currentApplication: null as LoanApplication | null,
  isLoading: false,
  error: null as string | null,

  // Actions
  fetchApplications: async () => { /* ... */ },
  createApplication: async (data: ApplicationFormData) => { /* ... */ },
  submitApplication: async (id: string) => { /* ... */ },
  fetchDecision: async (id: string) => { /* ... */ },
  setCurrentApplication: (app: LoanApplication) => set({ currentApplication: app }),
}))

// store/authStore.ts
export const useAuthStore = create((set) => ({
  user: null as { id, email, role } | null,
  token: null as string | null,
  login: async (email, password) => { /* ... */ },
  logout: () => set({ user: null, token: null }),
}))

// store/uiStore.ts
export const useUIStore = create((set) => ({
  activeTab: 'all',
  filterStatus: null,
  setActiveTab: (tab: string) => set({ activeTab: tab }),
  setFilterStatus: (status: string) => set({ filterStatus: status }),
}))
```

---

## ⚡ PERFORMANCE & ACCESSIBILITY

### Performance Requirements

```
Metrics (Lighthouse):
- FCP (First Contentful Paint): < 1.5s
- LCP (Largest Contentful Paint): < 2.5s
- CLS (Cumulative Layout Shift): < 0.1
- TTI (Time to Interactive): < 3.5s

Optimization Strategies:
- Code splitting per route (React.lazy)
- Image optimization (next/image or similar)
- Debounce table filters/sorts
- Virtual scrolling for large tables (react-window)
- Memoize expensive computations (useMemo)
- API response caching (HTTP 304 Not Modified)
```

### Accessibility (WCAG 2.1 AA)

```
Requirements:
- Semantic HTML: <button>, <form>, <nav>, <main>
- ARIA labels for icon-only buttons
- Keyboard navigation: Tab through all interactive elements
- Color contrast: 4.5:1 for normal text, 3:1 for large text
- Focus indicators: Visible outline on all focusable elements
- Form validation: Clear error messages, associated with inputs
- Alt text: All images and charts
- Skip links: "Skip to main content"
```

**Example Accessible Button:**

```tsx
<button
  onClick={handleClick}
  aria-label="Submit loan application"
  className="focus:outline-2 focus:outline-offset-2 focus:outline-primary-500"
>
  Submit Application
</button>
```

---

## 🧪 TESTING STRATEGY (Brief)

```
Unit Tests: Component rendering, form validation, store mutations
Integration Tests: API calls, authentication flow, form submission
E2E Tests: User journeys (apply for loan, review decision, override)

Tools:
- Vitest (unit tests)
- React Testing Library (component tests)
- Playwright / Cypress (E2E)
```

---

## 📋 IMPLEMENTATION CHECKLIST

### Phase 1: Core UI Structure
- [ ] Design system (colors, typography, spacing)
- [ ] Reusable components (Button, Card, Input, Badge)
- [ ] Landing page structure
- [ ] Page routing (React Router v6)

### Phase 2: Customer Dashboard
- [ ] Loan application form with validation
- [ ] File upload integration
- [ ] Application history table
- [ ] Form auto-save

### Phase 3: Organization Dashboard
- [ ] KPI cards
- [ ] Charts (Recharts)
- [ ] Application table with filtering/sorting
- [ ] Deferred applications tab

### Phase 4: Application Review & XAI
- [ ] Decision banner
- [ ] Model outputs visualization
- [ ] Feature contribution chart
- [ ] Human decision override
- [ ] Explanation text

### Phase 5: Backend Integration
- [ ] API client setup (Axios)
- [ ] Authentication flow
- [ ] Application submission pipeline
- [ ] File upload handling
- [ ] Decision retrieval

### Phase 6: Polish & Deploy
- [ ] Performance optimization
- [ ] Accessibility audit
- [ ] Error handling & edge cases
- [ ] Mobile responsiveness
- [ ] Deployment (Vercel / Netlify)

---

## 🚀 DEVELOPMENT WORKFLOW

```bash
# Setup
npm create vite@latest smartlend -- --template react-ts
cd smartlend
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Install dependencies
npm install \
  react-router-dom \
  zustand \
  react-hook-form \
  zod \
  axios \
  framer-motion \
  recharts \
  lucide-react

# Start dev server
npm run dev

# Build
npm run build
```

---

## 🎯 SUCCESS CRITERIA

This UI is production-ready when:

1. ✅ All pages render without errors
2. ✅ Forms validate correctly
3. ✅ API integration works (mock or real backend)
4. ✅ Charts display real data
5. ✅ XAI visualization is clear and intuitive
6. ✅ Mobile responsive (< 768px)
7. ✅ Accessibility compliant (WCAG AA)
8. ✅ Lighthouse score > 90
9. ✅ Zero console errors/warnings
10. ✅ Decision flow is intuitive for both customer and analyst

---

**End of Technical Specification**

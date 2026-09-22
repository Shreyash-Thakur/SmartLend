# 🚀 SMARTLEND QUICK REFERENCE GUIDE

## 📡 API INTEGRATION CHECKLIST

### Required Endpoints (Minimum Viable)

```
✅ POST   /api/applications
✅ GET    /api/applications
✅ GET    /api/applications/:id
✅ POST   /api/applications/:id/upload
✅ POST   /api/applications/:id/decide          (Trigger ML)
✅ POST   /api/applications/:id/manual-decision  (Human override)
✅ GET    /api/org/metrics
✅ GET    /api/org/applications
✅ POST   /api/auth/login
✅ GET    /api/auth/me
```

---

## 🧪 COMMON TESTING PATTERNS

### Form Submission Flow

```tsx
// Test: User fills form → submits → gets application ID

it('should submit loan application and redirect', async () => {
  render(<LoanApplicationForm onSubmit={mockSubmit} />)
  
  // Fill form
  fireEvent.change(screen.getByLabelText(/Loan Amount/), { 
    target: { value: '500000' } 
  })
  fireEvent.change(screen.getByLabelText(/Monthly Income/), { 
    target: { value: '50000' } 
  })
  
  // Submit
  fireEvent.click(screen.getByText(/Submit Application/))
  
  // Wait for submission
  await waitFor(() => {
    expect(mockSubmit).toHaveBeenCalled()
  })
})
```

### Decision Display Flow

```tsx
it('should display decision with explanation', async () => {
  const mockDecision = {
    status: 'approved',
    riskScore: 0.25,
    cbessScore: 85,
    uncertainty: 0.15,
    explanation: 'Strong income with manageable EMI obligations'
  }
  
  render(<DecisionBanner decision={mockDecision} />)
  
  expect(screen.getByText(/Loan Approved/)).toBeInTheDocument()
  expect(screen.getByText(/Strong income/)).toBeInTheDocument()
})
```

---

## 🐛 DEBUGGING TIPS

### Issue: Form validation not working

```tsx
// ❌ Wrong: Forgot validation schema
useForm()

// ✅ Correct: Include resolver
const { register } = useForm({
  resolver: zodResolver(validationSchema),
  mode: 'onChange'  // Real-time validation
})
```

### Issue: API call fails silently

```tsx
// Add error logging
try {
  const response = await fetch('/api/applications')
  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`)
  }
} catch (error) {
  console.error('Application fetch failed:', error)
  // Display user-friendly message
}
```

### Issue: Chart not rendering data

```tsx
// ❌ Wrong: Raw data format
const chartData = applications  // This is an array of objects

// ✅ Correct: Transform to chart format
const chartData = {
  labels: ['Approved', 'Rejected', 'Deferred'],
  datasets: [{
    data: [approved, rejected, deferred]
  }]
}
```

---

## 📊 MOCK DATA FOR DEVELOPMENT

### Sample Application

```json
{
  "id": "app-uuid-123",
  "createdAt": "2025-04-20T10:30:00Z",
  "status": "deferred",
  "applicantName": "Rajesh Kumar",
  "loanAmount": 500000,
  "applicationData": {
    "monthlyIncome": 50000,
    "emi": 8000,
    "assets": 1000000,
    "creditScore": 650,
    "age": 35,
    "dependents": 2
  },
  "decision": {
    "status": "deferred",
    "riskScore": 0.45,
    "cbessScore": 72,
    "uncertainty": 0.28,
    "confidence": "medium",
    "explanation": "Income is moderate but EMI obligations are reasonable. Credit score is average. Recommend human review for final decision.",
    "positiveFactors": [
      "Adequate assets for collateral",
      "Stable employment history"
    ],
    "negativeFactors": [
      "Average credit score",
      "High debt-to-income ratio"
    ],
    "featureImportance": [
      { "name": "EMI", "impact": -0.22, "value": 8000 },
      { "name": "Income", "impact": 0.18, "value": 50000 },
      { "name": "Assets", "impact": 0.15, "value": 1000000 }
    ]
  }
}
```

### Sample Dashboard Metrics

```json
{
  "totalApplications": 1250,
  "approved": 850,
  "rejected": 200,
  "deferred": 200,
  "averageProcessingTime": 120,
  "approvalRate": 68,
  "automationRate": 82
}
```

---

## 🎯 PERFORMANCE OPTIMIZATION CHECKLIST

### Before Going to Production

- [ ] **Code Splitting**: Routes lazy-loaded
  ```tsx
  const Dashboard = React.lazy(() => import('./pages/Dashboard'))
  ```

- [ ] **Image Optimization**: Images are webp / optimized
  ```tsx
  <img src="image.webp" alt="description" width={800} height={600} />
  ```

- [ ] **API Caching**: GET requests cached (HTTP 304)
  ```ts
  const apiClient = axios.create({
    headers: {
      'Cache-Control': 'max-age=300'  // 5 min cache
    }
  })
  ```

- [ ] **Debounce Filters**: Table filters debounced
  ```tsx
  const debouncedSearch = useMemo(
    () => debounce((value) => setFilter(value), 300),
    []
  )
  ```

- [ ] **Memo Components**: Expensive renders memoized
  ```tsx
  const FeatureChart = memo(({ features }) => {...})
  ```

- [ ] **CSS**: No unused CSS (Tailwind purge enabled)

- [ ] **Bundle Size**: < 150KB (gzipped)
  ```bash
  npm run build -- --analyze
  ```

---

## 🔐 SECURITY CHECKLIST

- [ ] **JWT Storage**: Token in httpOnly cookie (not localStorage)
  ```ts
  // Backend should set: Set-Cookie: token=...; HttpOnly; Secure
  ```

- [ ] **CSRF Protection**: POST requests include CSRF token
  ```ts
  headers: { 'X-CSRF-Token': csrfToken }
  ```

- [ ] **Input Validation**: All inputs validated client + server
  ```ts
  // Client: Zod schema
  // Server: Same validation (never trust client)
  ```

- [ ] **XSS Prevention**: All user input sanitized
  ```tsx
  // ❌ Wrong: Directly render HTML
  <div dangerHTML={explanation} />
  
  // ✅ Correct: Text content
  <p>{explanation}</p>
  ```

- [ ] **API Rate Limiting**: Implement on backend
  ```
  POST /api/applications: max 5 requests per minute
  GET  /api/applications: max 100 requests per minute
  ```

---

## 📱 RESPONSIVE BREAKPOINTS

Tailwind breakpoints used throughout:

```css
sm  → 640px
md  → 768px
lg  → 1024px
xl  → 1280px
2xl → 1536px
```

Example usage:

```tsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  // 1 col on mobile, 2 on tablet, 3 on desktop
</div>
```

---

## 🧠 STATE MANAGEMENT PATTERNS

### Using Zustand Store

```ts
// store/applicationStore.ts
import { create } from 'zustand'

export const useApplicationStore = create((set, get) => ({
  // State
  applications: [],
  currentApplicationId: null,

  // Selectors
  getCurrentApplication: () => {
    const { applications, currentApplicationId } = get()
    return applications.find(app => app.id === currentApplicationId)
  },

  // Actions (mutations)
  setApplications: (apps) => set({ applications: apps }),
  
  addApplication: (app) => set((state) => ({
    applications: [...state.applications, app]
  })),
  
  // Async actions
  fetchApplications: async () => {
    const response = await fetch('/api/applications')
    const data = await response.json()
    set({ applications: data })
  }
}))

// Usage in component
function MyComponent() {
  const applications = useApplicationStore(state => state.applications)
  const fetchApplications = useApplicationStore(state => state.fetchApplications)
  
  useEffect(() => {
    fetchApplications()
  }, [])
}
```

---

## 🎨 COMMON STYLING PATTERNS

### Gradient Background

```tsx
<div className="bg-gradient-to-r from-primary-500 via-accent-500 to-primary-600">
  {/* Content */}
</div>
```

### Glassmorphism Card

```tsx
<div className="
  rounded-xl
  bg-white/80
  backdrop-blur-md
  border border-white/20
  shadow-lg
">
  {/* Card content */}
</div>
```

### Responsive Grid

```tsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
  {/* Cards */}
</div>
```

### Hover Effects

```tsx
<div className="
  transition-all duration-300
  hover:shadow-lg hover:scale-105
  hover:bg-primary-50
">
  {/* Content */}
</div>
```

---

## 🚨 COMMON MISTAKES & FIXES

| Mistake | Problem | Fix |
|---------|---------|-----|
| `const data = await fetch()` | Returns Response, not JSON | `const data = await res.json()` |
| Forgetting `.preventDefault()` | Form submits and refreshes | `<form onSubmit={(e) => { e.preventDefault(); }}` |
| Setting state directly | React doesn't detect change | `setState(prev => ({ ...prev, key: value }))` |
| Infinite API calls | Network waterfall | Add dependency array: `useEffect(() => {...}, [])` |
| Table re-rendering on every keystroke | Performance issue | Use `useMemo` for filtered data |
| Image elements without alt | Accessibility issue | Always add `alt="descriptive text"` |
| Hardcoded API URL | Breaks in different environments | Use env variables: `process.env.REACT_APP_API_URL` |

---

## 📝 LOGGING & MONITORING

### Development Logging

```ts
// utils/logger.ts
export const logger = {
  info: (message: string, data?: any) => {
    if (process.env.NODE_ENV === 'development') {
      console.log(`[INFO] ${message}`, data)
    }
  },
  
  error: (message: string, error?: Error) => {
    console.error(`[ERROR] ${message}`, error)
    // Send to monitoring service in production
  },
  
  warn: (message: string) => {
    console.warn(`[WARN] ${message}`)
  }
}
```

### Usage

```ts
logger.info('Fetching applications', { page: 1 })
logger.error('API call failed', error)
logger.warn('Feature X not available')
```

---

## 🤝 GIT WORKFLOW

```bash
# Feature branch
git checkout -b feature/add-xai-visualization

# Commit messages
git commit -m "feat: add SHAP feature contribution chart"
git commit -m "fix: resolve application table sorting bug"
git commit -m "style: update button hover transitions"
git commit -m "docs: add API endpoint documentation"

# Push and create PR
git push origin feature/add-xai-visualization
```

---

## 📦 DEPENDENCY MANAGEMENT

### Install/Update Commands

```bash
# Add dependency
npm install react-hook-form zod

# Update all
npm update

# Remove unused
npm prune

# Audit security
npm audit
npm audit fix
```

### Keep Dependencies Minimal

- Don't install if native React can do it
- Prefer smaller libraries over large frameworks
- Example: Use `zustand` instead of Redux for simple state

---

## 🔄 DEPLOYMENT CHECKLIST

Before deploying to production:

- [ ] All tests passing: `npm test`
- [ ] No console errors: Check DevTools
- [ ] Build succeeds: `npm run build`
- [ ] Bundle size < 200KB gzipped
- [ ] All env variables set
- [ ] API endpoints updated for production
- [ ] Security headers configured
- [ ] Error tracking enabled (Sentry, etc.)
- [ ] Performance monitoring enabled
- [ ] Database backups scheduled
- [ ] SSL/TLS certificate valid
- [ ] CORS properly configured

---

## 🆘 GETTING HELP

### Documentation
- Tailwind CSS: https://tailwindcss.com/docs
- React Hook Form: https://react-hook-form.com/
- Framer Motion: https://www.framer.com/motion/
- Recharts: https://recharts.org/

### Debugging Tools
- React DevTools (Chrome extension)
- Network tab (see actual API calls)
- Console for errors and logs
- Lighthouse for performance

---

**Last Updated:** April 2026  
**Version:** 1.0

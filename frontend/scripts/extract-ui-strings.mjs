// Collect the English UI text of the customer-side pages into src/i18n/en.json.
//
// Run: node scripts/extract-ui-strings.mjs
// Then: python scripts/translate_ui_strings.py   (from the repo root; Sarvam)
//
// Uses the TypeScript parser, not regexes: it keeps JSX text, string-valued
// JSX attributes a user can read (label, placeholder, hint, ...), and string
// literals that look like sentences or labels. Class names, paths and enum
// values are filtered out. Text built with template literals (numbers in it)
// is not here; the page translates those live through /api/translate.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'src')

// Everything a customer can see. Org-only pages are deliberately absent.
const FILES = [
  'pages/CustomerNewApplication.tsx',
  'pages/Dashboard.customer.tsx',
  'pages/ApplicationReview.tsx',
  'components/sections/ApplicationTable.tsx',
  'components/sections/DecisionExplanation.tsx',
  'components/sections/FeatureContributionChart.tsx',
  'components/sections/StatusTimeline.tsx',
  'components/layouts/AppShell.tsx',
  'components/layouts/navigation.ts',
  'components/common/Select.tsx',
  'components/common/Input.tsx',
  'components/common/Textarea.tsx',
  'components/common/Card.tsx',
]

const READABLE_ATTRS = new Set(['label', 'placeholder', 'hint', 'title', 'description', 'aria-label', 'alt', 'eyebrow'])
const READABLE_KEYS = new Set(['label', 'title', 'description', 'hint', 'placeholder', 'text', 'message', 'detail', 'eyebrow'])

const looksLikeText = (s) =>
  /[A-Za-z]{2}/.test(s) &&
  s.length <= 300 &&
  !/^[a-z0-9_.:/#@-]+$/.test(s) && // identifiers, paths, keys
  !/^[A-Z0-9_]+$/.test(s) && // ENUM_VALUES
  !/\b(?:px|py|mt|mb|text|bg|border|rounded|flex|grid|gap|w|h)-[a-z0-9]/.test(s) && // tailwind
  !/^(?:https?:|\/|@\/|\.\/)/.test(s)

const clean = (s) => s.replace(/\s+/g, ' ').trim()

// Names that must stay as they are in every language.
const KEEP_AS_IS = new Set(['SmartLend', 'CIBIL', 'KYC', 'EMI', 'PAN'])

const strings = new Set()
for (const rel of FILES) {
  const file = path.join(ROOT, rel)
  const source = ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
  const visit = (node) => {
    if (ts.isJsxText(node)) {
      const text = clean(node.getText())
      if (looksLikeText(text)) strings.add(text)
    } else if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      const text = clean(node.text)
      const parent = node.parent
      const inReadableAttr = ts.isJsxAttribute(parent) && READABLE_ATTRS.has(parent.name.getText())
      const inReadableKey =
        ts.isPropertyAssignment(parent) && READABLE_KEYS.has(parent.name.getText().replace(/['"]/g, ''))
      const capitalised = /^[A-Z₹]/.test(text) && /[a-z]/.test(text)
      const skip = ts.isImportDeclaration(parent) || ts.isExportDeclaration(parent) ||
        (ts.isJsxAttribute(parent) && ['className', 'id', 'to', 'href', 'type', 'name', 'key', 'variant', 'size'].includes(parent.name.getText()))
      if (!skip && looksLikeText(text) && (inReadableAttr || inReadableKey || capitalised)) strings.add(text)
    }
    ts.forEachChild(node, visit)
  }
  visit(source)
}

for (const name of KEEP_AS_IS) strings.delete(name)

const out = path.join(ROOT, 'i18n', 'en.json')
fs.writeFileSync(out, JSON.stringify([...strings].sort(), null, 1) + '\n')
console.log(`${strings.size} strings -> ${path.relative(process.cwd(), out)}`)

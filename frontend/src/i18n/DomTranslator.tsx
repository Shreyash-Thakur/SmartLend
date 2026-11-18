import { useEffect } from 'react'
import { apiClient } from '@/services/api.client'
import { useLanguageStore, type Language } from '@/store/languageStore'
import hi from './hi.json'
import mr from './mr.json'

/**
 * Translates the customer pages in place, without touching their components.
 *
 * Static labels come from hi.json / mr.json, generated once with Sarvam by
 * scripts/translate_ui_strings.py, so they never need the network. Text inside
 * an element marked `data-i18n-live` (the decision explanation, factor names,
 * the status timeline: sentences with numbers in them) is translated through
 * /api/translate and cached; if that fails the English stays.
 *
 * It only rewrites Text.nodeValue and a few attributes, never replaces nodes,
 * so React keeps owning the DOM. The English original of every node is kept,
 * which is how switching back to English restores the page.
 */
const DICTIONARIES: Record<Exclude<Language, 'en'>, Record<string, string>> = { hi, mr }
const ATTRIBUTES = ['placeholder', 'title', 'aria-label'] as const
const LIVE_BATCH = 40

const originalText = new WeakMap<Text, string>()
const appliedText = new WeakMap<Text, string>()
const originalAttrs = new WeakMap<Element, Record<string, string>>()
const liveCache: Record<string, Map<string, string>> = { hi: new Map(), mr: new Map() }
const liveFailed = new Set<string>()

const worthTranslating = (text: string) =>
  /[A-Za-z]{2}/.test(text) && !/^\S*\d\S*$/.test(text) && !text.includes('@')

function lookup(language: Language, english: string, live: boolean, queue: Set<string>): string | undefined {
  if (language === 'en') return undefined
  const fixed = DICTIONARIES[language][english]
  if (fixed) return fixed
  const cached = liveCache[language].get(english)
  if (cached) return cached
  if (live && worthTranslating(english) && !liveFailed.has(`${language}:${english}`)) queue.add(english)
  return undefined
}

function translateTextNode(node: Text, language: Language, queue: Set<string>) {
  const current = node.nodeValue ?? ''
  // React may have changed the text since we last touched it: anything that is
  // neither our last output nor the stored original is fresh English.
  if (!originalText.has(node) || (current !== appliedText.get(node) && current !== originalText.get(node))) {
    originalText.set(node, current)
  }
  const english = originalText.get(node) ?? current
  const key = english.trim()
  if (!key) return
  const live = Boolean(node.parentElement?.closest('[data-i18n-live]'))
  const translated = lookup(language, key, live, queue)
  const next = translated ? english.replace(key, translated) : english
  appliedText.set(node, next)
  if (node.nodeValue !== next) node.nodeValue = next
}

function translateAttributes(element: Element, language: Language, queue: Set<string>) {
  for (const name of ATTRIBUTES) {
    const value = element.getAttribute(name)
    if (value === null) continue
    const originals = originalAttrs.get(element) ?? {}
    if (!(name in originals) || (value !== originals[name] && !Object.values(DICTIONARIES).some((d) => d[originals[name]] === value))) {
      originals[name] = value
      originalAttrs.set(element, originals)
    }
    const english = originals[name]
    const next = lookup(language, english.trim(), false, queue) ?? english
    if (value !== next) element.setAttribute(name, next)
  }
}

function translateTree(root: Node, language: Language, queue: Set<string>) {
  if (root.nodeType === Node.TEXT_NODE) {
    translateTextNode(root as Text, language, queue)
    return
  }
  if (!(root instanceof Element)) return
  if (root.closest('[data-i18n-skip]')) return
  translateAttributes(root, language, queue)
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT)
  let node = walker.nextNode()
  while (node) {
    if (node.nodeType === Node.TEXT_NODE) {
      const parent = node.parentElement
      if (parent && !['SCRIPT', 'STYLE', 'TEXTAREA'].includes(parent.tagName) && !parent.closest('[data-i18n-skip]')) {
        translateTextNode(node as Text, language, queue)
      }
    } else {
      translateAttributes(node as Element, language, queue)
    }
    node = walker.nextNode()
  }
}

async function fetchLive(language: Exclude<Language, 'en'>, texts: string[]) {
  for (let start = 0; start < texts.length; start += LIVE_BATCH) {
    const batch = texts.slice(start, start + LIVE_BATCH)
    try {
      const response = await apiClient.post<{ translations: Array<string | null> }>('/translate', { texts: batch, target: language })
      batch.forEach((text, index) => {
        const translated = response.data.translations[index]
        if (translated) liveCache[language].set(text, translated)
        else liveFailed.add(`${language}:${text}`)
      })
    } catch {
      // No key or Sarvam unreachable: keep English, and do not retry these.
      batch.forEach((text) => liveFailed.add(`${language}:${text}`))
    }
  }
}

/** Mount once inside the customer shell; renders nothing. */
export function DomTranslator({ rootSelector = '[data-i18n-root]' }: { rootSelector?: string }) {
  const language = useLanguageStore((state) => state.language)

  useEffect(() => {
    const root = document.querySelector(rootSelector)
    if (!root) return
    document.documentElement.lang = language
    let pending: number | undefined
    // Accumulates across runs: React renders in bursts, and each burst must
    // add to the next request instead of replacing what is already waiting.
    const waiting = new Set<string>()

    const run = (target: Node) => {
      translateTree(target, language, waiting)
      if (waiting.size > 0 && language !== 'en') {
        window.clearTimeout(pending)
        pending = window.setTimeout(() => {
          const texts = [...waiting]
          waiting.clear()
          void fetchLive(language, texts).then(() => translateTree(root, language, new Set<string>()))
        }, 250)
      }
    }

    run(root)
    const observer = new MutationObserver((records) => {
      for (const record of records) {
        if (record.type === 'characterData') run(record.target)
        else if (record.type === 'attributes') run(record.target)
        else record.addedNodes.forEach((node) => run(node))
      }
    })
    observer.observe(root, {
      subtree: true,
      childList: true,
      characterData: true,
      attributes: true,
      attributeFilter: [...ATTRIBUTES],
    })
    return () => {
      observer.disconnect()
      window.clearTimeout(pending)
    }
  }, [language, rootSelector])

  return null
}

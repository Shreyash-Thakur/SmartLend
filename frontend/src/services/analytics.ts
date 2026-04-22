export function trackEvent(eventName: string, payload?: Record<string, unknown>) {
  if (import.meta.env.DEV) {
    console.info('[analytics]', eventName, payload ?? {})
  }
}

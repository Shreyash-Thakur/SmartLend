import ShapeGrid from '@/components/background/ShapeGrid'
import { useThemeStore } from '@/store/themeStore'

// Page colours — keep in sync with the :root / :root.dark background in globals.css.
const PAGE_BG = { light: '#f7f8fa', dark: '#0a0a0a' }

const GRID = {
  light: { border: 'rgba(15, 23, 42, 0.07)', hover: 'rgba(59, 130, 246, 0.14)' },
  dark: { border: 'rgba(255, 255, 255, 0.06)', hover: 'rgba(255, 255, 255, 0.1)' },
}

/** Fixed, full-screen animated grid behind every page. Purely decorative. */
export function AppBackground() {
  const theme = useThemeStore((state) => state.theme)

  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 -z-10">
      <ShapeGrid
        direction="diagonal"
        speed={0.5}
        squareSize={40}
        shape="square"
        hoverTrailAmount={5}
        borderColor={GRID[theme].border}
        hoverFillColor={GRID[theme].hover}
        vignetteColor={PAGE_BG[theme]}
      />
    </div>
  )
}

import { createContext, useContext, type ReactNode } from 'react'

export type GatewayTab = 'overview' | 'users' | 'keys' | 'budgets' | 'usage' | 'providers' | 'capture'
export type GatewayCaptureView =
  | { name: 'status' }
  | { name: 'records' }
  | { name: 'detail'; requestId: string }

export type GatewayNav = {
  openTab: (tab: GatewayTab) => void
  openCapture: (view: GatewayCaptureView) => void
}

const GatewayNavContext = createContext<GatewayNav | null>(null)

export function GatewayNavProvider({ value, children }: { value: GatewayNav; children: ReactNode }) {
  return <GatewayNavContext.Provider value={value}>{children}</GatewayNavContext.Provider>
}

export function useGatewayNav(): GatewayNav {
  return useContext(GatewayNavContext) ?? { openTab: () => undefined, openCapture: () => undefined }
}

export function GatewayNavButton({
  tab,
  capture,
  children,
}: {
  tab?: GatewayTab
  capture?: GatewayCaptureView
  children: ReactNode
}) {
  const { openTab, openCapture } = useGatewayNav()
  return (
    <button
      type="button"
      className="link-button"
      onClick={() => (capture ? openCapture(capture) : tab ? openTab(tab) : undefined)}
    >
      {children}
    </button>
  )
}

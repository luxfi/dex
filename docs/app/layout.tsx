import "./global.css"
import { RootProvider } from "@hanzo/docs-ui/provider/next"
import { Zen } from "@hanzo/font/sans"
import { ZenMono } from "@hanzo/font/mono"
import type { ReactNode } from "react"

export const metadata = {
  title: {
    default: "LX - Ultra-High Performance Decentralized Exchange Documentation",
    template: "%s | LX - Ultra-High Performance Decentralized Exchange",
  },
  description: "434M+ orders/sec, sub-microsecond latency, quantum-resistant",
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${Zen.variable} ${ZenMono.variable}`}
      suppressHydrationWarning
    >
      <body className="min-h-svh bg-background font-sans antialiased">
        <RootProvider
          search={{
            enabled: true,
          }}
          theme={{
            enabled: true,
            defaultTheme: "dark",
          }}
        >
          <div className="relative flex min-h-svh flex-col bg-background">
            {children}
          </div>
        </RootProvider>
      </body>
    </html>
  )
}

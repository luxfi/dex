import { createMDX } from "@hanzo/docs/mdx/next"

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  reactStrictMode: true,
  typescript: {
    ignoreBuildErrors: true,
  },
  experimental: {
    webpackBuildWorker: true,
  },
  images: {
    unoptimized: true,
  },
  // For GitHub Pages deployment at dex.lux.network
  // No basePath needed since using custom domain
  basePath: '',
  trailingSlash: true,
}

const withMDX = createMDX()

export default withMDX(config)

import type { MDXComponents } from "mdx/types"
import { defaultMdxComponents } from "@hanzo/ui"
import {
  Tab,
  Tabs,
  Card,
  Cards,
  Step,
  Steps,
  Callout,
  Accordion,
  Accordions,
} from "@hanzo/ui/content"
import { TypeTable } from "@hanzo/ui/docs/components"

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    ...defaultMdxComponents,
    Tab,
    Tabs,
    Card,
    Cards,
    Step,
    Steps,
    Callout,
    Accordion,
    Accordions,
    TypeTable,
    ...components,
  }
}

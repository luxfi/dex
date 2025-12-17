import type { MDXComponents } from "mdx/types"
import defaultMdxComponents from "@hanzo/docs/ui/mdx"
import { Tab, Tabs } from "@hanzo/docs/ui/components/tabs"
import { Card, Cards } from "@hanzo/docs/ui/components/card"
import { Step, Steps } from "@hanzo/docs/ui/components/steps"
import { Callout } from "@hanzo/docs/ui/components/callout"
import { Accordion, Accordions } from "@hanzo/docs/ui/components/accordion"
import { TypeTable } from "@hanzo/docs/ui/components/type-table"

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

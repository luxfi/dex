import type { MDXComponents } from "mdx/types"
import defaultMdxComponents from "@hanzo/ui/docs/mdx"
import { Tab, Tabs } from "@hanzo/ui/docs/components/tabs"
import { Card, Cards } from "@hanzo/ui/docs/components/card"
import { Step, Steps } from "@hanzo/ui/docs/components/steps"
import { Callout } from "@hanzo/ui/docs/components/callout"
import { Accordion, Accordions } from "@hanzo/ui/docs/components/accordion"
import { TypeTable } from "@hanzo/ui/docs/components/type-table"

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

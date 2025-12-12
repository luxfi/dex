import type { MDXComponents } from "mdx/types"
import defaultMdxComponents from "fumadocs-ui/mdx"
import { Tab, Tabs } from "fumadocs-ui/components/tabs"
import { Card, Cards } from "fumadocs-ui/components/card"
import { Step, Steps } from "fumadocs-ui/components/steps"
import { Callout } from "fumadocs-ui/components/callout"
import { Accordion, Accordions } from "fumadocs-ui/components/accordion"
import { TypeTable } from "fumadocs-ui/components/type-table"

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

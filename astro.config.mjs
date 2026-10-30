import { defineConfig } from 'astro/config';
import { satteri } from '@astrojs/markdown-satteri';
import { defineMdastPlugin } from 'satteri';
import katex from 'katex';

// Renders $$ … $$ in posts with KaTeX at build time; the styles are imported in src/pages/writing/[slug].astro.
const katexMath = defineMdastPlugin({
  name: 'katex-math',
  math(node, ctx) {
    ctx.replaceNode(node, { type: 'html', value: katex.renderToString(node.value, { displayMode: true, throwOnError: true }) });
  },
  inlineMath(node, ctx) {
    ctx.replaceNode(node, { type: 'html', value: katex.renderToString(node.value, { throwOnError: true }) });
  },
});

export default defineConfig({
  site: 'https://iondy.com',
  output: 'static',
  trailingSlash: 'always',
  devToolbar: { enabled: false },
  markdown: {
    // Math only between $$ … $$, so a lone $ in prose (prices, shell variables) stays text.
    processor: satteri({ features: { math: { singleDollarTextMath: false } }, mdastPlugins: [katexMath] }),
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
      defaultColor: false,
    },
  },
});

# AGENTS.md

Guidance for coding agents working on iondy.com. The README describes the project for people; this file covers what to keep in mind when changing it.

## Project in brief

- Astro static site. One design, **Quiet**: `src/styles/site.css` holds the shared base rules, `src/styles/quiet.css` the design, scoped to `html[data-look="quiet"]` (set in `src/layouts/Base.astro`).
- Content: posts are `content/<slug>/index.md` with images beside them; papers are `papers` in `src/lib/content.ts`. The projects page is offline (`src/pages/_projects.astro`; Astro skips files starting with `_`).
- Commands: `npm run dev` (http://127.0.0.1:4321), `npm run build`, `npm test` (importer tests).
- Deploy: every push to `main` runs `.github/workflows/deploy.yml` and publishes to GitHub Pages. While the `COMING_SOON` repository variable is `true`, the live site shows `coming-soon/index.html` instead; local dev always shows the real site.
- Quiet is deliberately minimal: no filler slogans, decorative counters, badges, or heavy cards. AI slop (below) is the only exception.
- Never change the wording of articles, their figures, or their code.

## The AI slop slider: consider it for every visible change

The header has an **AI slop** slider with six cumulative levels. Level 0 is plain Quiet; each higher level adds more generated-website clichés. Any new page, section, component, or content type must look intentional at **all six levels**, not just in Quiet.

| Level | Name | Adds |
| --- | --- | --- |
| 0 | Off | Plain Quiet. |
| 1 | Mild | Purple accent, faint colour blobs, gradient wordmark and hairlines, rounded and shadowed code and images, hover pills on list rows. |
| 2 | Medium | The landing-page template: Inter, 1120px layout, dot grid, floating glass navbar, centred hero, glass cards, pill buttons, avatar, contents card, the GitHub header button. |
| 3 | Spicy | Gradient buttons and pills, neon card outlines, accent side stripes, a Featured first card, count badges, TL;DR box, outlined numbers, dark window-framed code. |
| 4 | Hot | Emoji everywhere: section and page badges, categories, dates, headings, ✅ bullets. |
| 5 | Maximum | Parody copy (taglines, "Trusted by localhost", buzzword ribbon), the AI orb, "reimagined.", shimmering text, stronger glows, bigger hover lifts. |

### How it works

- `src/components/SlopSlider.astro` is the slider; level names are `slopLevels` in `src/lib/slop.ts`.
- The head script in `Base.astro` reads `?slop=N` or the saved level (`localStorage` key `iondy-slop`) before first paint and sets `html[data-slop]` to every level up to the chosen one (`"1 2 3"` at level 3) and `--slop-level`.
- `src/styles/slop.css` has one nested block per level, `:root[data-look='quiet'][data-slop~='N'] { & … }`, in the `overlays` cascade layer. That layer comes after `responsive`, so slop rules win over Quiet at every screen width.

### When you add or change something visible

1. Style it in Quiet first (`quiet.css`: desktop rules in `@layer themes`, breakpoints at 1050, 760, and 550px in `@layer responsive`).
2. Decide what it becomes at each level and add rules to the matching block in `slop.css`. Usually: 1 colour and hover; 2 a glass card or pill in the wide layout; 3 gradient or badge treatment; 4 an emoji; 5 only if there is a good joke.
3. Reuse the existing selector lists where the new thing behaves like an existing one, for example `:is(.post-row, .project-card, .paper-row)`.
4. Check it at `?slop=0` through `?slop=5`, at desktop and at 390px and 760px widths, with no horizontal overflow (header included).

Per-item rules to update when content changes:

- **New post category**: add it to the `category` enum in `src/content.config.ts`, and its emoji to the `.post-row[data-category='…']` rules in level 4.
- **New page**: add its badge in level 4 (`main:has(.<page-class>) .page-intro::before`) and give it a distinctive container class to match on.
- **New homepage section**: add a `.section-heading::before` badge in level 4.
- **About links**: the level 4 emoji are matched by position (`.about-aside > a:nth-of-type(n)`); keep them in order when adding or removing links.
- **Restored projects**: the level 4 tile emoji are matched by `.project-card:nth-child(n)`.
- **Posts**: the level 3 TL;DR box shows the post's front-matter `description`, so keep descriptions to one clear sentence.

### Pitfalls (each one has bitten this codebase)

- **Breakpoints.** Slop rules beat Quiet's responsive rules at every width. A level that changes a size or layout must handle its own breakpoints, and a later level that changes something an earlier level adjusted in a media query has to restate it in its own media queries.
- **`!important` across layers.** Earlier layers win, so the base `.prose pre { background: var(--surface) !important }` beats any later `!important`. Change code-block backgrounds by setting `--surface` on the `pre`.
- **`:is()` specificity.** `:is()` takes its most specific argument; an `a` inside one argument made a level-1 colour rule beat a level-2 one.
- **CSS escapes.** `\00a0` followed by a hex-like letter (`\00a0Blazing`) becomes a different character; end the escape with a space (`\00a0 Blazing`).
- **Parody copy.** Add it with `content: '…' / ''` so screen readers skip it. Never add fake buttons or links that do nothing.
- **Motion.** `site.css` disables animations and transitions under `prefers-reduced-motion`; keep motion in CSS animations so that rule covers it.
- **Native nesting.** `slop.css` relies on CSS nesting, which the build leaves as is (current browsers only); start nested selectors with `&`.

Keep the README's "AI slop slider" section in sync when levels or behaviour change.

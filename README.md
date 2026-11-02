# iondy

A personal website for projects, research, and learning. Built with Astro and plain Markdown; the production output is a static website.

## Run locally

Requires Node.js 22.23.4 or newer.

```sh
npm ci
npm run dev
```

Open `http://127.0.0.1:4321/` for the homepage.

```sh
npm run build
npm run preview
```

The build creates `dist/`, suitable for static hosting, and `npm run preview` serves it at the same address. Fonts (Geist, Geist Mono, and, for the AI slop slider, Inter, JetBrains Mono, and Source Serif 4 italic), article images, and syntax highlighting are bundled locally; reading does not require third-party scripts or font services.

## Design

The site uses one design, Quiet: Geist in a narrow column, small type, whitespace instead of rules, navigation beside the name, and title-and-date lists. `src/styles/site.css` holds the shared base rules and `src/styles/quiet.css` the Quiet design, scoped to `html[data-look="quiet"]`.

The five alternative designs explored before choosing Quiet (Editorial, Index, Paper, Margin, Folio), their comparison gallery, and the design dock were removed. They are in the first commit, `7db2c49`, if you want to look at them again.

## AI slop slider

The header has an **AI slop** slider that visitors can drag from plain Quiet to a full parody of a generated website. Its ends read *sleek* and *slop*. Each level adds to the one before:

| Level | Name | Preview | What it adds |
| --- | --- | --- | --- |
| 0 | Off | [`/?slop=0`](http://127.0.0.1:4321/?slop=0) | Plain Quiet. |
| 1 | Mild | [`/?slop=1`](http://127.0.0.1:4321/?slop=1) | Purple accent, faint colour blobs, a gradient wordmark, gradient hairlines, rounded and shadowed code blocks and images, and the About page's list of profile links, which Quiet leaves to the footer. |
| 2 | Medium | [`/?slop=2`](http://127.0.0.1:4321/?slop=2) | Startup type, still in Quiet's narrow column: Inter, bigger gradient titles, purple uppercase labels, pill-shaped navigation, buttons, filters and tags, an author avatar on articles and the About page, taglines under the Writing, Research and About titles, and a reading progress bar on articles. |
| 3 | Spicy | [`/?slop=3`](http://127.0.0.1:4321/?slop=3) | The landing-page template: a wide page, a fading dot grid, a floating glass navbar with a GitHub button, a centred hero and section headings, glass cards, a contents card on articles, and the About page's "Take a look around" button. |
| 4 | Hot | [`/?slop=4`](http://127.0.0.1:4321/?slop=4) | Gradients, badges and emoji: gradient buttons, neon card outlines, accent side stripes, a Featured card, count badges, a TL;DR box, oversized outlined numbers on the research cards, dark window-framed code blocks, and emoji on headings, badges (the 404 page too), categories, filter pills, code-block languages, dates, and ✅ bullets. |
| 5 | Maximum | [`/?slop=5`](http://127.0.0.1:4321/?slop=5) | Generated copy, a glossy AI orb, "reimagined." in italic serif, "Trusted by teams at localhost", a buzzword ribbon, an `iondy.ai` wordmark on wide screens, an "AI Summary" with 99.8% confidence, an endorsement from Reviewer #2, shimmering gradient text, and stronger glows. |

The level is saved in the visitor's browser and applies to every page; `?slop=<0–5>` links to a level directly.

- `src/components/SlopSlider.astro` is the slider. Level names come from `slopLevels` in `src/lib/slop.ts`; screen readers announce them as the slider's value, while the slider itself shows only *sleek* and *slop*.
- `src/styles/slop.css` holds all five levels. The head script in `Base.astro` writes every level up to the chosen one to `html[data-slop]` (`"1 2 3"` at level 3), so `[data-slop~='3']` means "level 3 or higher". The file sits in its own `overlays` cascade layer after `responsive`, so it overrides Quiet at every screen width and handles its own breakpoints.
- Article text, figures, and code are unchanged at every level. The TL;DR box shows each post's `description`, and the parody copy and emoji are added with CSS and hidden from screen readers.
- Several of the level 1–5 tropes came from asking Codex for recognisable AI-generated-website patterns.

## Writing and content

Each post has its own folder directly inside `content/`:

```text
content/
  my-new-post/
    index.md
    images/
      example.png
```

`index.md` is the editable source of truth. Markdown keeps text, headings, links, lists, and code easy to edit; HTML is generated only when the website builds. There is no separate Medium content collection or duplicated article catalog to maintain.

Start a post with this front matter:

```yaml
---
title: "My new post"
date: "2026-10-29"
description: "A short summary for the archive and feed."
category: "Research"
tags: ["pytorch", "language-models"]
draft: false
---

The article starts here.

![A useful image](images/example.png)
```

A post whose figures come from data can keep the scripts and data in a `source/` folder next to `images/`, with a short README on how to rebuild them; the site builds only `index.md` and the images it shows. The sliding-window post does this.

The folder name becomes the URL: `/writing/my-new-post/`. Available categories are `Research`, `Engineering`, and `Learning`. Set `draft: true` to exclude a post from the site, archive, and feed. `original_url` is optional and records where an imported post was first published; the site doesn't display it. Dates and summaries drive the homepage, archive, and RSS automatically. Article content is the same at every AI slop level.

### Diagrams

A diagram drawn for the site keeps its SVG source next to the PNG the post shows, as in `content/introducing-pytora-a-minimal-pytorch-implementation-of-lora/images/lora.svg`. The SVG uses the site's fonts from `node_modules`, so render it with Chrome from the repository root, at twice its size:

```sh
dir=content/introducing-pytora-a-minimal-pytorch-implementation-of-lora/images
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --allow-file-access-from-files --hide-scrollbars \
  --force-device-scale-factor=2 --window-size=720,380 --virtual-time-budget=3000 \
  --screenshot=$dir/lora.png "file://$PWD/$dir/lora.svg"
```

`--window-size` is the SVG's `width` and `height`.

### Link previews

Every page shares one link-preview image, `public/og-image.png`, drawn in `src/og-image.svg`; `Base.astro` adds it with the page's canonical URL, and articles are marked as articles with their date. Render it at its real size:

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --allow-file-access-from-files --hide-scrollbars \
  --force-device-scale-factor=1 --window-size=1200,630 --virtual-time-budget=3000 \
  --screenshot=public/og-image.png "file://$PWD/src/og-image.svg"
```

Code blocks take a language after the opening fence (` ```python `) for syntax highlighting. Formulas go between `$$` lines and are typeset with KaTeX when the site builds; a single `$` stays text:

```md
$$
\theta_\tau = \theta_{pre} + \delta_\tau
$$
```

For a side note set apart from the main text, wrap it in an info box. Keep a blank line after the opening tag and before the closing one, so the Markdown inside still renders:

```md
<aside class="note">

**A short question as the title**

The note's text, with [links](https://example.com) and *emphasis*.

</aside>
```

### Papers

Publications live in `papers` in `src/lib/content.ts`, in the order of the CV. `/research/` lists all of them with authors, venue, a one-line summary, and links; the homepage shows the first four. The first link of each paper is where its title points, so put arXiv first when there is an arXiv version. `authors` is optional, for papers whose author list isn't public yet.

The seven imported posts retain their original article wording and 50 code blocks. Short descriptions and broad categories were added for the site's navigation. The titles are preserved as originally written, apart from two typos. Since the import, these Medium leftovers were fixed, without changing the wording except where noted:

- Code blocks name their language, so all of them are highlighted.
- The diff-pruning post's formulas, which were small images, are typeset with KaTeX.
- The packed-sequences post's tensor printouts, which were screenshots, are text. The batched ones came from a run with other sentences; they now show the output of the article's own code, taken from its notebook.
- The Mistral post's headings are main headings, so its contents list isn't empty.
- The pytora post no longer repeats its title as a heading, its title uses normal spaces around the dash, and its LoRA diagram has been redrawn in the site's style.
- Two titles had typos: "Adverserial Training" and "An practical introduction to Diff-Pruning for BERT" are now "Adversarial Training" and "A practical introduction to Diff-Pruning for BERT", and their folders and URLs match. The text of the posts keeps its original spelling.
- Dead links: the Mistral post's skmultilearn link points to its GitHub repository instead of the defunct scikit.ml, and the diff-pruning and adversarial-training posts add "(TensorBoard.dev has since shut down)" after their TensorBoard links.

## Site source

- `src/content.config.ts`: validation and discovery of posts.
- `src/pages/`: homepage, writing archive, article routes, research, about, RSS, and 404 page. `_projects.astro` is the projects page, kept offline for now: Astro skips files starting with `_`. Rename it to `projects.astro` and add Projects back to the links in `Base.astro` to publish it; the homepage section it had is in commit `9c9184d`.
- `src/layouts/Base.astro`: shared navigation, footer, metadata, fonts, and the script that applies the saved AI slop level before the page paints.
- `src/styles/site.css`: shared base layout and article styles.
- `src/styles/quiet.css`: the Quiet design.
- `src/components/SlopSlider.astro`, `src/lib/slop.ts`, and `src/styles/slop.css`: the AI slop slider, its level names, and its five levels.
- `src/lib/content.ts`: post helpers, the papers list, and project descriptions for the offline projects page.
- `public/`: the site's favicon and future static assets.

Quiet uses a light background, restrained typography, and minimal decoration; AI slop is the deliberate exception. Elsewhere, avoid filler taglines, status notes, decorative counters, and unnecessary panels. Project links come from the imported articles. Your PhD, current position and research interests are stated once, in the About page's first paragraph; keep them up to date there rather than repeating them elsewhere.

## Import provenance

`archive/imports/` holds the original RSS feed, source HTML, metadata, and the eleven recovered embedded code snippets with their exact revision-specific Gist URLs. These source files are kept outside the site's content and are not published as static assets.

The importer writes to this staging archive rather than to the editable `content/` folders:

```sh
python3 scripts/import_medium.py
```

For a saved feed:

```sh
python3 scripts/import_medium.py --feed-file archive/imports/raw/feed.xml
```

Review any new staged posts before copying them into `content/`. Reimporting does not overwrite your edited website articles. RSS coverage is limited to the feed snapshot; a full account export would be needed to establish account-wide completeness.

## Checks

```sh
npm run build
npm test
```

The importer tests check source preservation, image handling, code whitespace, missing embeds, verified embed recovery, and repeatable offline imports. The website build validates article front matter and renders every published post.

## Hosting

The site is published to GitHub Pages at [iondy.com](https://iondy.com). `.github/workflows/deploy.yml` builds it with Node 24 and deploys `dist/` on every push to `main`; you can also start it by hand from the repository's Actions tab. The custom domain is set in the repository's Pages settings, so the build doesn't need a `CNAME` file.

### Coming-soon placeholder

While the `COMING_SOON` repository variable is `true`, the deploy still builds the site but publishes `coming-soon/index.html` instead, for every path (it doubles as the 404 page). The page is not indexed by search engines. To launch the real site:

```sh
gh variable set COMING_SOON --body false
gh workflow run deploy.yml
```

Set the variable back to `true` and run the workflow again to restore the placeholder.

DNS is managed at GoDaddy. `iondy.com` has A records for GitHub Pages (`185.199.108.153`, `185.199.109.153`, `185.199.110.153`, `185.199.111.153`), and `www` is a CNAME to `sirluk.github.io`, which GitHub redirects to `iondy.com`. A `_github-pages-challenge-sirluk` TXT record verifies the domain for the `sirluk` account.

The ignored `.site-preview/` checkout belongs to the earlier private ChatGPT Sites preview and is not used by the GitHub Pages deploy.

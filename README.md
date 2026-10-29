# iondy

A personal website for projects, research, and learning. Built with Astro and plain Markdown; the production output is a static website.

## Run locally

Requires Node.js 22.23.4 or newer.

```sh
npm ci
npm run dev
```

Open `http://127.0.0.1:4321/` for the homepage, which uses the Quiet design by default, or `/directions/` to compare all six designs. The design dock at the bottom of the page works throughout the site and remembers your choice in this browser.

```sh
npm run build
npm run preview
```

The build creates `dist/`, suitable for static hosting, and `npm run preview` serves it at the same address. Fonts, article images, and syntax highlighting are bundled locally; reading does not require third-party scripts or font services.

## AI slop slider

Quiet is the default design. Its header has an **AI slop** slider that visitors can drag from plain Quiet to a full parody of a generated website. Each level adds to the one before:

| Level | Name | Preview | What it adds |
| --- | --- | --- | --- |
| 0 | Off | [`/?slop=0`](http://127.0.0.1:4321/?slop=0) | Plain Quiet. |
| 1 | Mild | [`/?slop=1`](http://127.0.0.1:4321/?slop=1) | Purple accent, faint colour blobs, a gradient wordmark, gradient hairlines, rounded and shadowed code blocks and images. |
| 2 | Medium | [`/?slop=2`](http://127.0.0.1:4321/?slop=2) | The landing-page template: Inter, a wide page, a fading dot grid, a floating glass navbar, a centred hero with a gradient name, glass cards, pill buttons, an author avatar, and a contents card. |
| 3 | Spicy | [`/?slop=3`](http://127.0.0.1:4321/?slop=3) | Gradient buttons, neon card outlines, accent side stripes, a Featured card, count badges, a TL;DR box, oversized outlined numbers, and dark window-framed code blocks. |
| 4 | Hot | [`/?slop=4`](http://127.0.0.1:4321/?slop=4) | Emoji everywhere: headings, badges, categories, dates, and ✅ bullets. |
| 5 | Maximum | [`/?slop=5`](http://127.0.0.1:4321/?slop=5) | Generated copy, a glossy AI orb, "reimagined." in italic serif, "Trusted by teams at localhost", a buzzword ribbon, shimmering gradient text, and stronger glows. |

The level is saved in the visitor's browser and applies to every page; `?slop=<0–5>` links to a level directly, and `&preview=1` previews one without saving it. The slider only appears in Quiet.

- `src/components/SlopSlider.astro` is the slider. Level names come from `slopLevels` in `src/lib/directions.ts`, and `defaultLook` sets the design new visitors see.
- `src/styles/slop.css` holds all five levels. The head script in `Base.astro` writes every level up to the chosen one to `html[data-slop]` (`"1 2 3"` at level 3), so `[data-slop~='3']` means "level 3 or higher". The file sits in its own `overlays` cascade layer after `responsive`, so it overrides Quiet at every screen width and handles its own breakpoints.
- Article text, figures, and code are unchanged at every level. The TL;DR box shows each post's `description`, and the parody copy and emoji are added with CSS and hidden from screen readers.
- Several of the level 1–5 tropes came from asking Codex for recognisable AI-generated-website patterns.

## Preview the designs

Every design renders the same pages, articles, and projects. Add `?look=<id>` to any URL to open it in that design:

| Design | ID | Preview | Character |
| --- | --- | --- | --- |
| Editorial | `editorial` | [`/?look=editorial`](http://127.0.0.1:4321/?look=editorial) | Warm white, serif headlines, green accents, writing and projects alongside each other. |
| Index | `index` | [`/?look=index`](http://127.0.0.1:4321/?look=index) | White, sans-serif and monospace type, blue accents, navigation rail, projects before writing. |
| Paper | `paper` | [`/?look=paper`](http://127.0.0.1:4321/?look=paper) | White, monochrome, one column, sans-serif headings, and serif article text. |
| Margin | `margin` | [`/?look=margin`](http://127.0.0.1:4321/?look=margin) | IBM Plex on a research-paper grid. A margin column holds section labels, dates, archive filters, and the article contents; figures and code use the full main column. |
| Quiet | `quiet` | [`/?look=quiet`](http://127.0.0.1:4321/?look=quiet) | The default. Geist in one narrow column, small type, whitespace instead of rules, navigation beside the name, and title-and-date lists. Has the AI slop slider. |
| Folio | `folio` | [`/?look=folio`](http://127.0.0.1:4321/?look=folio) | Source Serif throughout, centered and book-like, small-caps navigation, figures widened to 860px and code to 760px. |

Articles are the best comparison for reading, code, and figures, for example `/writing/efficient-llm-pretraining-packed-sequences-and-masked-attention/?look=margin`.

A `?look=` link also saves that design as your choice in this browser. Add `&preview=1` to look without saving it, which is what the comparison gallery's iframes do. The directions page is excluded from search indexing.

The new designs use self-hosted Fontsource packages: IBM Plex Sans and Mono (Margin), Geist and Geist Mono (Quiet), and Source Serif 4 with optical sizes (Folio). Browsers download only the fonts a page uses. Designs that are not used can be removed together with their font imports in `Base.astro`.

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

The folder name becomes the URL: `/writing/my-new-post/`. Available categories are `Research`, `Engineering`, and `Learning`. Set `draft: true` to exclude a post from the site, archive, and feed. `original_url` is optional and retains the source link for imported writing. Dates and summaries drive the homepage, archive, and RSS automatically. Article content is independent of the selected design.

The seven imported posts retain their original article wording, 32 images, and 50 code blocks. Short descriptions and broad categories were added for the site's navigation. The titles are preserved as originally written.

## Site source

- `src/content.config.ts`: validation and discovery of posts.
- `src/pages/`: homepage, writing archive, article routes, projects, about, design comparison, RSS, and 404 page.
- `src/layouts/Base.astro`: shared navigation, footer, metadata, and fonts.
- `src/styles/site.css`: shared layout and article styles.
- `src/styles/looks/`: one stylesheet per additional design, imported by `Base.astro`.
- `src/lib/directions.ts`: the single registry used by the design dock, comparison gallery, and initial design selection, plus the slider's level names.
- `src/components/SlopSlider.astro` and `src/styles/slop.css`: the AI slop slider and its five levels, layered on Quiet.
- `src/lib/content.ts`: post helpers and project descriptions/links.
- `public/`: the site's favicon and future static assets.

The designs use light backgrounds, restrained typography, and minimal decoration; AI slop is the deliberate exception. Elsewhere, avoid filler taglines, status notes, decorative counters, and unnecessary panels. Project links come from the imported articles. The introductory copy avoids assuming a current job title, affiliation, or research position.

### Adding another design

Add its metadata to `src/lib/directions.ts` and a stylesheet in `src/styles/looks/<id>.css`, then import that stylesheet in `src/layouts/Base.astro`. Scope styles with `[data-look="<id>"]` inside the existing CSS layers. Both the selector and comparison gallery are generated from the registry and support more than three designs. Use `/?look=<id>` for a direct preview. Preserve the existing designs when adding alternatives. Old `afterhours` bookmarks now resolve to the light Paper design.

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

## Preview hosting

`.openai/hosting.json` identifies the private Sites preview. `.site-preview/` is an ignored, separate publishing checkout so the original GitHub remote and local repository history remain intact. It is a deployment copy, not the editing source. Production content and site source stay in this checkout. A custom domain has not been attached; `iondy.com` can be connected when the design is chosen.

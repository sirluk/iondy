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

The header has an **AI slop** slider that visitors can drag from plain Quiet to a full parody of a generated website. Each level adds to the one before:

| Level | Name | Preview | What it adds |
| --- | --- | --- | --- |
| 0 | Off | [`/?slop=0`](http://127.0.0.1:4321/?slop=0) | Plain Quiet. |
| 1 | Mild | [`/?slop=1`](http://127.0.0.1:4321/?slop=1) | Purple accent, faint colour blobs, a gradient wordmark, gradient hairlines, rounded and shadowed code blocks and images. |
| 2 | Medium | [`/?slop=2`](http://127.0.0.1:4321/?slop=2) | The landing-page template: Inter, a wide page, a fading dot grid, a floating glass navbar, a centred hero with a gradient name, glass cards, pill buttons, an author avatar, and a contents card. |
| 3 | Spicy | [`/?slop=3`](http://127.0.0.1:4321/?slop=3) | Gradient buttons, neon card outlines, accent side stripes, a Featured card, count badges, a TL;DR box, oversized outlined numbers, and dark window-framed code blocks. |
| 4 | Hot | [`/?slop=4`](http://127.0.0.1:4321/?slop=4) | Emoji everywhere: headings, badges, categories, dates, and ✅ bullets. |
| 5 | Maximum | [`/?slop=5`](http://127.0.0.1:4321/?slop=5) | Generated copy, a glossy AI orb, "reimagined." in italic serif, "Trusted by teams at localhost", a buzzword ribbon, shimmering gradient text, and stronger glows. |

The level is saved in the visitor's browser and applies to every page; `?slop=<0–5>` links to a level directly.

- `src/components/SlopSlider.astro` is the slider. Level names come from `slopLevels` in `src/lib/slop.ts`.
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

The folder name becomes the URL: `/writing/my-new-post/`. Available categories are `Research`, `Engineering`, and `Learning`. Set `draft: true` to exclude a post from the site, archive, and feed. `original_url` is optional and retains the source link for imported writing. Dates and summaries drive the homepage, archive, and RSS automatically. Article content is the same at every AI slop level.

The seven imported posts retain their original article wording, 32 images, and 50 code blocks. Short descriptions and broad categories were added for the site's navigation. The titles are preserved as originally written.

## Site source

- `src/content.config.ts`: validation and discovery of posts.
- `src/pages/`: homepage, writing archive, article routes, projects, about, RSS, and 404 page.
- `src/layouts/Base.astro`: shared navigation, footer, metadata, fonts, and the script that applies the saved AI slop level before the page paints.
- `src/styles/site.css`: shared base layout and article styles.
- `src/styles/quiet.css`: the Quiet design.
- `src/components/SlopSlider.astro`, `src/lib/slop.ts`, and `src/styles/slop.css`: the AI slop slider, its level names, and its five levels.
- `src/lib/content.ts`: post helpers and project descriptions/links.
- `public/`: the site's favicon and future static assets.

Quiet uses a light background, restrained typography, and minimal decoration; AI slop is the deliberate exception. Elsewhere, avoid filler taglines, status notes, decorative counters, and unnecessary panels. Project links come from the imported articles. The introductory copy avoids assuming a current job title, affiliation, or research position.

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

DNS is managed at GoDaddy. `iondy.com` has A records for GitHub Pages (`185.199.108.153`, `185.199.109.153`, `185.199.110.153`, `185.199.111.153`), and `www` is a CNAME to `sirluk.github.io`, which GitHub redirects to `iondy.com`. A `_github-pages-challenge-sirluk` TXT record verifies the domain for the `sirluk` account.

`.openai/hosting.json` and the ignored `.site-preview/` checkout belong to the earlier private ChatGPT Sites preview and are not used by the GitHub Pages deploy.

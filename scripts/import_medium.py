#!/usr/bin/env python3
"""Archive a Medium RSS feed as Markdown, original HTML, metadata, and images.

Uses only Python's standard library. Generated imports live under archive/imports;
review and copy new posts to content/ without overwriting editorial changes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
FEED_URL = "https://medium.com/feed/@lukas.hauzenberger"
CONTENT = "{http://purl.org/rss/1.0/modules/content/}encoded"
AUTHOR = "{http://purl.org/dc/elements/1.1/}creator"
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}


def fetch(url: str) -> bytes:
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL: {url}")
    request = Request(url, headers={"User-Agent": "iondy-medium-import/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[Node | str] = field(default_factory=list)


class ArticleParser(HTMLParser):
    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.stack = [self.root]
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        node = Node(tag, {key: value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def walk(node: Node):
    yield node
    for child in node.children:
        if isinstance(child, Node):
            yield from walk(child)


def text_content(node: Node | str) -> str:
    if isinstance(node, str):
        return node
    if node.tag == "br":
        return "\n"
    return "".join(text_content(child) for child in node.children)


def is_tracker(node: Node) -> bool:
    source = urlsplit(node.attrs.get("src", ""))
    return (source.hostname == "medium.com" and source.path == "/_/stat") or (
        node.attrs.get("width") == "1" and node.attrs.get("height") == "1"
    )


def escape_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"([\\`*_\[\]])", r"\\\1", value)
    return value.replace("<", "&lt;").replace(">", "&gt;")


def wrap_inline(text: str, marker: str) -> str:
    match = re.fullmatch(r"(\s*)(.*?)(\s*)", text, re.S)
    before, middle, after = match.groups()
    return f"{before}{marker}{middle}{marker}{after}" if middle else text


class MarkdownRenderer:
    def __init__(self, directory: Path, original_url: str, download_images: bool, fetcher=fetch,
                 embed_sources=None, embed_cache=None):
        self.directory = directory
        self.original_url = original_url
        self.download_images = download_images
        self.fetcher = fetcher
        self.images: dict[str, dict] = {}
        self.warnings: list[dict] = []
        self.code_blocks = 0
        self.inline_code_blocks = 0
        self.removed_trackers = 0
        self.embed_sources = embed_sources or []
        self.embed_cache = embed_cache
        self.embed_number = 0
        self.recovered_embeds = []

    def warn(self, kind: str, **details):
        self.warnings.append({"type": kind, **details})

    def image(self, node: Node) -> str:
        if is_tracker(node):
            self.removed_trackers += 1
            return ""
        url = urljoin(self.original_url, node.attrs.get("src", ""))
        if url not in self.images:
            extension = Path(urlsplit(url).path).suffix.lower()
            if extension not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".svg"}:
                extension = ".img"
            name = hashlib.sha256(url.encode()).hexdigest()[:16] + extension
            relative = f"images/{name}"
            target = self.directory / relative
            record = {"source_url": url, "path": None, "alt": node.attrs.get("alt", "")}
            if self.download_images:
                try:
                    if not target.is_file():
                        data = self.fetcher(url)
                        if not data or data.lstrip().lower().startswith((b"<!doctype html", b"<html")):
                            raise ValueError("Image URL returned an empty response or HTML")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
                    record.update(path=relative, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
                except (OSError, ValueError) as error:
                    self.warn("image_download_failed", url=url, error=str(error))
            else:
                self.warn("image_download_skipped", url=url)
            self.images[url] = record
        record = self.images[url]
        destination = record["path"] or url
        return f"![{escape_text(node.attrs.get('alt', ''))}](<{destination}>)"

    def children(self, node: Node) -> str:
        return "".join(self.render(child) for child in node.children)

    def code_block(self, code: str, language="") -> str:
        self.code_blocks += 1
        fence = "`" * max(3, max((len(x) + 1 for x in re.findall(r"`+", code)), default=0))
        # Keep indentation and blank lines exactly as delivered by the source.
        return f"\n\n{fence}{language}\n{code}{'' if code.endswith(chr(10)) else chr(10)}{fence}\n\n"

    def recover_embed(self, index: int) -> str | None:
        if index >= len(self.embed_sources) or self.embed_cache is None:
            return None
        source = self.embed_sources[index]
        filename = Path(source["filename"]).name
        target = self.embed_cache / filename
        try:
            if not target.is_file():
                code = self.fetcher(source["raw_url"])
                if not code or code.lstrip().lower().startswith((b"<!doctype html", b"<html")):
                    raise ValueError("Code URL returned an empty response or HTML")
                code.decode("utf-8")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(code)
            code = target.read_bytes()
            decoded = code.decode("utf-8")
        except (OSError, ValueError) as error:
            self.warn("embed_download_failed", url=source["raw_url"], error=str(error))
            return None
        self.recovered_embeds.append({
            **source,
            "raw_path": f"raw/embeds/{self.embed_cache.name}/{filename}",
            "sha256": hashlib.sha256(code).hexdigest(),
        })
        return self.code_block(decoded, "python" if filename.endswith(".py") else "")

    def render(self, node: Node | str) -> str:
        if isinstance(node, str):
            return escape_text(node)
        tag = node.tag
        if tag == "img":
            return self.image(node)
        if tag == "pre":
            self.inline_code_blocks += 1
            return self.code_block(text_content(node))
        if tag == "code":
            code = text_content(node)
            fence = "`" * max(1, max((len(x) + 1 for x in re.findall(r"`+", code)), default=0))
            return f"{fence} {code} {fence}"
        if tag in {"iframe", "embed", "script", "object"}:
            if tag == "iframe":
                index = self.embed_number
                self.embed_number += 1
                recovered = self.recover_embed(index)
                if recovered is not None:
                    return recovered
            url = node.attrs.get("src") or node.attrs.get("data")
            self.warn("unavailable_embed", tag=tag, url=url or None)
            link = urljoin(self.original_url, url) if url else self.original_url
            return f"\n\n> Import note: embedded content is unavailable in the RSS feed. [View the original source](<{link}>).\n\n"
        if tag in {"ul", "ol"}:
            lines = []
            start = int(node.attrs.get("start", "1"))
            for index, child in enumerate(c for c in node.children if isinstance(c, Node) and c.tag == "li"):
                marker = f"{start + index}. " if tag == "ol" else "- "
                content = self.children(child).strip().splitlines()
                if content:
                    lines.append(marker + content[0])
                    lines.extend(" " * len(marker) + line if line else "" for line in content[1:])
            return "\n\n" + "\n".join(lines) + "\n\n"
        if tag == "a":
            label = self.children(node)
            href = urljoin(self.original_url, node.attrs.get("href", ""))
            if urlsplit(href).scheme not in {"http", "https", "mailto"}:
                self.warn("unsupported_link", url=href)
                return label
            return f"[{label}](<{href.replace('>', '%3E')}>)"
        if tag == "br":
            return "  \n"
        if tag == "hr":
            return "\n\n---\n\n"
        body = self.children(node)
        if re.fullmatch(r"h[1-6]", tag):
            # Medium starts article sections at h3; front matter supplies the title.
            level = max(2, int(tag[1]) - 1)
            return f"\n\n{'#' * level} {body.strip()}\n\n"
        if tag in {"p", "div", "figure", "section"}:
            return f"\n\n{body.strip()}\n\n" if body.strip() else ""
        if tag == "blockquote":
            return "\n\n" + "\n".join("> " + line for line in body.strip().splitlines()) + "\n\n"
        if tag in {"strong", "b"}:
            return wrap_inline(body, "**")
        if tag == "figcaption":
            return "\n\n" + wrap_inline(body.strip(), "*") + "\n\n"
        if tag in {"em", "i"}:
            return wrap_inline(body, "*")
        if tag in {"del", "s"}:
            return wrap_inline(body, "~~")
        if tag not in {"root", "span", "li", "small"}:
            self.warn("unsupported_html_tag", tag=tag)
        return body


def frontmatter(metadata: dict) -> str:
    # JSON strings and arrays are also valid YAML, including embedded colons/quotes.
    return "---\n" + "\n".join(
        f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in metadata.items()
    ) + "\n---\n\n"


def clean_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def import_feed(feed: bytes, output: Path, download_images=True, fetcher=fetch, embed_sources=None) -> dict:
    channel = ET.fromstring(feed).find("channel")
    if channel is None:
        raise ValueError("Expected an RSS feed with a channel")
    items = channel.findall("item")
    if not items:
        raise ValueError("The feed contains no posts; existing files have been left untouched")
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "feed.xml").write_bytes(feed)
    posts, seen = [], set()
    duplicates = 0
    for item in items:
        title = item.findtext("title", "").strip()
        original = clean_url(item.findtext("link", ""))
        match = re.search(r"([a-f0-9]{12})$", urlsplit(original).path)
        if not title or not match:
            raise ValueError(f"Post is missing a title or stable Medium ID: {original}")
        medium_id = match.group(1)
        if medium_id in seen:
            duplicates += 1
            continue
        seen.add(medium_id)
        published = parsedate_to_datetime(item.findtext("pubDate", "")).isoformat()
        slug = urlsplit(original).path.rsplit("/", 1)[-1].removesuffix("-" + medium_id)
        slug = re.sub(r"[^a-z0-9-]", "-", slug.lower()).strip("-") or medium_id
        folder = f"{published[:10]}-{slug}-{medium_id}"
        directory = output / "posts" / folder
        directory.mkdir(parents=True, exist_ok=True)
        html = item.findtext(CONTENT, "")
        if not html.strip():
            raise ValueError(f"Post has no article body: {original}")
        (raw / f"{medium_id}.html").write_text(html, encoding="utf-8")
        tree = ArticleParser(html).root
        html_sha256 = hashlib.sha256(html.encode()).hexdigest()
        sources = (embed_sources or {}).get(medium_id, {})
        embeds = sources.get("embeds", [])
        source_matches = sources.get("html_sha256") == html_sha256
        count_matches = len(embeds) == sum(node.tag == "iframe" for node in walk(tree))
        renderer = MarkdownRenderer(
            directory, original, download_images, fetcher,
            embed_sources=embeds if source_matches and count_matches else [],
            embed_cache=raw / "embeds" / medium_id,
        )
        if embeds and not (source_matches and count_matches):
            renderer.warn("embed_manifest_source_mismatch", message="RSS HTML or iframe count changed; check the saved embed mapping before reuse.")
        markdown = renderer.render(tree).strip() + "\n"
        intro = next((text_content(n) for n in walk(tree) if n.tag == "p" and text_content(n).strip()), "")
        description = " ".join(intro.split())
        if len(description) > 220:
            description = description[:217].rsplit(" ", 1)[0] + "…"
        status = "needs_review" if renderer.warnings else "imported"
        images = list(renderer.images.values())
        metadata = {
            "title": title,
            "slug": slug,
            "author": item.findtext(AUTHOR, "").strip(),
            "published_at": published,
            "description": description,
            "tags": [tag.text for tag in item.findall("category") if tag.text],
            "source": "medium",
            "original_url": original,
            "medium_id": medium_id,
            "import_status": status,
        }
        if images:
            metadata["image"] = images[0]["path"] or images[0]["source_url"]
        (directory / "index.md").write_text(frontmatter(metadata) + markdown, encoding="utf-8")
        posts.append({
            **metadata,
            "markdown_path": f"posts/{folder}/index.md",
            "original_html_path": f"raw/{medium_id}.html",
            "html_sha256": html_sha256,
            "images": images,
            "code_blocks": renderer.code_blocks,
            "inline_code_blocks": renderer.inline_code_blocks,
            "recovered_embeds": renderer.recovered_embeds,
            "removed_tracking_pixels": renderer.removed_trackers,
            "warnings": renderer.warnings,
            "project_links": list(dict.fromkeys(
                node.attrs["href"] for node in walk(tree)
                if node.tag == "a" and urlsplit(node.attrs.get("href", "")).hostname == "github.com"
            )),
        })
        print(f"Imported: {title} ({status})", flush=True)
    posts.sort(key=lambda post: post["published_at"], reverse=True)
    write_json(output / "index.json", posts)
    report = {
        "feed_url": FEED_URL,
        "feed_title": channel.findtext("title"),
        "feed_last_build_date": channel.findtext("lastBuildDate"),
        "feed_sha256": hashlib.sha256(feed).hexdigest(),
        "coverage": "Posts present in this RSS snapshot only; not a verified complete Medium account export.",
        "post_count": len(posts),
        "duplicate_items_skipped": duplicates,
        "downloaded_images": sum(bool(image["path"]) for post in posts for image in post["images"]),
        "inline_code_blocks": sum(post["inline_code_blocks"] for post in posts),
        "recovered_code_embeds": sum(len(post["recovered_embeds"]) for post in posts),
        "total_code_blocks": sum(post["code_blocks"] for post in posts),
        "removed_tracking_pixels": sum(post["removed_tracking_pixels"] for post in posts),
        "posts_needing_review": [
            {"title": post["title"], "medium_id": post["medium_id"], "warnings": post["warnings"]}
            for post in posts if post["warnings"]
        ],
    }
    write_json(output / "import-report.json", report)
    overview = [
        "# Medium archive", "",
        f"{len(posts)} posts from [Lukas Hauzenberger's Medium profile](https://medium.com/@lukas.hauzenberger).",
        f"{report['downloaded_images']} local images and {report['total_code_blocks']} code blocks, "
        f"including {report['recovered_code_embeds']} recovered from the original article embeds.", "",
        "| Published | Article | Import status |", "| --- | --- | --- |",
    ]
    for post in posts:
        title = post["title"].replace("|", "\\|")
        status = "Needs review" if post["warnings"] else "Imported"
        overview.append(f"| {post['published_at'][:10]} | [{title}]({post['markdown_path']}) | {status} |")
    overview.extend([
        "", "Original text, dates, tags, links, images, captions, and available code have been retained.",
        "Original RSS/HTML and recovered embed source files are in `raw/`. `index.json` records provenance and warnings.",
        "", "Coverage is limited to this RSS snapshot; a full Medium account export would be needed to verify account-wide completeness.",
        "", "This directory is generated. Keep edited/published versions separately; rerunning the importer replaces these Markdown files.", "",
    ])
    (output / "README.md").write_text("\n".join(overview), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed-file", type=Path, help="Read a saved RSS XML file instead of fetching the feed")
    parser.add_argument("--output", type=Path, default=ROOT / "archive" / "imports")
    parser.add_argument("--skip-images", action="store_true", help="Keep remote image links; do not download images")
    parser.add_argument("--embeds-file", type=Path, help="Article-verified embed mapping; defaults to OUTPUT/embed-sources.json")
    args = parser.parse_args()
    try:
        feed = args.feed_file.read_bytes() if args.feed_file else fetch(FEED_URL)
        embeds_file = args.embeds_file or args.output / "embed-sources.json"
        sources = json.loads(embeds_file.read_text()) if embeds_file.is_file() else {}
        report = import_feed(feed, args.output, not args.skip_images, embed_sources=sources)
    except (OSError, ValueError, ET.ParseError) as error:
        print(f"Import failed: {error}", file=sys.stderr)
        return 1
    print(f"\n{report['post_count']} posts; {report['downloaded_images']} local images; "
          f"{len(report['posts_needing_review'])} posts need review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

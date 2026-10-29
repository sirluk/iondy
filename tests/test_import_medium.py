"""Regression checks for preserving article content during the Medium import."""

from contextlib import redirect_stdout
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


SPEC = spec_from_file_location("import_medium", Path(__file__).resolve().parents[1] / "scripts/import_medium.py")
medium = module_from_spec(SPEC)
sys.modules[SPEC.name] = medium
SPEC.loader.exec_module(medium)


def feed(body, duplicate=False):
    item = f"""<item><title>A: test &amp; example</title>
    <link>https://medium.com/@example/a-test-123456abcdef?source=rss</link>
    <pubDate>Mon, 07 Oct 2024 06:47:26 GMT</pubDate>
    <dc:creator>Example Author</dc:creator><category>pytorch</category>
    <content:encoded><![CDATA[{body}]]></content:encoded></item>"""
    return (f'<rss xmlns:content="http://purl.org/rss/1.0/modules/content/" '
            f'xmlns:dc="http://purl.org/dc/elements/1.1/"><channel><title>Stories</title>'
            f'{item}{item if duplicate else ""}</channel></rss>').encode()


class ImportTests(unittest.TestCase):
    def run_import(self, html, directory, **kwargs):
        with redirect_stdout(io.StringIO()):
            report = medium.import_feed(feed(html), directory, **kwargs)
        post = json.loads((directory / "index.json").read_text())[0]
        return report, post, (directory / post["markdown_path"]).read_text()

    def test_code_preserves_indentation_blank_lines_and_entities(self):
        html = '<p>Use x_y and *literal*.</p><pre>if x &lt; 2:<br>    print("&amp;")<br><br>    # ```</pre>'
        with tempfile.TemporaryDirectory() as folder:
            report, post, markdown = self.run_import(html, Path(folder))
            self.assertIn('````\nif x < 2:\n    print("&")\n\n    # ```\n````', markdown)
            self.assertIn(r"x\_y and \*literal\*", markdown)
            self.assertEqual(report["inline_code_blocks"], 1)
            self.assertEqual(post["published_at"], "2024-10-07T06:47:26+00:00")

    def test_images_captions_links_and_tracking_removal(self):
        html = '<figure><img src="https://cdn.example.com/a.png"><figcaption>Credit <a href="https://example.com">Author</a></figcaption></figure><img src="https://medium.com/_/stat?postId=1" width="1" height="1">'
        with tempfile.TemporaryDirectory() as folder:
            report, post, markdown = self.run_import(html, Path(folder), fetcher=lambda url: b"image data")
            self.assertEqual(report["downloaded_images"], 1)
            self.assertEqual(report["removed_tracking_pixels"], 1)
            self.assertIn('images/', markdown)
            self.assertIn('*Credit [Author](<https://example.com>)*', markdown)
            self.assertNotIn('/_/stat', markdown)
            asset = Path(folder) / Path(post["markdown_path"]).parent / post["images"][0]["path"]
            self.assertEqual(asset.read_bytes(), b"image data")

    def test_missing_embeds_are_visible_and_flagged(self):
        with tempfile.TemporaryDirectory() as folder:
            report, post, markdown = self.run_import('<p>Example below:</p><iframe src=""></iframe>', Path(folder))
            self.assertEqual(post["import_status"], "needs_review")
            self.assertEqual(report["posts_needing_review"][0]["warnings"][0]["type"], "unavailable_embed")
            self.assertIn('Import note:', markdown)
            self.assertIn('https://medium.com/@example/a-test-123456abcdef', markdown)

    def test_verified_embeds_are_recovered_at_the_correct_location_and_cached(self):
        html = '<p>Before.</p><iframe src=""></iframe><p>After.</p>'
        sources = {"123456abcdef": {
            "html_sha256": hashlib.sha256(html.encode()).hexdigest(),
            "embeds": [{"filename": "example.py", "raw_url": "https://gist.github.com/example/raw/revision/example.py",
                        "medium_embed_url": "https://medium.com/media/example"}],
        }}
        code = b"def example():\n    return 42\n"
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            report, post, markdown = self.run_import(html, directory, embed_sources=sources, fetcher=lambda url: code)
            self.assertEqual(post["import_status"], "imported")
            self.assertEqual(report["recovered_code_embeds"], 1)
            self.assertEqual(report["total_code_blocks"], 1)
            self.assertEqual(report["inline_code_blocks"], 0)
            self.assertLess(markdown.index('Before.'), markdown.index(code.decode()))
            self.assertLess(markdown.index(code.decode()), markdown.index('After.'))
            self.assertEqual((directory / post["recovered_embeds"][0]["raw_path"]).read_bytes(), code)
            def no_network(url):
                self.fail("Cached code should not be fetched again")
            _, _, second = self.run_import(html, directory, embed_sources=sources, fetcher=no_network)
            self.assertEqual(markdown, second)

    def test_changed_article_does_not_reuse_an_old_embed_mapping(self):
        html = '<p>New version.</p><iframe src=""></iframe>'
        sources = {"123456abcdef": {
            "html_sha256": "outdated",
            "embeds": [{"filename": "example.py", "raw_url": "https://example.com/example.py"}],
        }}
        with tempfile.TemporaryDirectory() as folder:
            report, post, markdown = self.run_import(html, Path(folder), embed_sources=sources)
            self.assertEqual(report["recovered_code_embeds"], 0)
            self.assertIn("embed_manifest_source_mismatch", [w["type"] for w in post["warnings"]])
            self.assertIn("Import note:", markdown)

    def test_failed_image_download_retains_source_link(self):
        def fail(url):
            raise OSError("Unavailable")
        with tempfile.TemporaryDirectory() as folder:
            report, post, markdown = self.run_import('<img src="https://cdn.example.com/a.png">', Path(folder), fetcher=fail)
            self.assertEqual(report["downloaded_images"], 0)
            self.assertIn('https://cdn.example.com/a.png', markdown)
            self.assertEqual(post["warnings"][0]["type"], "image_download_failed")

    def test_lists_and_inline_spacing(self):
        renderer = medium.MarkdownRenderer(Path('/unused'), 'https://example.com', False)
        html = '<p>A <strong>bold </strong>word.</p><ol><li>First<ul><li>Nested</li></ul></li><li>Second<br>continued</li></ol>'
        result = renderer.render(medium.ArticleParser(html).root)
        self.assertIn('A **bold** word.', result)
        self.assertIn('1. First\n\n   - Nested', result)
        self.assertIn('2. Second  \n   continued', result)

    def test_raw_sources_deduplication_and_offline_repeatability(self):
        html = '<p>Original wording.</p><pre>    keep whitespace</pre>'
        source = feed(html, duplicate=True)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with redirect_stdout(io.StringIO()):
                report = medium.import_feed(source, directory)
                first = {p.relative_to(directory): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
                medium.import_feed(source, directory)
            second = {p.relative_to(directory): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
            self.assertEqual(first, second)
            self.assertEqual(report['post_count'], 1)
            self.assertEqual(report['duplicate_items_skipped'], 1)
            self.assertEqual((directory / 'raw/123456abcdef.html').read_text(), html)
            self.assertEqual(report['feed_sha256'], hashlib.sha256(source).hexdigest())


if __name__ == '__main__':
    unittest.main()

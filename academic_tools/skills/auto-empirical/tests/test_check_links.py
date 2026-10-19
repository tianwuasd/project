"""Tests for external-link checking helpers."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest

from _helpers import load_module

check_links = load_module("scripts/check-links.py", "aers_check_links")


class TestLinkExtraction(unittest.TestCase):
    def test_markdown_code_fences_are_skipped_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = check_links.Path(tmp)
            md = root / "README.md"
            md.write_text(
                "[rendered](https://example.com/page.)\n"
                "```markdown\n"
                "[example](https://example.invalid/placeholder)\n"
                "```\n",
                encoding="utf-8",
            )
            old_root = check_links.ROOT
            try:
                check_links.ROOT = root
                links = check_links.iter_links([md])
                self.assertEqual(links, {"https://example.com/page": ["README.md"]})

                with_examples = check_links.iter_links([md], include_code_fences=True)
                self.assertIn("https://example.invalid/placeholder", with_examples)
            finally:
                check_links.ROOT = old_root

    def test_html_links_and_repeated_files_are_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = check_links.Path(tmp)
            html = root / "docs" / "search.html"
            html.parent.mkdir()
            html.write_text(
                '<a href="https://example.com/a">one</a>'
                '<a href="https://example.com/a">two</a>',
                encoding="utf-8",
            )
            old_root = check_links.ROOT
            try:
                check_links.ROOT = root
                self.assertEqual(
                    check_links.iter_links([html]),
                    {"https://example.com/a": ["docs/search.html"]},
                )
            finally:
                check_links.ROOT = old_root


class TestCheckLinksCli(unittest.TestCase):
    def test_main_can_write_or_skip_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = check_links.Path(tmp)
            md = root / "README.md"
            md.write_text("[site](https://example.com)\n", encoding="utf-8")
            output = root / "catalog" / "external-link-check.json"

            old_root = check_links.ROOT
            old_maintained_docs = check_links.maintained_docs
            old_check_url = check_links.check_url
            try:
                check_links.ROOT = root
                check_links.maintained_docs = lambda: [md]
                check_links.check_url = lambda url, timeout: {
                    "url": url,
                    "status": 200,
                    "ok": True,
                }
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(check_links.main(["--output", str(output)]), 0)
                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(payload["checked_links"], 1)
                self.assertEqual(payload["failures"], [])

                output.unlink()
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        check_links.main(["--output", str(output), "--no-write"]),
                        0,
                    )
                self.assertFalse(output.exists())
            finally:
                check_links.ROOT = old_root
                check_links.maintained_docs = old_maintained_docs
                check_links.check_url = old_check_url

    def test_main_returns_failure_for_broken_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = check_links.Path(tmp)
            md = root / "README.md"
            md.write_text("[site](https://example.invalid)\n", encoding="utf-8")

            old_root = check_links.ROOT
            old_maintained_docs = check_links.maintained_docs
            old_check_url = check_links.check_url
            try:
                check_links.ROOT = root
                check_links.maintained_docs = lambda: [md]
                check_links.check_url = lambda url, timeout: {
                    "url": url,
                    "status": 404,
                    "ok": False,
                }
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(check_links.main(["--no-write"]), 1)
            finally:
                check_links.ROOT = old_root
                check_links.maintained_docs = old_maintained_docs
                check_links.check_url = old_check_url


class TestSignInGatedGithubPages(unittest.TestCase):
    """GitHub answers signed-out clients with 404 on /stargazers and /watchers.

    Left unhandled that is a permanent, unfixable failure in the weekly
    link-check job: the page works in a browser, so no edit to the repo can
    make the checker happy. The rule is deliberately narrow, and these tests
    exist mostly to keep it that way — a general 404 amnesty would silently
    stop catching real dead links.
    """

    def test_the_two_gated_paths_are_recognized(self):
        for url in (
            "https://github.com/owner/repo/stargazers",
            "https://github.com/owner/repo/watchers",
            "https://github.com/owner/repo/stargazers/",
        ):
            with self.subTest(url=url):
                self.assertTrue(check_links.is_login_gated(url))

    def test_public_github_paths_are_not_exempt(self):
        for url in (
            "https://github.com/owner/repo",
            "https://github.com/owner/repo/forks",
            "https://github.com/owner/repo/network/members",
            "https://github.com/owner/repo/issues",
            "https://github.com/owner/repo/pulls",
            "https://github.com/owner/repo/blob/main/README.md",
        ):
            with self.subTest(url=url):
                self.assertFalse(check_links.is_login_gated(url))

    def test_the_pattern_is_anchored_to_github_com(self):
        # A path that merely contains the string must not inherit the exemption.
        for url in (
            "https://evil.example/github.com/owner/repo/stargazers",
            "https://github.com.evil.example/owner/repo/stargazers",
            "https://github.com/owner/repo/stargazers/extra",
        ):
            with self.subTest(url=url):
                self.assertFalse(check_links.is_login_gated(url))

    def test_a_gated_404_is_reported_as_reachable_with_a_warning(self):
        result = check_links.login_gated_result(
            "https://github.com/owner/repo/stargazers", 404
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], 404)
        self.assertIn("Sign-in-gated", result["warning"])

    def test_a_404_elsewhere_is_still_a_failure(self):
        import urllib.error

        def raise_404(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 404, "Not Found", hdrs=None, fp=None
            )

        old_urlopen = check_links.urllib.request.urlopen
        try:
            check_links.urllib.request.urlopen = raise_404
            gated = check_links.check_url(
                "https://github.com/owner/repo/stargazers", timeout=1
            )
            real = check_links.check_url("https://github.com/owner/repo/gone", timeout=1)
        finally:
            check_links.urllib.request.urlopen = old_urlopen
        self.assertTrue(gated["ok"])
        self.assertFalse(real["ok"], "the exemption must not generalize to any 404")


if __name__ == "__main__":
    unittest.main()

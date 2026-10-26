"""Private GitHub routing without credentials or external network in tests."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aaoca_server import network, updater
from aaoca_server.panel import Panel, main


class NetworkTests(unittest.TestCase):
    def test_private_default_and_legacy_mirrors_cannot_enable_public_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "absent"
            self.assertEqual(
                network.routes(network.load(root)), (network.HTTPS_UPSTREAM,)
            )
            self.assertFalse(root.exists())
            with self.assertRaises(ValueError):
                network.save(root, git_transport="public-mirrors")
            self.assertFalse(root.exists())

    def test_scoped_proxy_and_https_credentials_retained(self):
        before = dict(os.environ)
        calls = []

        def run(args, **kw):
            calls.append((args, kw))
            return (network.HTTPS_UPSTREAM + "\n").encode()

        with network.use(dict(network.DEFAULT, proxy="http://127.0.0.1:17891")):
            with patch.object(updater, "command", side_effect=run):
                updater.git(Path("."), "ls-remote", network.HTTPS_UPSTREAM, "HEAD")
        args, kw = calls[-1]
        self.assertNotIn("credential.helper=", args)
        self.assertNotIn("GIT_CONFIG_GLOBAL", kw["env"])
        self.assertIn("http.followRedirects=false", args)
        self.assertIn(
            "http." + network.HTTPS_UPSTREAM + ".proxy=http://127.0.0.1:17891", args
        )
        self.assertEqual(kw["env"]["NO_PROXY"], "")
        self.assertEqual(kw["env"]["https_proxy"], "http://127.0.0.1:17891")
        self.assertEqual(dict(os.environ), before)

    def test_url_rewrite_is_rejected_before_network(self):
        with patch.object(
            updater, "command", return_value=b"https://mirror.invalid/project.git\n"
        ) as run:
            with self.assertRaisesRegex(ValueError, "重写"):
                updater.git(Path("."), "ls-remote", network.HTTPS_UPSTREAM, "HEAD")
        self.assertEqual(run.call_count, 1)
        self.assertIn("--get-url", run.call_args.args[0])

    def test_git_errors_do_not_disclose_credentials(self):
        with patch.object(
            updater,
            "command",
            side_effect=[network.HTTPS_UPSTREAM.encode(), ValueError("secret-token")],
        ):
            with self.assertRaises(ValueError) as caught:
                updater.git(Path("."), "fetch", "origin", "main")
        self.assertNotIn("secret-token", str(caught.exception))
        self.assertIn("认证", str(caught.exception))

    def test_ssh_uses_explicit_key_strict_host_and_scoped_connect_proxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "synthetic key"
            key.touch()
            settings = dict(
                network.DEFAULT,
                git_transport="github-ssh",
                ssh_key=str(key),
                proxy="http://127.0.0.1:17891",
            )
            with (
                network.use(settings),
                patch.object(network.shutil, "which", return_value="/usr/bin/nc"),
            ):
                env = network.child_env(ssh=True)
            command = env["GIT_SSH_COMMAND"]
            for text in [
                "IdentitiesOnly=yes",
                "StrictHostKeyChecking=yes",
                "BatchMode=yes",
                "-p 443",
                "-X connect",
                "17891",
            ]:
                self.assertIn(text, command)
            with (
                network.use(settings),
                patch.object(network.shutil, "which", return_value=None),
            ):
                with self.assertRaisesRegex(ValueError, "nc"):
                    network.child_env(ssh=True)
            key.unlink()
            with network.use(settings):
                with self.assertRaisesRegex(ValueError, "密钥路径"):
                    network.child_env(ssh=True)

    def test_invalid_and_corrupt_settings_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for proxy in [
                "http://user:secret@127.0.0.1:1",
                "https://evil.test",
                "http://127.0.0.1:1/path",
            ]:
                with self.assertRaises(ValueError):
                    network.save(root, proxy=proxy)
            (root / "network.json").write_text("{broken")
            with self.assertRaises(ValueError):
                network.save(root, proxy="")
            self.assertEqual((root / "network.json").read_text(), "{broken")

    def test_menu_cancel_and_cli_show_are_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "panel"
            answers = iter(["a", "http://127.0.0.1:17891", "b", "q"])
            Panel(root, ask=lambda _: next(answers), emit=lambda _: None).run()
            self.assertFalse(root.exists())
            self.assertEqual(main(["--runtime", str(root), "--network-status"]), 0)
            self.assertFalse(root.exists())

    def test_network_cli_saves_only_nonsecret_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "panel"
            self.assertEqual(
                main(
                    [
                        "--runtime",
                        str(root),
                        "--configure-network",
                        "--github-proxy",
                        "http://127.0.0.1:17891",
                        "--github-transport",
                        "github-ssh",
                        "--github-ssh-key",
                        "~/.ssh/project_readonly_ed25519",
                    ]
                ),
                0,
            )
            self.assertEqual(network.load(root)["git_transport"], "github-ssh")
            self.assertEqual(
                {p.name for p in root.iterdir()}, {"network.json", ".dispatch.lock"}
            )

    def test_bootstrap_freezes_actual_old_release_and_restores_process_context(self):
        from aaoca_server import bootstrap

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old-release"
            initial = updater.CODE_ROOT
            before = dict(os.environ)
            with (
                patch.object(updater, "managed_release", return_value=old),
                patch.object(
                    updater,
                    "update",
                    side_effect=lambda registry, emit: {"root": updater.CODE_ROOT},
                ) as run,
            ):
                result = bootstrap.upgrade(
                    root / "runtime", root / "checkout", emit=lambda _: None
                )
            self.assertEqual(result["root"], old)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(updater.CODE_ROOT, initial)
            self.assertEqual(dict(os.environ), before)

    def test_bootstrap_requires_original_entry_when_no_release(self):
        from aaoca_server import bootstrap

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "旧入口"):
                bootstrap.upgrade(Path(tmp) / "runtime", Path(tmp) / "checkout")


if __name__ == "__main__":
    unittest.main()

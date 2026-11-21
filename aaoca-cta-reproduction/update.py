import argparse
import importlib.util
from pathlib import Path
import sys


def shared_updater(path):
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("aaoca_shared_code_updater", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SUBDIRECTORY = "aaoca-cta-reproduction"
    module.ROOT_FILES = {".gitignore", ".gitattributes", "README.md", "THIRD_PARTY.md",
                         "pyproject.toml", "run.py", "smoke.py", "verify_inputs.py", "update.py", "setup_env.py"}

    class ProjectRepository(module.GitRepository):
        def choose_source(self, *args, **kwargs):
            self.target_commit = super().choose_source(*args, **kwargs)
            return self.target_commit

        def inventory(self, commit):
            # 首次安装时旧提交可以没有本工程；目标提交仍须满足完整入口检查。
            if commit != self.target_commit and not self.run("ls-tree", commit, "--", module.SUBDIRECTORY):
                return {}
            return super().inventory(commit)

    module.GitRepository = ProjectRepository
    return module


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--shared-updater", default=str(Path(__file__).resolve().parent.parent / "chd-segmentation/update.py"))
    args, remaining = parser.parse_known_args()
    if "--relocate" in remaining or "--remove-legacy" in remaining:
        raise ValueError("本工程不支持迁移或清理其他项目的历史目录")
    updater = shared_updater(args.shared_updater)
    base = Path(__file__).resolve().parent.parent
    if not any(arg == "--repo" or arg.startswith("--repo=") for arg in remaining):
        remaining.extend(["--repo", str(base)])
    if not any(arg == "--runtime" or arg.startswith("--runtime=") for arg in remaining):
        remaining.extend(["--runtime", str(base.parent / "course/aaoca-cta-reproduction")])
    sys.argv = [str(Path(__file__).resolve()), *remaining]
    updater.main()


if __name__ == "__main__":
    main()

"""Standalone entry point; does not depend on the AAOCA desktop application."""
from aaoca_seg.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

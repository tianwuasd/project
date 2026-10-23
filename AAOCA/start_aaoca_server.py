"""Standalone server entry; never imports the desktop raw-data workflow."""
from aaoca_server.panel import main

if __name__ == '__main__':
    raise SystemExit(main())

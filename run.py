from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from chd_mini.cli import main
    main()

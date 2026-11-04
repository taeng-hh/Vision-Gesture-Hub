"""Standalone sign recognizer; optional legacy --live-fusion flag is accepted."""
import sys
from fusion.cli import main
if __name__ == "__main__":
    if "--live-fusion" in sys.argv:sys.argv.remove("--live-fusion")
    main()

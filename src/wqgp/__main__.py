"""Allow `python -m wqgp` to invoke the CLI."""
import sys

from wqgp.cli import main

if __name__ == "__main__":
    sys.exit(main())
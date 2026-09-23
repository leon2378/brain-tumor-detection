"""Allow ``python -m btd``."""

from btd.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

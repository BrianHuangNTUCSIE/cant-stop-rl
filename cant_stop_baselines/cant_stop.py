"""Compatibility facade and CLI for the modular Can't Stop baseline.

Importing from this module remains supported; new code may import from
``cant_stop_core`` or one of its focused submodules.
"""

from cant_stop_core import *  # noqa: F401,F403
from cant_stop_core.cli import main


if __name__ == "__main__":
    main()

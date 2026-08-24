"""Make the ``pdstl`` package importable from the repository root.

The project has no ``pytest.ini``/``pyproject.toml`` and ``setup.py`` installs
the tree under the name ``src``, so ``import pdstl`` only resolves when ``src/``
is on ``sys.path``. Running ``src/main.py`` as a script gets that for free;
pytest does not.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

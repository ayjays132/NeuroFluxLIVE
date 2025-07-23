"""
SelfResearch – package wrapper & legacy-import shim.

• Adds the repo root to sys.path so internal modules resolve.
• Makes SelfResearch a namespace pkg (PEP-420-friendly).
• Aliases old absolute imports like  `import analysis.foo`
  to the new fully-qualified path `SelfResearch.analysis.foo`.
"""
__version__ = "1.0.2"
from pathlib import Path
import importlib, pkgutil, sys as _sys

# ------------------------------------------------------------------ #
# 1.  Put repo root on sys.path                                      #
# ------------------------------------------------------------------ #
_ROOT = Path(__file__).resolve().parent      # …/SelfResearch
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

# ------------------------------------------------------------------ #
# 2.  Support implicit namespace packages                            #
# ------------------------------------------------------------------ #
__path__ = pkgutil.extend_path(__path__, __name__)

# ------------------------------------------------------------------ #
# 3.  Alias legacy top-level names                                   #
# ------------------------------------------------------------------ #
_forwards = (
    "analysis", "data", "train", "models", "eval",
    "security", "assessment", "peer_collab",
    "simulation_lab", "digital_literacy", "research_workflow"
)

for _name in _forwards:
    fq = f"{__name__}.{_name}"            # e.g. SelfResearch.analysis
    try:
        _sys.modules[_name] = importlib.import_module(fq)
    except ModuleNotFoundError:
        # skip if the folder doesn’t actually exist
        pass

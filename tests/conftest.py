import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
FAQ_PATH = (
    APP_ROOT
    / "examples"
    / "rag_faq.md"
)
DOCUASK_BACKEND_FAQ_PATH = (
    APP_ROOT
    / "examples"
    / "docuask_backend_faq.md"
)
ONCALL_RUNBOOK_PATH = (
    APP_ROOT
    / "examples"
    / "oncall_runbook.md"
)

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

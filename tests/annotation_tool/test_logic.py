import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "js" / "annotation_logic_test.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_page_logic_passes_its_node_checks():
    result = subprocess.run(["node", str(SCRIPT)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all checks passed" in result.stdout

import re
from pathlib import Path

BUILD_SH = Path(__file__).resolve().parents[3] / "capture" / "cellular" / "build.sh"


def test_itpp_clone_is_pinned_to_commit_sha():
    text = BUILD_SH.read_text()
    assert "git clone" in text and "itpp" in text
    # IT++ is built and installed with sudo, so it must not track upstream HEAD.
    assert re.search(r"git -C \"\$ITPP_SRC\" checkout --detach [0-9a-f]{40}\b", text), (
        "build.sh must check out a pinned 40-hex IT++ commit after cloning"
    )

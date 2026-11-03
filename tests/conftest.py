"""共用夹具：脚本各跑一次，结果在整个测试会话里复用。只读仓库已有数据，不调用任何模型或 API。"""
import json, subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for d in ("scripts", "analysis"):
    if str(ROOT / d) not in sys.path:
        sys.path.insert(0, str(ROOT / d))


def run(script, *args, check=True):
    return subprocess.run([sys.executable, str(ROOT / script), *args], capture_output=True, text=True,
                          encoding="utf-8", check=check, cwd=ROOT)


@pytest.fixture(scope="session")
def metrics(tmp_path_factory):
    """reproduce_metrics.py --json 的结果与打印输出。"""
    out = tmp_path_factory.mktemp("metrics") / "m.json"
    p = run("scripts/reproduce_metrics.py", "--json", str(out))
    return {"json": json.loads(out.read_text(encoding="utf-8")), "stdout": p.stdout}


@pytest.fixture(scope="session")
def robust(tmp_path_factory):
    """analysis/robustness.py 按默认参数（与提交的 稳健性分析.md 相同）跑一次。"""
    d = tmp_path_factory.mktemp("robust")
    p = run("analysis/robustness.py", "--out", str(d / "r.md"), "--json", str(d / "r.json"))
    return {"md": (d / "r.md").read_text(encoding="utf-8"), "json": json.loads((d / "r.json").read_text(encoding="utf-8")),
            "stdout": p.stdout}

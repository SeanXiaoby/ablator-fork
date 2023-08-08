import io
import random
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
import pandas as pd
import pytest

import ray

from ablator.modules.loggers.file import FileLogger, RemoteFileLogger


def assert_console_output(fn, assert_fn):
    f = io.StringIO()
    with redirect_stdout(f):
        fn()
    s = f.getvalue()
    assert assert_fn(s)


def test_file_logger(tmp_path: Path):
    logpath = tmp_path.joinpath("test.log")
    l = FileLogger(logpath, verbose=True, prefix="1")
    assert_console_output(lambda: l.info("hello"), lambda s: s.endswith("1 - hello\n"))
    lines = logpath.read_text().split("\n")
    assert len(lines) == 3
    assert lines[0].startswith("Starting Logger")
    assert lines[1].endswith("hello")
    assert_console_output(lambda: l.warn("hello"), lambda s: s.endswith("1 - \x1b[93mhello\x1b[0m\n"))
    lines = logpath.read_text().split("\n")
    assert len(lines) == 4
    assert lines[0].startswith("Starting Logger")
    assert lines[2].endswith("1 - \x1b[93mhello\x1b[0m")

    assert_console_output(lambda: l.error("hello"), lambda s: s.endswith("1 - \x1b[91mhello\x1b[0m\n"))
    lines = logpath.read_text().split("\n")
    assert len(lines) == 5
    assert lines[0].startswith("Starting Logger")
    assert lines[3].endswith("\x1b[91mhello\x1b[0m")

    l.verbose = False
    assert_console_output(lambda: l.info("hello"), lambda s: len(s) == 0)
    assert_console_output(
        lambda: l.info("hello", verbose=True), lambda s: s.endswith("hello\n")
    )
    assert_console_output(
        lambda: l.warn("hello"), lambda s: s.endswith("1 - \x1b[93mhello\x1b[0m\n")
    )
    assert_console_output(lambda: l.warn("hello", verbose=False), lambda s: len(s) == 0)
    assert_console_output(
        lambda: l.error("hello"), lambda s: s.endswith("\x1b[91mhello\x1b[0m\n")
    )


@ray.remote
def mock_remote(i: int, file_logger: FileLogger):
    file_logger.info(f"\\xx {i} info \\xx")
    file_logger.warn(f"\\xx {i} warn \\xx")
    file_logger.error(f"\\xx {i} error \\xx")




def test_remote_file_logger(tmp_path: Path, ray_cluster):
    logpath = tmp_path.joinpath("test.log")
    if logpath.exists():
        logpath.unlink()
    node_ips = ray_cluster.node_ips()
    l = RemoteFileLogger(logpath, verbose=True, prefix="1")
    l.to_remote()
    ray.get(
        [
            mock_remote.options(
                resources={f"node:{random.choice(node_ips)}": 0.001}
            ).remote(i, l)
            for i in range(100)
        ]
    )

    def clean_msg(msg):
        _, trial_id, msg, _ = msg.split(" ")
        return {"trial_id": trial_id, "msg": msg}

    df = pd.DataFrame(
        list(map(clean_msg, re.findall("\\\\xx.*\\\\xx", logpath.read_text())))
    )
    assert set(df["trial_id"].unique().astype(int)) == set(range(100))
    assert df.nunique()["trial_id"] == 100
    assert df.nunique()["msg"] == 3
    assert df.shape[0] == 3 * 100


if __name__ == "__main__":
    test_remote_file_logger(Path("/tmp/"))

    pass

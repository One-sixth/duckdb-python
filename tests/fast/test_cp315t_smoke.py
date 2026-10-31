"""Smoke test for the free-threaded (cp315t) wheel.

Skips automatically on GIL-enabled interpreters. On a free-threaded
interpreter it verifies:

* the module declares ``Py_mod_gil = NOT_USED`` — the GIL stays disabled
  after ``import duckdb`` (regression guard for the nanobind ``-ft`` variant);
* a basic query works;
* a 12-thread concurrent workload (random queries + DDL storm) runs with
  zero errors under a true no-GIL runtime.

Self-contained: only needs ``duckdb`` itself, no numpy/pandas.
"""

import random
import sys
import threading

import pytest

# Skip on GIL-enabled interpreters and on Python < 3.13 (this test is only
# meaningful on a free-threaded build with the GIL disabled).
# sys._is_gil_enabled() exists only on Python >= 3.13; the getattr fallback
# returns True there so older interpreters skip instead of raising
# AttributeError at collection time.
pytestmark = pytest.mark.skipif(
    getattr(sys, "_is_gil_enabled", lambda: True)(),
    reason="free-threaded interpreter with GIL disabled only",
)


def test_gil_stays_disabled_after_import():
    # The module must declare Py_mod_gil = NOT_USED so that importing it does
    # not re-enable the GIL. If this fails, the wheel was built without the
    # nanobind free-threaded variant.
    import duckdb

    assert not sys._is_gil_enabled(), "duckdb re-enabled the GIL on import"
    assert duckdb.__version__


def test_basic_query():
    import duckdb

    assert duckdb.connect().execute("select 41 + 1").fetchone() == (42,)


def test_concurrent_workload():
    import duckdb

    errors: list[str] = []
    lock = threading.Lock()

    def worker(tid: int) -> None:
        rng = random.Random(tid)
        try:
            c = duckdb.connect()
            cur = c.cursor()
            # Random read workload with exact expected values.
            for i in range(200):
                n = rng.randint(10, 500)
                got = cur.execute(f"select sum(range) from range({n})").fetchone()[0]
                exp = n * (n - 1) // 2
                if got != exp:
                    with lock:
                        errors.append(f"t{tid} iter{i}: sum mismatch {got} != {exp}")
            # DDL storm: create/drop many small tables.
            for i in range(20):
                t = f"t{tid}_{i}"
                cur.execute(f"create table {t} as select range({rng.randint(1, 100)}) r")
                assert cur.execute(f"select count(*) from {t}").fetchone()[0] >= 1
                cur.execute(f"drop table {t}")
            c.close()
        except Exception as e:
            with lock:
                errors.append(f"t{tid}: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"{len(errors)} errors: {errors[:5]}"

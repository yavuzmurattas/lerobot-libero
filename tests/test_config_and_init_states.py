"""Regression tests for first-import config creation and init-state loading.

Every test runs Python in a subprocess with ``HOME`` and ``LIBERO_CONFIG_PATH``
pointing at a temporary directory, so the user's real ``~/.libero`` is never
read or written.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import yaml

EXPECTED_KEYS = {"benchmark_root", "bddl_files", "init_states", "datasets", "assets"}
PROMPT = "Do you want to specify a custom path"


def _env(tmp_path, **extra):
    env = os.environ.copy()
    env.pop("LIBERO_NONINTERACTIVE", None)
    env["HOME"] = str(tmp_path)
    env["LIBERO_CONFIG_PATH"] = str(tmp_path / "libero_cfg")
    env.update(extra)
    return env


def _run(code, env, stdin=subprocess.DEVNULL, timeout=120):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        timeout=timeout,
    )


def _read_config(tmp_path):
    with open(tmp_path / "libero_cfg" / "config.yaml") as f:
        return yaml.safe_load(f)


def test_first_import_without_tty_writes_default_config(tmp_path):
    proc = _run("import libero.libero", _env(tmp_path), stdin=subprocess.DEVNULL)
    assert proc.returncode == 0, proc.stdout
    assert PROMPT not in proc.stdout
    config = _read_config(tmp_path)
    assert set(config) == EXPECTED_KEYS
    assert os.path.isdir(config["init_states"])


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="needs a pseudo-terminal")
def test_first_import_with_tty_still_prompts(tmp_path):
    master, slave = os.openpty()
    try:
        os.write(master, b"n\n")
        proc = _run("import libero.libero", _env(tmp_path), stdin=slave)
    finally:
        os.close(slave)
        os.close(master)
    assert proc.returncode == 0, proc.stdout
    assert PROMPT in proc.stdout
    assert set(_read_config(tmp_path)) == EXPECTED_KEYS


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="needs a pseudo-terminal")
def test_env_var_disables_prompt_even_with_tty(tmp_path):
    master, slave = os.openpty()
    try:
        proc = _run(
            "import libero.libero",
            _env(tmp_path, LIBERO_NONINTERACTIVE="1"),
            stdin=slave,
            timeout=60,
        )
    finally:
        os.close(slave)
        os.close(master)
    assert proc.returncode == 0, proc.stdout
    assert PROMPT not in proc.stdout
    assert set(_read_config(tmp_path)) == EXPECTED_KEYS


def _write_default_config(tmp_path):
    """Write a config up front so only init-state loading is exercised."""
    import importlib.util

    pkg_dir = importlib.util.find_spec("libero.libero").submodule_search_locations[0]
    cfg_dir = tmp_path / "libero_cfg"
    cfg_dir.mkdir()
    with open(cfg_dir / "config.yaml", "w") as f:
        yaml.safe_dump(
            {
                "benchmark_root": pkg_dir,
                "bddl_files": os.path.join(pkg_dir, "bddl_files"),
                "init_states": os.path.join(pkg_dir, "init_files"),
                "datasets": os.path.join(pkg_dir, "../datasets"),
                "assets": os.path.join(pkg_dir, "assets"),
            },
            f,
        )


def test_bundled_init_states_load(tmp_path):
    _write_default_config(tmp_path)
    code = """
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()["libero_10"]()
        states = suite.get_task_init_states(0)
        print("SHAPE", tuple(states.shape))
    """
    proc = _run(code, _env(tmp_path))
    assert proc.returncode == 0, proc.stdout
    assert "SHAPE (" in proc.stdout


def test_init_states_with_numpy_payload_load(tmp_path):
    """A pickled numpy array (as in the bundled files) needs weights_only=False."""
    _write_default_config(tmp_path)
    states_file = tmp_path / "suite" / "task.pruned_init"
    code = f"""
        import os
        import numpy as np
        import torch
        from libero.libero import benchmark

        path = {str(states_file)!r}
        os.makedirs(os.path.dirname(path))
        torch.save(np.arange(6, dtype=np.float64).reshape(2, 3), path)

        task = benchmark.Task("task", "task", "Libero", "suite", "task.bddl", "task.pruned_init")
        suite = benchmark.get_benchmark_dict()["libero_10"]()
        suite.tasks = [task]
        benchmark.get_libero_path = lambda key: {str(tmp_path)!r}
        states = suite.get_task_init_states(0)
        assert isinstance(states, np.ndarray) and states.shape == (2, 3), states
        print("OK")
    """
    proc = _run(code, _env(tmp_path))
    assert proc.returncode == 0, proc.stdout
    assert "OK" in proc.stdout

from backend.__main__ import PROFILE, browser_command


def test_gpu_window_forces_the_gpu_in_its_own_profile():
    command = browser_command("edge", "http://127.0.0.1:8000/")
    assert "--force_high_performance_gpu" in command
    assert f"--user-data-dir={PROFILE}" in command   # a running browser would otherwise swallow the flags
    assert command[-1] == "--app=http://127.0.0.1:8000/"

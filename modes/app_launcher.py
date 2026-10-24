"""Optional Windows application launcher. Only known commands are permitted.

Not connected to a gesture yet: wire up deliberately after adding activation,
confidence and cooldown checks. Never execute arbitrary recognized text as a command.
"""
import subprocess

ALLOWED_APPS = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
}


def launch_app(app_key: str):
    if app_key not in ALLOWED_APPS:
        raise ValueError(f"Application not allowlisted: {app_key}")
    return subprocess.Popen(ALLOWED_APPS[app_key])

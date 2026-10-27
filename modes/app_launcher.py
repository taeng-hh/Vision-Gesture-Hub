"""Optional Windows application launcher. Only known commands are permitted.

Never execute arbitrary recognized text as a command.
"""
import subprocess
import webbrowser

ALLOWED_APPS = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
    "paint": ["mspaint.exe"],
}


def launch_app(app_key: str):
    if app_key == "chatgpt":
        return webbrowser.open("https://chatgpt.com/", new=2)
    if app_key not in ALLOWED_APPS:
        raise ValueError(f"Application not allowlisted: {app_key}")
    return subprocess.Popen(ALLOWED_APPS[app_key])

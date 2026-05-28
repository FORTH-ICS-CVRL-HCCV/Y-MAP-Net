"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""

import os
import sys
import shutil as _shutil
import subprocess


#-------------------------------------------------------------------------------
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


#-------------------------------------------------------------------------------
def read_json_file(file_path):
    import json
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{file_path}'.")
    except Exception as e:
        print(f"Error: {e}")


#-------------------------------------------------------------------------------
def checkIfPathExists(filename):
    return os.path.exists(filename)


#-------------------------------------------------------------------------------
def checkIfPathIsDirectory(filename):
    return os.path.isdir(filename)


#-------------------------------------------------------------------------------
def checkIfFileExists(filename):
    return os.path.isfile(filename)


#-------------------------------------------------------------------------------
def convert_bytes(num):
    #This function will convert bytes to MB.... GB... strings
    step_unit = 1000.0  #1024 bad the size
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < step_unit:
            return "%3.1f %s" % (num, x)
        num /= step_unit


#-------------------------------------------------------------------------------
def inrange(min_value, value, max_value):
    return max(min_value, min(value, max_value))


# =============================================================================
# Cross-platform system wrappers
#
# Each function below tries the Python stdlib path first (works on Linux,
# macOS, Windows) and falls back to a shell-out only if the stdlib attempt
# raises. If both paths fail, the wrapper raises a single descriptive
# exception so callers don't have to deal with platform-specific error modes.
# =============================================================================

# Tracks which external binaries have already triggered a "not on PATH"
# warning, so run_tool doesn't spam the console when the same tool is
# called repeatedly (e.g. ffmpeg every frame in some workflows).
_warned_missing_binaries = set()


#-------------------------------------------------------------------------------
def warn_unsupported_platform():
    """Warn once on Windows/macOS that several Linux/X11-only tools used by
    the demo (xset, xdotool, scrot, xrandr, xdpyinfo) will be skipped.

    Inference still works; the affected features are screensaver
    suppression, screenshot capture, and multi-monitor screen detection.
    """
    if sys.platform.startswith("linux"):
        return
    plat = "Windows" if sys.platform == "win32" else (
        "macOS" if sys.platform == "darwin" else sys.platform)
    print(bcolors.WARNING +
          f"[warn] {plat} detected. Screensaver suppression, screenshot capture, "
          "and multi-monitor screen detection rely on Linux/X11 tools "
          "(xset, xdotool, scrot, xrandr, xdpyinfo) and will be skipped." +
          bcolors.ENDC)


#-------------------------------------------------------------------------------
def run_tool(cmd, optional=True, capture_output=False, timeout=None, check=False):
    """Run an external CLI tool. Friendly handling for missing binaries.

    If `cmd[0]` is not on PATH:
      * optional=True (default): warn once per missing binary, return None.
      * optional=False: raise RuntimeError.

    Otherwise behaves like subprocess.run with sensible defaults. When
    capture_output is True the result is returned as text. Returns a
    CompletedProcess on success, or None when the binary is missing and
    optional=True.
    """
    binary = cmd[0]
    if _shutil.which(binary) is None:
        if optional:
            if binary not in _warned_missing_binaries:
                print(bcolors.WARNING +
                      f"[run_tool] '{binary}' not found on PATH; skipping." +
                      bcolors.ENDC)
                _warned_missing_binaries.add(binary)
            return None
        raise RuntimeError(f"'{binary}' not found on PATH")
    return subprocess.run(cmd, check=check, capture_output=capture_output,
                          text=capture_output, timeout=timeout)


#-------------------------------------------------------------------------------
def rm_rf(path):
    """Recursively remove a directory or file. Stdlib first, `rm -rf` fallback.

    Idempotent: returns silently if `path` does not exist. Raises OSError
    only if every removal attempt fails.
    """
    import shutil
    if not os.path.exists(path):
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return
    except Exception as primary_err:
        if sys.platform != "win32" and _shutil.which("rm") is not None:
            r = subprocess.run(["rm", "-rf", path], check=False)
            if r.returncode == 0:
                return
        raise OSError(f"rm_rf({path!r}) failed: {primary_err}") from primary_err


#-------------------------------------------------------------------------------
def download(url, dest):
    """Download `url` to `dest`. Tries urllib first, then wget, then curl.

    Raises RuntimeError if every method fails.
    """
    import urllib.request
    last_err = None
    try:
        urllib.request.urlretrieve(url, dest)
        return
    except Exception as e:
        last_err = e

    for cli in (["wget", "-O", dest, url],
                ["curl", "-L", "-o", dest, url]):
        if _shutil.which(cli[0]) is None:
            continue
        r = subprocess.run(cli, check=False)
        if r.returncode == 0 and os.path.exists(dest):
            return
        last_err = RuntimeError(f"{cli[0]} returned {r.returncode}")

    raise RuntimeError(f"download({url!r}, {dest!r}) failed: {last_err}")


#-------------------------------------------------------------------------------
def unzip(zip_path, dest="."):
    """Extract `zip_path` into `dest`. Tries stdlib zipfile, then `unzip` CLI.

    Raises RuntimeError if every method fails.
    """
    import zipfile
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        return
    except Exception as primary_err:
        if _shutil.which("unzip") is not None:
            r = subprocess.run(["unzip", "-o", zip_path, "-d", dest], check=False)
            if r.returncode == 0:
                return
        raise RuntimeError(f"unzip({zip_path!r}, {dest!r}) failed: {primary_err}") \
            from primary_err


#-------------------------------------------------------------------------------
def upload_file(url, filepath, field="file"):
    """POST a multipart/form-data file upload. Equivalent to `curl -F`.

    Tries stdlib urllib first (no external dependency); falls back to the
    curl binary if available. Raises RuntimeError on total failure.
    """
    import mimetypes
    import uuid
    import urllib.request
    last_err = None

    # 1. Pure-Python multipart via urllib.
    try:
        boundary = uuid.uuid4().hex
        with open(filepath, "rb") as f:
            file_bytes = f.read()
        ctype = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
        filename = os.path.basename(filepath)
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        return
    except Exception as e:
        last_err = e

    # 2. curl fallback.
    if _shutil.which("curl") is not None:
        r = subprocess.run(
            ["curl", "-F", f"{field}=@{filepath}", url], check=False,
        )
        if r.returncode == 0:
            return
        last_err = RuntimeError(f"curl returned {r.returncode}")

    raise RuntimeError(f"upload_file({url!r}, {filepath!r}) failed: {last_err}")


#-------------------------------------------------------------------------------

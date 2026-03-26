@echo off
setlocal EnableDelayedExpansion

:: =============================================================================
:: YMAPNet - Windows setup and run script
:: Requires: Python 3.10-3.12 installed and on PATH, internet access
:: Run once to set up; subsequent runs skip steps already completed.
:: =============================================================================

:: Script lives in scripts\ so the repo root is one level up
set REPO_DIR=%~dp0..
set VENV_DIR=%REPO_DIR%\venv
set MODEL_DIR=%REPO_DIR%\2d_pose_estimation
set MODEL_ZIP=%REPO_DIR%\2d_pose_estimation.zip
set MODEL_URL=http://ammar.gr/ymapnet/archive/2d_pose_estimation_v264_onnx.zip

echo.
echo ============================================================
echo  YMAPNet Windows Setup
echo ============================================================
echo.

:: ---------------------------------------------------------------------------
:: 1. Check Python
:: ---------------------------------------------------------------------------
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    echo        Install Python 3.10-3.12 from https://www.python.org/downloads/
    echo        and make sure to tick "Add Python to PATH" during installation.
    goto :error
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo        Found Python %PY_VER%

:: ---------------------------------------------------------------------------
:: 2. Create virtual environment
:: ---------------------------------------------------------------------------
echo [2/5] Setting up virtual environment...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo        Creating venv at %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        goto :error
    )
    echo        Created.
) else (
    echo        Already exists, skipping.
)

:: ---------------------------------------------------------------------------
:: 3. Install dependencies
:: ---------------------------------------------------------------------------
echo [3/5] Installing Python dependencies...
call "%VENV_DIR%\Scripts\activate.bat"

:: Upgrade pip quietly first
python -m pip install --upgrade pip --quiet

:: tensorflow[and-cuda] targets Linux/WSL2 GPU. On Windows we install the
:: standard tensorflow package (CPU inference; GPU requires WSL2 + CUDA).
:: Everything else in requirements.txt installs as-is.
echo        Installing packages from requirements.txt ...
echo        (tensorflow will be installed as CPU-only on Windows)

python -m pip install ^
    tf_keras ^
    numpy ^
    numba ^
    etils ^
    importlib_resources ^
    wget ^
    onnx ^
    onnxruntime ^
    opencv-python ^
    gradio

if errorlevel 1 (
    echo ERROR: pip install failed.
    goto :error
)
echo        Dependencies installed.

:: ---------------------------------------------------------------------------
:: 4. Download and extract the pretrained model
:: ---------------------------------------------------------------------------
echo [4/5] Checking pretrained model...
if exist "%MODEL_DIR%\configuration.json" (
    echo        Model already present, skipping download.
) else (
    echo        Downloading model from %MODEL_URL% ...
    echo        This may take a few minutes depending on your connection.

    :: curl.exe ships with Windows 10 1803+. The .exe suffix avoids
    :: conflicts with any PowerShell curl alias.
    curl.exe -L --progress-bar -o "%MODEL_ZIP%" "%MODEL_URL%"
    if errorlevel 1 (
        echo ERROR: Download failed. Check your internet connection.
        goto :error
    )
    echo        Download complete. Extracting...

    :: Expand-Archive is available on all PowerShell 5+ systems (Win10+).
    powershell -NoProfile -Command ^
        "Expand-Archive -Path '%MODEL_ZIP%' -DestinationPath '%REPO_DIR%' -Force"
    if errorlevel 1 (
        echo ERROR: Extraction failed.
        goto :error
    )

    :: Clean up the zip to save disk space
    del /f /q "%MODEL_ZIP%"
    echo        Model ready.
)

:: ---------------------------------------------------------------------------
:: 5. Run YMAPNet
:: ---------------------------------------------------------------------------
echo [5/5] Launching YMAPNet...
echo.
echo        Starting webcam inference. Press Q in the preview window to quit.
echo        Other options:
echo          scripts\run_windows.bat --from path\to\video.mp4
echo          scripts\run_windows.bat --cpu
echo          scripts\run_windows.bat --update   (re-download the model)
echo.

cd /d "%REPO_DIR%"
python runYMAPNet.py --engine onnx %*

goto :end

:: ---------------------------------------------------------------------------
:error
echo.
echo ============================================================
echo  Setup failed. See error messages above.
echo ============================================================
endlocal
exit /b 1

:end
endlocal
exit /b 0

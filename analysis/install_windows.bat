@echo off
rem Double-click to install the TONIQ analysis environment on Windows (requires Miniforge or Miniconda).
cd /d "%~dp0"
call :find_conda
if not defined CONDA_ACTIVATE (
    echo Could not find Miniforge or Miniconda. Install Miniforge first: https://conda-forge.org/download/
    pause
    exit /b 1
)
call "%CONDA_ACTIVATE%"
echo Creating the toniq environment. This takes several minutes...
call conda env update -n toniq -f environment.yml || goto :failed
call conda activate toniq || goto :failed
pip install -e . || goto :failed
pip install notebook || goto :failed
echo.
echo Installation complete. Double-click start_notebook.bat to begin.
pause
exit /b 0

:failed
echo.
echo Installation failed. Please send a screenshot of this window for help.
pause
exit /b 1

:find_conda
for %%D in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\miniforge3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
    if exist "%%~D\Scripts\activate.bat" if not defined CONDA_ACTIVATE set "CONDA_ACTIVATE=%%~D\Scripts\activate.bat"
)
exit /b 0

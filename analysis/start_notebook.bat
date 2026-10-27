@echo off
rem Double-click to open the pedicle screw analysis notebook in your web browser.
cd /d "%~dp0"
call :find_conda
if not defined CONDA_ACTIVATE (
    echo Could not find Miniforge or Miniconda. Run install_windows.bat first.
    pause
    exit /b 1
)
call "%CONDA_ACTIVATE%"
call conda activate toniq || (
    echo The toniq environment is missing. Run install_windows.bat first.
    pause
    exit /b 1
)
cd notebooks
jupyter notebook pedicle_screw_analysis.ipynb
pause
exit /b 0

:find_conda
for %%D in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\miniforge3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
    if exist "%%~D\Scripts\activate.bat" if not defined CONDA_ACTIVATE set "CONDA_ACTIVATE=%%~D\Scripts\activate.bat"
)
exit /b 0

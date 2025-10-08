@echo off
TITLE Python Dependencies Installer

ECHO #######################################################
ECHO # Installing required Python libraries...             #
ECHO # This may take a few minutes. Please be patient.     #
ECHO #######################################################
ECHO.

REM Change directory to the location of this script to find requirements.txt
cd /d "%~dp0"

py -m pip install -r requirements.txt

ECHO.
ECHO #######################################################
ECHO # Installation complete.                              #
ECHO # You can now run the model scripts.                  #
ECHO #######################################################
ECHO.
pause
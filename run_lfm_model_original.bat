@echo off
TITLE Run LFM2-VL Base Model

ECHO #######################################################
ECHO # Starting the LFM2-VL Base Model...                  #
ECHO # Please wait for the model to load.                  #
ECHO #######################################################
ECHO.

REM Change directory to the location of this script
cd /d "%~dp0"

ECHO Starting the Python server in a new window...
start "LFM2-VL Server" python running_model_original_lfm2_vl.py

ECHO Waiting for the server to initialize...
timeout /t 15 /nobreak > nul

ECHO Opening the Gradio interface in your default browser...
start http://127.0.0.1:7861

ECHO.
ECHO The server is now running in a separate window.
ECHO To stop the model, simply close that new command prompt window.
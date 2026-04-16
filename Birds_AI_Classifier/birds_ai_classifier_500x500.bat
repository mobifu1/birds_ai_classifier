@echo off
REM Titel für das Konsolenfenster setzen
TITLE Birds AI Classifier Launcher

REM Informiere den Benutzer
echo Starte Birds AI Classifier...

REM Wechsele in das Projektverzeichnis
REM Der Parameter /d stellt sicher, dass auch das Laufwerk gewechselt wird
cd /d "C:\Users\Andreas\source\repos\Birds_AI_Classifier\Birds_AI_Classifier"

REM Führe das Python-Skript aus
python birds_ai_classifier_500x500.py

REM Pausiere am Ende, damit das Fenster bei einem Fehler nicht sofort zugeht
echo.
echo Das Programm wurde beendet.
pause
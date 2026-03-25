@echo off
echo Syncing to HPC (10.16.1.50)...

:: Sync ngr-yolov8 folder to ~/ngr-yolov8 on remote host via WSL
:: Excludes huge directories and python cache
wsl rsync -avz --progress --exclude ".git" --exclude "__pycache__" --exclude "runs" --exclude "yolo_dataset" --exclude "archive" --exclude "archive2" ./ abhyam.121822@10.16.1.50:~/ngr-yolov8/

pause

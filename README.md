# How to
1. Set input and output paths in `docker-compose.yml`
- Input path is the path of folder containing images you want to run the pipeline on. The code will read all files in the folder.
- Output path is the path where the output CSV file will be saved on your machine.

2. Run Docker Compose file
```
    docker compose up
```
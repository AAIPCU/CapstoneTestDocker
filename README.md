# How to
1. Set input and output paths in `docker-compose.yml`
- Input path is the path of folder containing images you want to run the pipeline on. The code will read all files in the folder.
- Output path is the path where the output CSV file will be saved on your machine.

2. Set sample_limit in config.yaml, this will limit the number of files that will be processed by the pipeline. Set the value to `-1` to process all files in the folder.

3. Run Docker Compose file
```
    docker compose up
```
> *Build can take more than 5 minutes due to libraries installation.*
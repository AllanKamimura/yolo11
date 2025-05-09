# yolo11
Function to use [Ultralytics](https://github.com/ultralytics/ultralytics/tree/main) yolo11.

## Init

### Clone this repo.
```bash
git clone https://github.com/AllanKamimura/yolo11
cd yolo11
```

## Export

### Use Docker Container

#### Edit the `export.yaml` file.
1. Under `volumes`, change the home path to your user.
    1. `/home/<your-user>/yolo11-outside:/app/outside`
2. This is going to set a path **in your PC** where you can configure the script
    1. `/home/<your-user>/yolo11-outside`
   
#### Start the Export Container
```
docker compose --file export.yaml up -d
docker exec -it yolo11-export
```

1. You should be able to see the folder specified above
    1.  `/home/<your-user>/yolo11-outside`


#### Run the script
1. Inside the container, navigate to `/app/outside`
2. Check the `export.py` file
3. Run the script
    ```shell
    python3 export.py
    ```
    1. This should take a few minutes (15~20mins).
4. Check the output file
    1. `/home/<your-user>/yolo11-outside/yolo11n-seg_saved_model/yolo11n-seg_full_integer_quant.tflite`

## Predict

### Create a new vevn

```shell
python3 -m venv yolo11
source yolo11/bin/activate
```

### Install the dependencies (inside venv)

```shell
pip install -r requirements.txt
```

### Run the Script

1. Get some sample images

    ```shell
    wget https://ultralytics.com/images/bus.jpg
    wget https://ultralytics.com/images/zidane.jpg
    ```

1. Copy the previous exported model

1. Run script

    ```shell
    python3 main.py
    ```
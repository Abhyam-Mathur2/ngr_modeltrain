# NGR Pothole and Street Light Detection

This project uses YOLOv8 to detect potholes (broken streets) and street lights, and classifies the severity of potholes.

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Run the following command to install required libraries:
```bash
pip install -r requirements.txt
```

### 2. Download Datasets
The datasets are not included in this repository. You need to download them using the provided scripts:
```bash
python download_roboflow.py
python download_defects.py
# Note: You may need to manually download 'archive2' and 'pothole-detection-DatasetNinja' if scripts for them are not fully automated.
```

### 3. Prepare Dataset
Once the raw data is downloaded, run the preparation script to organize it into the YOLO format:
```bash
python prepare_dataset.py
```

### 4. Training (Optional)
If you don't have the pre-trained weights (`best.pt`), you can train the model yourself:
```bash
python train.py
```
This will create a `runs/` directory with the trained weights.

### 5. Running the Inference Server
To start the FastAPI server for detection:
```bash
python main.py
```
The server will be available at `http://localhost:8000`. You can interact with it via the `index.html` interface or by sending POST requests to `/predict`.

## Project Structure
- `main.py`: FastAPI server for real-time inference.
- `train.py`: Script to train the YOLOv8 detection model.
- `train_severity.py`: Script to train the pothole severity classification model.
- `prepare_dataset.py`: Combines and formats multiple datasets for training.
- `data.yaml`: Configuration for YOLOv8 training.
- `index.html`: Simple frontend for testing the detection API.

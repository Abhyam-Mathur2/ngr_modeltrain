from roboflow import Roboflow
import os

rf = Roboflow(api_key="QHN6PQYVjTQUuZgQ6atn")
project = rf.workspace("vinu-priyanka").project("street-light-detection-sznuq")
version = project.version(1)
dataset = version.download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")

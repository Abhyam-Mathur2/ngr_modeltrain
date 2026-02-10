from roboflow import Roboflow
import os

rf = Roboflow(api_key="QHN6PQYVjTQUuZgQ6atn")
project = rf.workspace("sashank-s").project("street-light")
version = project.version(1)
dataset = version.download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")
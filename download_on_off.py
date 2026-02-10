from roboflow import Roboflow
import os

rf = Roboflow(api_key="QHN6PQYVjTQUuZgQ6atn")
project = rf.workspace("mohamed-traore-26wiu").project("street-lights-j99vk")
version = project.version(2) # Version 2 seems to be the latest
dataset = version.download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")

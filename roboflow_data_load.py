!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="K66OYXtvYb56ZXv8k6eC")
project = rf.workspace("debotosh").project("cracks-3ii36-tekeg")
version = project.version(5)
dataset = version.download("coco")
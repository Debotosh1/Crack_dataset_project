import time
import torch
import cv2
import os
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = TextSAM("/content/drive/MyDrive/sam_vit_b_01ec64.pth").to(device)
model.load_state_dict(torch.load("/content/drive/MyDrive/model_epoch_9.pth"))
model.eval()

# Input/output paths
img_dir = "/content/segment-anything/cracks-5/test"
# pick ONE image
img_name = os.listdir(img_dir)[0]
img_path = os.path.join(img_dir, img_name)

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (1024, 1024))

img_tensor = torch.tensor(img).permute(2,0,1).float()/255.
img_tensor = img_tensor.unsqueeze(0).to(device)

prompts = ["segment crack", "segment wall crack"]

#  Warmup (VERY IMPORTANT for GPU)
for _ in range(5):
    with torch.no_grad():
        for p in prompts:
            _ = model(img_tensor, [p])

# Timing
runs = 100
start = time.time()

for _ in range(runs):
    with torch.no_grad():
        preds = []
        for p in prompts:
            out = model(img_tensor, [p])
            out = torch.sigmoid(out)
            preds.append(out)

        pred = torch.mean(torch.stack(preds), dim=0)

# GPU sync (IMPORTANT)
if device == "cuda":
    torch.cuda.synchronize()

end = time.time()

avg_time = (end - start) / runs
fps = 1 / avg_time

print(f"Avg inference time: {avg_time:.4f} sec")
print(f"FPS: {fps:.2f}")
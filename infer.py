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
out_dir = "/content/pred_masks"
os.makedirs(out_dir, exist_ok=True)

#prompt = "segment crack"   #  change if needed

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024))

    img_tensor = torch.tensor(img).permute(2,0,1).float()/255.
    img_tensor = img_tensor.unsqueeze(0).to(device)

    prompts = ["segment crack", "segment wall crack"]

    with torch.no_grad():
        preds = []
        for p in prompts:
            out = model(img_tensor, [p])
            out = torch.sigmoid(out)
            preds.append(out)

        pred = torch.mean(torch.stack(preds), dim=0)
    #  CHANGE ENDS HERE

    pred = pred[0,0].cpu().numpy()

    # Threshold
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    # Save
    save_name = img_name.split(".")[0] + "__segment_crack.png"
    cv2.imwrite(os.path.join(out_dir, save_name), pred_mask)

print(" Inference Done!")
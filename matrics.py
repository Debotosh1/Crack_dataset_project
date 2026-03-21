def compute_metrics(pred, gt):
    pred = (pred > 0).astype(np.float32)   # already 0/255 → convert to 0/1
    gt = (gt > 0).astype(np.float32)

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection

    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-6)

    return iou, dice
mask_dir = "/content/segment-anything/cracks-5/test_masks1"
img_dir = "/content/segment-anything/cracks-5/test"
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

total_iou = 0
total_dice = 0
count = 0

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    mask_path = os.path.join(mask_dir, img_name)

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024))

    img_tensor = torch.tensor(img).permute(2,0,1).float()/255.
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Prediction
    prompts = ["segment crack", "segment wall crack"]

    with torch.no_grad():
        preds = []
        for p in prompts:
            out = model(img_tensor, [p])
            out = torch.sigmoid(out)
            preds.append(out)

        pred = torch.mean(torch.stack(preds), dim=0)
        pred = pred[0,0].cpu().numpy()

    # Threshold
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    # Load GT mask
    gt = cv2.imread(mask_path, 0)
    if gt is None:
        continue

    gt = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    # Metrics
    iou, dice = compute_metrics(pred_mask, gt)
    print(iou)

    total_iou += iou
    total_dice += dice
    count += 1

# Final scores
print("mIoU:", total_iou / count)
print("Dice:", total_dice / count)
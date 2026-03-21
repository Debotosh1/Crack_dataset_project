def focal_loss(logits, targets, alpha=0.25, gamma=2):
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    return loss.mean()
def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-5
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = TextSAM("/content/drive/MyDrive/sam_vit_b_01ec64.pth").to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

print(f"Total params: {total_params/1e6:.2f}M")
print(f"Trainable params: {trainable_params/1e6:.2f}M")
print(f"Frozen params: {frozen_params/1e6:.2f}M")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

train_dataset = SegTextDataset(
    "/content/segment-anything/cracks-5/train",
    "/content/segment-anything/cracks-5/train_masks",
    ["segment crack", "segment wall crack"]
)
print("Train dataset size:", len(train_dataset))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataset = SegTextDataset(
    "/content/segment-anything/cracks-5/valid",
    "/content/segment-anything/cracks-5/valid_masks",
    ["segment crack", "segment wall crack"]
)
print("val dataset size:", len(val_dataset))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

for epoch in range(10):
    model.train()
    train_loss = 0

    for imgs, masks, texts, _ in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
       # print("Masks shape:", masks.shape)
       # print("Masks range:", masks.min().item(), masks.max().item())

        preds = model(imgs, texts)
        preds = torch.clamp(preds, -10, 10)
      #  print("Preds shape:", preds.shape)
        #print("Preds range:", preds.min().item(), preds.max().item())
        loss = 0.3 * focal_loss(preds, masks) + 0.7 * dice_loss(preds, masks)

        print("Train Loss:",loss.detach().cpu())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    model.eval()
    val_loss = 0

    with torch.no_grad():   # VERY IMPORTANT
        for imgs, masks, texts, _ in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs, texts)

            loss = 0.3 * focal_loss(preds, masks) + 0.7 * dice_loss(preds, masks)
            print("Val Loss:",loss.detach().cpu())

            val_loss += loss.item()

    val_loss /= len(val_loader)

    # -------------------


    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    torch.save(model.state_dict(), f"/content/drive/MyDrive/model_epoch_{epoch}.pth")

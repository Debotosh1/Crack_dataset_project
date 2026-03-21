import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from transformers import CLIPTextModel, CLIPTokenizer


class TextSAM(nn.Module):
    def __init__(self, sam_ckpt):
        super().__init__()

        # SAM
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt)

        # Freeze image encoder (important)
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = False

        # CLIP text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True
        # Projection to SAM prompt dim (256)
        self.proj = nn.Linear(512, 256)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, images, texts):
        device = images.device
        B = images.shape[0]

        # -----------------------------
        # 1. Image encoder (SAM)
        # -----------------------------
        image_embeddings = self.sam.image_encoder(images)  # (B, C, 64, 64)

        # -----------------------------
        # 2. Text encoding (CLIP)
        # -----------------------------
        tokens = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt"
        ).to(device)
        text_features = self.text_encoder(**tokens).pooler_output  # (B, 512)
        text_emb = self.proj(text_features)
        text_emb = F.normalize(text_emb, dim=-1)
        text_emb = 0.05 * text_emb                       # (B, 256)
        text_emb = text_emb.unsqueeze(1)                           # (B, 1, 256)

        # -----------------------------
        # 3. Get SAM prompt embeddings (base for a single image, will be used per image in loop)
        # -----------------------------
        # These are initially (1, 0, 256) for sparse and (1, 256, 64, 64) for dense
        initial_sparse_embeddings, initial_dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None
        )

        # The image positional encoding is also (1, 256, 64, 64)
        image_pe = self.sam.prompt_encoder.get_dense_pe() # (1, 256, 64, 64)

        # List to store results for each image in the batch
        batch_low_res_masks = []

        # Process each image in the batch individually
        for i in range(B):
            # Prepare sparse embeddings for the current image: expand base sparse and add text_emb
            current_sparse_embeddings = torch.cat(
                [initial_sparse_embeddings, text_emb[i].unsqueeze(0)],
                dim=1
            ) # Resulting shape: (1, 1, 256)

            # Dense embeddings for the current image: use the initial dense embeddings
            current_dense_embeddings = initial_dense_embeddings # (1, 256, 64, 64)

            # Mask decoder expects single-image features and single-prompt features
            low_res_mask_i, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),  # (1, C, 64, 64)
                image_pe=image_pe,                                  # (1, 256, 64, 64)
                sparse_prompt_embeddings=current_sparse_embeddings, # (1, 1, 256)
                dense_prompt_embeddings=current_dense_embeddings,   # (1, 256, 64, 64)
                multimask_output=False
            )
            batch_low_res_masks.append(low_res_mask_i)
        low_res_masks = torch.cat(batch_low_res_masks, dim=0) # (B, 1, 256, 256)

        # -----------------------------
        # 6. Upsample to original size
        # -----------------------------
        masks = F.interpolate(
            low_res_masks,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False
        )

        return masks

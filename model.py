import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch import nn
from torch.optim.lr_scheduler import CyclicLR, ConstantLR, SequentialLR, LinearLR
from lightly.models import utils
from lightly.models.modules import masked_autoencoder
from lightly.transforms.mae_transform import MAETransform
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from block import Block, FeedForward
from timm.models.layers import trunc_normal_
from einops import rearrange
import torch.nn.functional as F
import torchmetrics

class MAEEncoder(nn.Module):
    def __init__(self, 
                 backbone: nn.Module, 
                 masking_rate: float = 0.75, 
                 freeze_projection: bool = True,
                 freeze_embeddings: bool = True, 
                 decoder_dim: int = 1024):
        
        super().__init__()

        self.decoder_dim = decoder_dim
        self.mask_ratio = masking_rate
        self.patch_size = backbone.patch_size
        self.sequence_length = backbone.seq_length
        self.mask_token = nn.Parameter(torch.full([1, 1, decoder_dim], -1.0))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(backbone)
        
        self.freeze_projection = freeze_projection
        self.freeze_embeddings = freeze_embeddings
        
        # freeze conv projection and positional embedding layers
        if self.freeze_projection:
            for param in list(self.backbone.parameters())[:3]:
                param.requires_grad = False
                
        if self.freeze_embeddings:
            list(self.backbone.parameters())[3].requires_grad = False
        
        
        
    def forward_encoder(self, images):
        return self.backbone.encode(images)

    def forward(self, images):
        x_encoded = self.forward_encoder(images)
        return x_encoded

class DecoderLinear(nn.Module):
    def __init__(self,n_cls = 3,patch_size = 16 ,d_encoder = 768,n_layers = 1,n_heads = 6,d_model = 384,d_ff = 768*4,drop_path_rate = 0,dropout = 0):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

class CustomDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.images = sorted(os.listdir(image_folder))
        self.masks = sorted(os.listdir(mask_folder))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.images[idx])
        mask_name = os.path.join(self.mask_folder, self.masks[idx]) 

        image = Image.open(img_name) 
        # image = add_margin(image,168,0,168,0,(0,0,0))
        mask = np.load(mask_name)  # Assuming mask is saved as numpy array
        # mask = np.pad(mask, ((168,168),(0,0)), 'constant', constant_values=0)
        mask = torch.from_numpy(mask.astype(np.float32))

        if self.transform:
            image = self.transform(image)

        return image, mask

vit_model="ViT_B_16"
backbone = torchvision.models.get_model(vit_model)
backbone.load_state_dict(torch.load("/media/umic/my_label/stranger_sections_2/stranger-sections-2-starter-notebook/checkpoints/ViT_B_16_stranger_sections2_1360x1360.pth"))
# backbone.load_state_dict(torch.load("/media/umic/my_label/stranger_sections_2/stranger-sections-2-starter-notebook/decoder_checkpoints/encoder_weights_fine_V2.pth"))


masking_rate=0
decoder_dim = 256
freeze_embeddings= False
freeze_projection = False
model = MAEEncoder(backbone, 
                masking_rate=masking_rate, 
                decoder_dim=decoder_dim, 
                freeze_embeddings=freeze_embeddings,
                freeze_projection=freeze_projection)
decoder = DecoderLinear()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
decoder.to(device)
decoder.load_state_dict(torch.load("/media/umic/my_label/stranger_sections_2/stranger-sections-2-starter-notebook/decoder_checkpoints/decoder_weights_V2_epoch=200_loss=1489.98474.pth"))

transform_kwargs={
        "input_size": (1360,1360),
        "min_scale": 1,
        "normalize": False,
    }

optimizer_kwargs={
        "lr": 0.00011
    }

transform = MAETransform(**transform_kwargs)
# transform = transforms.Compose([transforms.ColorJitter(),transforms.ToTensor()])
# dataset = torchvision.datasets.ImageFolder(root=dataset, transform=transform)
dataset = CustomDataset(image_folder='/media/umic/my_label/stranger_sections_2/augmented_images',mask_folder='/media/umic/my_label/stranger_sections_2/augmented_masks',transform=transform)


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    drop_last=True,
    num_workers=8,
)
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
warmup_epochs = 10
linear_schedule = True
cyclic_schedule = False
cyclic_step_size = 1000
start_factor = 0.2
end_factor = 0.5
n_epochs = 50
def get_schedulers(optimizer):
        base_lr = optimizer_kwargs.get('lr')
        warmup = warmup_epochs > 0  
        if linear_schedule:
            linear_scheduler = LinearLR(optimizer, start_factor=1.0, total_iters=n_epochs, end_factor=end_factor)

        else:
            # placeholder
            linear_scheduler = ConstantLR(optimizer, factor=1.0)
        
        cyclic_scheduler = None
        if cyclic_schedule:
            cyclic_scheduler = CyclicLR(optimizer, base_lr=base_lr/4.0, max_lr=base_lr,
                                        step_size_up=cyclic_step_size, mode='exp_range', gamma=0.99994)

        if warmup:
            warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
            sequential_scheduler = SequentialLR(optimizer, [warmup_scheduler, linear_scheduler], milestones=[warmup_epochs])
            return sequential_scheduler, cyclic_scheduler
        
        return linear_scheduler, cyclic_scheduler

optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
total_epochs = n_epochs if warmup_epochs < 1 else n_epochs + warmup_epochs
main_scheduler, cyclic_scheduler = get_schedulers(optimizer)


# for param in model.parameters():
#     param.requires_grad = False
min_loss = 0

print('Entering Training Loop')
for epoch in range(total_epochs):
        total_loss = .0
        
            
        for batch in dataloader:
            views = batch[0]
            images = views[0].to(device)

            views = batch[1]
            masks = views.to(device)
            # print(f'Masks shape: {masks.shape}')
            # print(f'Image size: {images.shape}')

            encoded = model(images)
            #print(f'Encoder Output: {encoded.shape}')
            encoded = encoded[:,1:]
            #print(f'After Slicing encoder output: {encoded.shape}')
            output = decoder(encoded, (1360, 1360))
            output = F.interpolate(output, size=(1360, 1360), mode="bilinear")
            output = output[:,0,:,:] + 2*output[:,1,:,:] + 3*output[:,2,:,:]
            loss = criterion(output,masks)
            total_loss += loss.detach()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e5)
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        current_lr = main_scheduler.get_last_lr()[-1]
        print(f"epoch: {epoch:>03}, loss: {avg_loss:.5f}, base_lr: {current_lr:.7f}")
        if epoch == 0:
            min_loss = avg_loss
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(decoder.state_dict(), '/media/umic/my_label/stranger_sections_2/stranger-sections-2-starter-notebook/decoder_checkpoints/decoder_weights_V3_Min_Loss.pth')
        main_scheduler.step()
print("Training Completed")
#print(output)
torch.save(decoder.state_dict(), '/media/umic/my_label/stranger_sections_2/stranger-sections-2-starter-notebook/decoder_checkpoints/decoder_weights_V3.pth')
torch.save(model.state_dict(), '/media/umic/my_label/stranger_sections_2/stranger-sections-2-starter-notebook/decoder_checkpoints/encoder_weights_fine_V3.pth')



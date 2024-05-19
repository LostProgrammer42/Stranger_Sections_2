import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import CyclicLR, ConstantLR, SequentialLR, LinearLR
import os

def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
        
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x

def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.

    Args:
        x: input array
        y: array with desired shape.

    """
    assert (
        y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    ), "Ensure that y dimensions are smaller than x dimensions!"

    x_shape = x.size()
    y_shape = y.size()
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)



class TFSamepaddingLayer(nn.Module):
    """To align with tf `same` padding. 
    
    Putting this before any conv layer that need padding
    Assuming kernel has Height == Width for simplicity
    """

    def __init__(self, ksize, stride):
        super(TFSamepaddingLayer, self).__init__()
        self.ksize = ksize
        self.stride = stride

    def forward(self, x):
        if x.shape[2] % self.stride == 0:
            pad = max(self.ksize - self.stride, 0)
        else:
            pad = max(self.ksize - (x.shape[2] % self.stride), 0)

        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)
        # print(x.shape, padding)
        x = F.pad(x, padding, "constant", 0)
        # print(x.shape)
        return x

class ResidualBlock(nn.Module):
    """Residual block as defined in:

    He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning 
    for image recognition." In Proceedings of the IEEE conference on computer vision 
    and pattern recognition, pp. 770-778. 2016.

    """

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, stride=1):
        super(ResidualBlock, self).__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            unit_layer = [
                ("preact/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                ("preact/relu", nn.ReLU(inplace=True)),
                (
                    "conv1",
                    nn.Conv2d(
                        unit_in_ch,
                        unit_ch[0],
                        unit_ksize[0],
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                ),
                ("conv1/bn", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                ("conv1/relu", nn.ReLU(inplace=True)),
                (
                    "conv2/pad",
                    TFSamepaddingLayer(
                        ksize=unit_ksize[1], stride=stride if idx == 0 else 1
                    ),
                ),
                (
                    "conv2",
                    nn.Conv2d(
                        unit_ch[0],
                        unit_ch[1],
                        unit_ksize[1],
                        stride=stride if idx == 0 else 1,
                        padding=0,
                        bias=False,
                    ),
                ),
                ("conv2/bn", nn.BatchNorm2d(unit_ch[1], eps=1e-5)),
                ("conv2/relu", nn.ReLU(inplace=True)),
                (
                    "conv3",
                    nn.Conv2d(
                        unit_ch[1],
                        unit_ch[2],
                        unit_ksize[2],
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                ),
            ]
            # * has bna to conclude each previous block so
            # * must not put preact for the first unit of this block
            unit_layer = unit_layer if idx != 0 else unit_layer[2:]
            self.units.append(nn.Sequential(OrderedDict(unit_layer)))
            unit_in_ch = unit_ch[-1]

        if in_ch != unit_ch[-1] or stride != 1:
            self.shortcut = nn.Conv2d(in_ch, unit_ch[-1], 1, stride=stride, bias=False)
        else:
            self.shortcut = None

        self.blk_bna = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )
    def forward(self, prev_feat, freeze=False):
        if self.shortcut is None:
            shortcut = prev_feat
        else:
            shortcut = self.shortcut(prev_feat)

        for idx in range(0, len(self.units)):
            new_feat = prev_feat
            new_feat = self.units[idx](new_feat)
            prev_feat = new_feat + shortcut
            shortcut = prev_feat
        feat = self.blk_bna(prev_feat)
        return feat

class DenseBlock(nn.Module):
    """Dense Block as defined in:

    Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. 
    "Densely connected convolutional networks." In Proceedings of the IEEE conference 
    on computer vision and pattern recognition, pp. 4700-4708. 2017.

    Only performs `valid` convolution.

    """

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, split=1):
        super(DenseBlock, self).__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("preact_bna/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                            ("preact_bna/relu", nn.ReLU(inplace=True)),
                            (
                                "conv1",
                                nn.Conv2d(
                                    unit_in_ch,
                                    unit_ch[0],
                                    unit_ksize[0],
                                    stride=1,
                                    padding=2,
                                    bias=False,
                                ),
                            ),
                            ("conv1/bn", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                            ("conv1/relu", nn.ReLU(inplace=True)),
                            # ('conv2/pool', TFSamepaddingLayer(ksize=unit_ksize[1], stride=1)),
                            (
                                "conv2",
                                nn.Conv2d(
                                    unit_ch[0],
                                    unit_ch[1],
                                    unit_ksize[1],
                                    groups=split,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                ),
                            ),
                        ]
                    )
                )
            )
            unit_in_ch += unit_ch[1]

        self.blk_bna = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

    def out_ch(self):
        return self.in_ch + self.nr_unit * self.unit_ch[-1]

    def forward(self, prev_feat):
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            # prev_feat = crop_to_shape(prev_feat, new_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        prev_feat = self.blk_bna(prev_feat)
        return prev_feat
class UpSample2x(nn.Module):
    """Upsample input by a factor of 2.
    
    Assume input is of NCHW, port FixedUnpooling from TensorPack.
    """

    def __init__(self):
        super(UpSample2x, self).__init__()
        # correct way to create constant within module
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self, x):
        input_shape = list(x.shape)
        # unsqueeze is expand_dims equivalent
        # permute is transpose equivalent
        # view is reshape equivalent
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret
    

class HoVerNet(nn.Module):
    def __init__(self):
        super().__init__()        
        self.conv_1 = nn.Conv2d(3,64,7,stride = 1, padding = 3, bias = False)
        self.bn_1 = nn.BatchNorm2d(64, eps=1e-5)
        self.relu_1 = nn.ReLU(inplace=True)

        self.rb_1 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.rb_2 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.rb_3 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.rb_4 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)

        self.conv_2 = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        self.upsample2x = UpSample2x()
        self.conv_3 =  nn.Conv2d(1024, 256, 5, stride=1, padding=2, bias=False)
        self.db_1 = DenseBlock(256, [1, 5], [128, 32], 8, split=4)
        self.conv_4 = nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False)

        self.conv_5 = nn.Conv2d(512, 128, 5, stride=1, padding=2, bias=False)
        self.db_2 = DenseBlock(128, [1, 5], [128, 32], 4, split=4)
        self.conv_6 = nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)

        self.conv_7 = nn.Conv2d(256, 64, 5, stride=1, padding=2, bias=False)
        self.conv_8 = nn.Conv2d(64, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, imgs):
        # imgs = imgs / 255.0
        x = self.relu_1(self.bn_1(self.conv_1(imgs)))   #d0
        x = self.rb_1(x) #d0
        d0 = x
        x = self.rb_2(x) #d1
        d1 = x
        x = self.rb_3(x) #d2
        d2 = x
        x = self.rb_4(x) #d3
        x =self.conv_2(x) #d3
        
        x = self.upsample2x(x) + d2
        x = self.conv_3(x)
        x = self.db_1(x)
        x = self.conv_4(x)
        x = self.upsample2x(x) + d1
        x = self.conv_5(x)
        x = self.db_2(x)
        x = self.conv_6(x)
        x = self.upsample2x(x) + d0
        x = self.conv_7(x)
        x = self.conv_8(x)

        return x
model = HoVerNet()

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
        mask = np.load(mask_name)  # Assuming mask is saved as numpy array
        mask = torch.from_numpy(mask.astype(np.float32))

        if self.transform:
            image = self.transform(image)

        return image, mask
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = CustomDataset(image_folder='/media/umic/my_label/Stranger_Sections_Working_Directory/Augmented_Images',mask_folder='/media/umic/my_label/Stranger_Sections_Working_Directory/Augmented_Masks',transform=transforms.ToTensor())

optimizer_kwargs={
        "lr": 0.00011
    }
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=True,
    num_workers=8,
)

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
for epoch in range(1):
        total_loss = .0
        
            
        for batch in dataloader:
            views = batch[0]
            images = views.to(device)

            views = batch[1]
            masks = views.to(device)
            print(f'Masks shape: {masks.shape}')
            print(f'Image size: {images.shape}')

            output = model(images)
            print(f'Encoder Output: {output.shape}')
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
            torch.save(model.state_dict(),'/media/umic/my_label/Stranger_Sections_Working_Directory/HoverNet_CheckPoints/HoverNet_V1_Min_Loss')
        main_scheduler.step()
print("Training Completed")
print(f'Min Loss = {min_loss}')

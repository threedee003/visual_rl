import math
import torch
import torch.nn as nn
from torchvision import models
from models.spatial_softmax import SpatialSoftmax


def _get_norm_(norm: str):
    if norm == "batch_norm":
        return nn.BatchNorm2d
    elif norm == "group_norm":
        num_groups = 16
        return lambda num_channels: nn.GroupNorm(num_groups, num_channels)
    else:
        raise NotImplementedError("We have only 'batch_norm' and 'group_norm'")
    



def _construct_resnet_(size: int, norm: nn.Module, weights = None):
    if size == 18:
        w = models.ResNet18_Weights
        m = models.resnet18(norm_layer=norm)
    elif size == 34:
        w = models.ResNet34_Weights
        m = models.resnet34(norm_layer=norm)
    else:
        raise NotImplementedError("We do not suppor resnet higher than 34 for these exps")
    
    if weights is not None:
        w = w.verify(weights).get_state_dict(progress = True)
        if norm is not nn.BatchNorm2d:
            w = {
                k: v for k, v in w.items()
                if "running_mean" not in k and "running_var" not in k
            }
        m.load_state_dict(w)
    return m


class ResNet(nn.Module):
    def __init__(self, 
                 size: int = 18, 
                 norm: str = "group_norm",
                 img_size: int = 224,
                 feature_dim: int = 64
    ) -> None:
        super(ResNet, self).__init__()
        norm_layer = _get_norm_(norm)
        model = _construct_resnet_(size, norm_layer)
        self.resnet = nn.Sequential(*(list(model.children())[:-2]))
        res_out_dim = int(math.ceil(img_size / 32.))
        resnet_out_shape = [512, res_out_dim, res_out_dim]
        self.sp_soft = SpatialSoftmax(input_shape=resnet_out_shape)
        pool_op_shp = self.sp_soft.output_shape(resnet_out_shape)
        self.flatten = nn.Flatten(1, -1)
        self.proj = nn.Linear(torch.prod(torch.tensor(pool_op_shp)).item(), feature_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = self.sp_soft(x)
        x = self.flatten(x)
        x = self.proj(x)
        return x
    






# def main(x):
#     norm_layer = _get_norm_(norm="group_norm")
#     resnet = _construct_resnet_(size=18, norm = norm_layer)
#     print(resnet)
#     return resnet(x)

# if __name__ == '__main__':
#     x = torch.randn(1, 3, 224, 224)
#     y = main(x)
#     print(y.shape)
#     # f = _get_norm_(norm="group_norm")


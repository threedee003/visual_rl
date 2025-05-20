import torch
import torch.nn as nn


class SpatialSoftmax(nn.Module):
    def __init__(self,
                 make_learnable_temp: bool = True,
                 num_key_points: int = 7,
                 temp: float = 1.,
                 input_shape: tuple = (32, 32, 32),
                 output_var: bool = False,
                 noise_std: float = 0.
    ) -> None:
        super(SpatialSoftmax, self).__init__()
        self.make_learn_temp = make_learnable_temp
        self.key_points = num_key_points
        self.output_var = output_var
        self.noise_std = noise_std
        assert len(input_shape) == 3, "Input shape should be of size 3"
        self.channels, self.height, self.width = input_shape[0], input_shape[1], input_shape[2]
        
        if num_key_points is not None:
            self.nets = nn.Conv2d(self.channels, num_key_points, kernel_size=1, stride=1)
            self.num_kp = num_key_points
        else:
            self.nets = None
            self.num_kp = self.channels
        if self.make_learn_temp:
            temperature = nn.Parameter(torch.ones(1)* temp, requires_grad=True)
            self.register_buffer('temperature', temperature)
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1., 1., self.width),
            torch.linspace(-1., 1, self.height),
            indexing='xy'
        )
        self.register_buffer('pos_x', pos_x.reshape(-1, self.width * self.height))
        self.register_buffer('pos_y', pos_y.reshape(-1, self.width * self.height))


    def output_shape(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[0] == self.channels
        return [self.num_kp, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert h == self.height and w == self.width
        if self.nets is not None:
            x = self.nets(x)
   
        
        x = x.view(-1, h*w)
      
        softmax_attn = torch.nn.functional.softmax(x / self.temperature, dim=-1)
        expec_x = torch.sum(self.pos_x * softmax_attn, dim=1, keepdim=True)
        expec_y = torch.sum(self.pos_y * softmax_attn, dim =1, keepdim=True)

        expec_kp = torch.cat([expec_x, expec_y], dim = 1)
        feature_kp = expec_kp.view(-1, self.num_kp, 2)
        if self.training: # during agent.train() mode
            noise = torch.randn_like(feature_kp) * self.noise_std
            feature_kp += noise
        if self.output_var:
            expec_xx = torch.sum(self.pos_x * self.pos_x * softmax_attn, dim = 1, keepdim=True)
            expec_yy = torch.sum(self.pos_y * self.pos_y * softmax_attn, dim = 1, keepdim=True)
            expec_xy = torch.sum(self.pos_x * self.pos_y * softmax_attn, dim = 1, keepdim=True)
            var_x = expec_xx - expec_x*expec_x
            var_y = expec_yy - expec_y*expec_y
            var_xy = expec_xy - expec_x*expec_y
            # print(var_x, var_y, var_xy)
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self.num_kp, 2, 2)
            feature_kp = (feature_kp, feature_covar)

        return feature_kp


class CNN(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 feature_dim: int  = 64
                 ) -> None:
        super(CNN, self).__init__()
        self.random = torch.randn(32, 32, 32)
        self.sp_soft = SpatialSoftmax()
        pool_op_shp = self.sp_soft.output_shape(self.random.shape)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.proj = nn.Linear(torch.prod(torch.tensor(pool_op_shp)).item(), feature_dim)

    def forward(self, x):
        x = self.sp_soft(x)
        x = self.flatten(x)
        x = self.proj(x)
        return x

        








if  __name__ == '__main__':
    # sp = SpatialSoftmax()
    # x = torch.randn(32, 32, 32)
    # y = sp(x[None])
    # print(y.shape)
    cnn = CNN()
    x = torch.randn(1, 32, 32, 32)
    y = cnn(x)
    print(y.shape)





        

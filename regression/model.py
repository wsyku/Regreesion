import torch
import torch.nn as nn
from utils import print_model_summary, create_onnx


class ConfigurableResNet3D(nn.Module):
    """
     参数量7,968,769
    """

    def __init__(
            self,
            in_channels=1,
            n_stages=None,
            features_per_stage=None,
            kernel_sizes=None,
            strides=None,
            conv_bias=False,
            norm_op=nn.BatchNorm3d,
            norm_op_kwargs=None,
            block=None,
            layers=None, **kwargs):
        super().__init__()
        if block is None:
            block = BasicBlock3D
        # n_stages优先级最高
        if n_stages is not None:
            stage_count = n_stages
        elif features_per_stage is not None:
            stage_count = len(features_per_stage)
        else:
            stage_count = 4
        if features_per_stage is None:
            features_per_stage = [64 * (2 ** i) for i in range(stage_count)]
        if kernel_sizes is None:
            kernel_sizes = [(3, 3, 3)] * stage_count
        if strides is None:
            strides = [1] + [2] * (stage_count - 1)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if layers is None:
            layers = [2] * stage_count  # 默认每stage 2个block
        self.in_planes = features_per_stage[0]
        self.conv1 = nn.Conv3d(in_channels, features_per_stage[0], kernel_size=7, stride=2, padding=3, bias=conv_bias)
        self.bn1 = norm_op(features_per_stage[0], **norm_op_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.stages = nn.ModuleList()
        in_planes = features_per_stage[0]
        for i, (planes, num_blocks, ksize, stride) in enumerate(zip(features_per_stage, layers, kernel_sizes, strides)):
            stage = self._make_layer(
                block,
                in_planes,
                planes,
                num_blocks,
                kernel_size=ksize,
                stride=stride,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                conv_bias=conv_bias
            )
            self.stages.append(stage)
            in_planes = planes
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_out = nn.Linear(features_per_stage[-1], 1)
        # self.score_max = 108  # 最大分数

    def _make_layer(self, block, in_planes, planes, blocks, kernel_size, stride, norm_op, norm_op_kwargs, conv_bias):
        # 保证stride为三元组
        if isinstance(stride, int):
            stride_tuple = (stride, stride, stride)
        elif isinstance(stride, (list, tuple)):
            if len(stride) == 3:
                stride_tuple = tuple(int(s) for s in stride)
            else:
                raise ValueError("stride must be int or tuple/list of 3 ints")
        else:
            raise ValueError("stride must be int or tuple/list of 3 ints")
        downsample = None
        if stride_tuple != (1, 1, 1) or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride_tuple, bias=conv_bias),
                norm_op(planes, **norm_op_kwargs),
            )
        layers = []
        layers.append(
            block(in_planes, planes, stride_tuple, downsample, kernel_size, norm_op, norm_op_kwargs, conv_bias))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, (1, 1, 1), None, kernel_size, norm_op, norm_op_kwargs, conv_bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_out(x)
        # x = torch.sigmoid(x) * self.score_max
        x = torch.sigmoid(x)
        return x


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=(1, 1, 1), downsample=None, kernel_size=(3, 3, 3),
                 norm_op=nn.BatchNorm3d,
                 norm_op_kwargs=None, conv_bias=False):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        # 保证kernel_size为tuple[int, int, int]
        kernel_size = tuple(int(k) for k in kernel_size)
        if len(kernel_size) != 3:
            raise ValueError("kernel_size must be a tuple of 3 ints")
        # 保证stride为tuple[int, int, int]
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        elif isinstance(stride, (list, tuple)):
            if len(stride) == 3:
                stride = tuple(int(s) for s in stride)
            else:
                raise ValueError("stride must be int or tuple/list of 3 ints")
        else:
            raise ValueError("stride must be int or tuple/list of 3 ints")
        k0, k1, k2 = kernel_size
        s0, s1, s2 = stride
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(k0, k1, k2), stride=(s0, s1, s2),
                               padding=(k0 // 2, k1 // 2, k2 // 2), bias=conv_bias)
        self.bn1 = norm_op(planes, **norm_op_kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(k0, k1, k2), stride=(1, 1, 1),
                               padding=(k0 // 2, k1 // 2, k2 // 2), bias=conv_bias)
        self.bn2 = norm_op(planes, **norm_op_kwargs)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block=BasicBlock3D, layers=None, in_channels=1):
        if layers is None:
            layers = [2, 2, 2, 2]
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MLP(nn.Module):
    """
    mask输入友好型MLP，支持LayerNorm/BatchNorm1d、可选残差、激活和Dropout灵活配置
    """

    def __init__(self, input_shape, hidden_dims=[256, 128, 64], out_dim=1, activation="ReLU", dropout=False,
                 final_activation=nn.Sigmoid, normalization=None, use_residual=False, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        self.flatten = nn.Flatten()
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        dims = [input_dim] + hidden_dims
        layers = []
        self.use_residual = use_residual
        self.normalization = getattr(nn, normalization) if normalization else nn.Identity()
        self.activation = getattr(nn, activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(self.normalization(dims[i + 1]))
            layers.append(self.activation(
                inplace=True) if 'inplace' in self.activation.__init__.__code__.co_varnames else self.activation())
            layers.append(self.dropout)

        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        layers.append(final_activation()) if final_activation else None

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, D, H, W) or (B, D, H, W) or (B, N)
        if x.ndim == 5:
            x = x.squeeze(1)
        x = x.float()  # 确保mask为float
        x = self.flatten(x)
        if self.use_residual and len(self.mlp) > 3:
            out = x
            for i, layer in enumerate(self.mlp):
                out_new = layer(out)
                # 只在Linear+Norm+Act+Dropout后加残差
                if self.norm_type and isinstance(layer, (nn.LayerNorm, nn.BatchNorm1d)) and i + 2 < len(self.mlp):
                    if out_new.shape == out.shape:
                        out = out + out_new
                    else:
                        out = out_new
                else:
                    out = out_new
            x = out
        else:
            x = self.mlp(x)
        return x


if __name__ == "__main__":
    model = MLP(input_shape=(16, 648, 256))
    model.to("cuda")
    x = torch.randn(2, 1, 16, 648, 256).to("cuda")
    y = model(x)
    print("Output shape:", y.shape)
    print_model_summary(model, input_size=(1, 16, 648, 256))

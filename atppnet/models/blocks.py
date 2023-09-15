# @brief:     Pytorch module for ConvLSTMcell
# @author     Kaustab Pal    [kaustab21@gmail.com]

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time

class CustomConv2d(nn.Module):
    """Custom 3D Convolution that enables circular padding along the width dimension only"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        bias=False,
        circular_padding=False,
    ):
        """Init custom 3D Conv with circular padding"""
        super().__init__()
        self.circular_padding = circular_padding
        self.padding = padding

        if self.circular_padding:
            # Only apply zero padding in time and height
            zero_padding = (self.padding[0], 0)
        else:
            zero_padding = padding

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=zero_padding,
            padding_mode="zeros",
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        """Forward custom 3D convolution

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor
        """
        if self.circular_padding:
            x = F.pad(
                x, (self.padding[1], self.padding[1],0, 0), mode="circular"
            )
        x = self.conv(x)
        return x

class CustomConv3d(nn.Module):
    """Custom 3D Convolution that enables circular padding along the width dimension only"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        bias=False,
        circular_padding=False,
    ):
        """Init custom 3D Conv with circular padding"""
        super().__init__()
        self.circular_padding = circular_padding
        self.padding = padding

        if self.circular_padding:
            # Only apply zero padding in time and height
            zero_padding = (self.padding[0], self.padding[1], 0)
        else:
            zero_padding = padding

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=zero_padding,
            padding_mode="zeros",
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        """Forward custom 3D convolution

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor
        """
        if self.circular_padding:
            x = F.pad(
                x, (self.padding[2], self.padding[2], 0, 0, 0, 0), mode="circular"
            )
        x = self.conv(x)
        return x
class ConvLSTMCell(nn.Module):
    '''
    Conv-LSTM paper: https://arxiv.org/pdf/1506.04214.pdf
    '''
    def __init__(self, input_dim, hidden_dim, kernel_size, padding,
            activation, frame_size, peep=True):
        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            # Using ReLU6 as ReLU as exploding grradient problems
            self.activation = nn.ReLU6() 

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.peep = peep

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim, kernel_size=kernel_size,
            padding=self.padding, bias = True
            )
        if(self.peep == True):
            # Init weights for Peep-hole connections to previous cell state
            self.W_ci = nn.Parameter(torch.randn(hidden_dim, *frame_size))
            self.W_co = nn.Parameter(torch.randn(hidden_dim, *frame_size))
            self.W_cf = nn.Parameter(torch.randn(hidden_dim, *frame_size))

    def forward(self, input_tensor, cur_state): 
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        assert not torch.any(torch.isnan(combined))

        combined_conv = self.conv(combined)
        assert not torch.any(torch.isnan(combined_conv))

        i_conv, f_conv, c_conv, o_conv =\
                torch.chunk(combined_conv, chunks=4, dim=1)

        if(self.peep == True):
            input_gate = torch.sigmoid(i_conv + self.W_ci * c_cur )
            forget_gate = torch.sigmoid(f_conv + self.W_cf * c_cur )
            # Current Cell output
            c_next = forget_gate*c_cur + input_gate * self.activation(c_conv)
            output_gate = torch.sigmoid(o_conv + self.W_co * c_next )
        else:
            input_gate = torch.sigmoid(i_conv)
            forget_gate = torch.sigmoid(f_conv)
            # Current Cell output
            c_next = forget_gate*c_cur + input_gate * self.activation(c_conv)
            output_gate = torch.sigmoid(o_conv)

        # Current Hidden State
        h_next = output_gate * self.activation(c_next)

        assert not torch.any(torch.isnan(input_gate))
        assert not torch.any(torch.isnan(forget_gate))
        assert not torch.any(torch.isnan(c_next))
        assert not torch.any(torch.isnan(output_gate))
        assert not torch.any(torch.isnan(h_next))

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding,
            activation, frame_size, num_layers=1,
            peep=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        padding = self._extend_for_multilayer(padding, num_layers)
        frame_size = self._extend_for_multilayer(frame_size, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.frame_size = frame_size
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers
        self.peep = peep
        self.activation = nn.ReLU6()

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim =\
                    self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          padding=self.padding[i],
                                          activation=activation,
                                          frame_size=self.frame_size[i],
                                          peep=self.peep))

        self.cell_list = nn.ModuleList(cell_list)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input_tensor, hidden_state=None):
        # X is a frame sequence (batch_size, seq_len, num_channels, height, width)
        # Get the dimensions
        b, seq_len, _, h, w = input_tensor.size()

        if hidden_state is None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                inp = self.activation(cur_layer_input[:, t, :, :, :])
                h, c = self.cell_list[layer_idx](
                    input_tensor = self.dropout(inp),\
                            cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(
                in_channels=in_features, out_channels=in_features,
                kernel_size=1, padding=0, bias=True)
        self.W_g = nn.Conv2d(
                in_channels=in_features, out_channels=in_features,
                kernel_size=1, padding=0, bias=True)

        self.gate_c = nn.Sequential() # channel gate
        self.gate_c.add_module('gate_c_fc_reduce0',torch.nn.Conv1d(2*in_features, in_features,1))
        self.gate_c.add_module('gate_c_in_reduce0',
                torch.nn.InstanceNorm2d(in_features)
                )
        self.gate_c.add_module('gate_c_relu_reduce0', torch.nn.ReLU())

        self.gate_s = nn.Sequential() # spatial gate
        self.gate_s.add_module('gate_s_conv_reduce0',
                nn.Conv2d(in_channels=2*in_features, out_channels=2*in_features//8,
                kernel_size=1, padding=0, bias=True)
                )
        self.gate_s.add_module('gate_s_in_reduce0',
                torch.nn.InstanceNorm2d(in_features//8)
                )
        self.gate_s.add_module('gate_s_relu_reduce0', torch.nn.ReLU())

        for i in range(1):
            self.gate_s.add_module('gate_s_conv_reduce_di_%d'%i,
                    nn.Conv2d(in_channels=2*in_features//8, out_channels=2*in_features//8,
                    kernel_size=3, padding=4, dilation=4, bias=True)
                    )
            self.gate_s.add_module('gate_s_bn_reduce_di_%d'%i,
                    torch.nn.InstanceNorm2d(2*in_features//8)
                    )
            self.gate_s.add_module('gate_s_relu_reduce_di_%d'%i, torch.nn.ReLU())

        self.gate_s.add_module('gate_s_conv_final',
                nn.Conv2d(in_channels=2*in_features//8, out_channels=1,
                kernel_size=1, bias=True)
                )

    def forward(self, output, g):
        batch, seq_len, chn, H_out, W_out = output.shape
        c = torch.zeros((batch, chn, H_out, W_out),
                device=output.device)
        a_maps = []
        for i in range(output.shape[1]):
            l = output[:,i]
            B, C, H, W = l.size()
            l_ = self.W_l(l)
            g_ = self.W_g(g)
            cat_f = F.relu(torch.cat((l_, g_),1))
            avg_f = F.avg_pool2d(cat_f, (H,W)).view(B,2*C,1)
            avg_f = self.gate_c(avg_f).view(B,C,1,1)
            c +=  l*torch.sigmoid(avg_f * self.gate_s(cat_f)) #self.op(torch.cat((l,g),1)) # batch_sizex1xHxW
        return c, a_maps

class Normalization(nn.Module):
    """Custom Normalization layer to enable different normalization strategies"""

    def __init__(self, cfg, n_channels):
        """Init custom normalization layer"""
        super(Normalization, self).__init__()
        self.cfg = cfg
        self.norm_type = self.cfg["MODEL"]["NORM"]
        n_channels_per_group = self.cfg["MODEL"]["N_CHANNELS_PER_GROUP"]

        if self.norm_type == "batch":
            self.norm = nn.BatchNorm2d(n_channels)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(n_channels // n_channels_per_group, n_channels)
        elif self.norm_type == "instance":
            self.norm = nn.InstanceNorm3d(n_channels)
        elif self.norm_type == "none":
            self.norm = nn.Identity()

    def forward(self, x):
        """Forward normalization pass

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor
        """
        x = self.norm(x)
        return x

class Normalization3D(nn.Module):
    """Custom Normalization layer to enable different normalization strategies"""

    def __init__(self, cfg, n_channels):
        """Init custom normalization layer"""
        super(Normalization3D, self).__init__()
        self.cfg = cfg
        self.norm_type = self.cfg["MODEL"]["NORM"]
        n_channels_per_group = self.cfg["MODEL"]["N_CHANNELS_PER_GROUP"]

        if self.norm_type == "batch":
            self.norm = nn.BatchNorm3d(n_channels)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(n_channels // n_channels_per_group, n_channels)
        elif self.norm_type == "instance":
            self.norm = nn.InstanceNorm3d(n_channels)
        elif self.norm_type == "none":
            self.norm = nn.Identity()

    def forward(self, x):
        """Forward normalization pass

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor
        """
        x = self.norm(x)
        return x

class DownBlock(nn.Module):
    """Downsamples the input tensor"""

    def __init__(
        self, cfg, in_channels, out_channels, kernel_size=(2,4), skip=False
    ):
        """Init module"""
        super(DownBlock, self).__init__()
        self.skip = skip
        self.circular_padding = cfg["MODEL"]["CIRCULAR_PADDING"]
        self.conv0 = CustomConv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
            circular_padding=self.circular_padding,
        )
        self.norm0 = Normalization(cfg, in_channels)
        self.relu = nn.LeakyReLU()
        self.conv1 = CustomConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
            circular_padding=self.circular_padding,
        )
        self.norm1 = Normalization(cfg, out_channels)

    def forward(self, x):
        """Forward pass for downsampling

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Downsampled output tensor
        """
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        return x

class DownBlock3D(nn.Module):
    """Downsamples the input tensor"""

    def __init__(
        self, cfg, in_channels, out_channels, temporal_kernel_size, skip=False
    ):
        """Init module"""
        super(DownBlock3D, self).__init__()
        self.skip = skip
        self.circular_padding = cfg["MODEL"]["CIRCULAR_PADDING"]
        self.conv0 = CustomConv3d(
            in_channels,
            in_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False,
            circular_padding=self.circular_padding,
        )
        self.norm0 = Normalization3D(cfg, in_channels)
        self.relu = nn.LeakyReLU()
        self.conv1 = CustomConv3d(
            in_channels,
            out_channels,
            kernel_size=(temporal_kernel_size, 2, 4),
            stride=(1, 2, 4),
            bias=False,
            circular_padding=self.circular_padding,
        )
        self.norm1 = Normalization3D(cfg, out_channels)

    def forward(self, x):
        """Forward pass for downsampling

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Downsampled output tensor
        """
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        return x

class UpBlock(nn.Module):
    """Upsamples the input tensor using transposed convolutions"""

    def __init__(
        self, cfg, in_channels, out_channels, skip=False
    ):
        """Init module"""
        super(UpBlock, self).__init__()
        self.skip = skip
        self.circular_padding = cfg["MODEL"]["CIRCULAR_PADDING"]
        if self.skip:
            self.conv_skip = CustomConv2d(
                2 * in_channels,
                in_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
                circular_padding=self.circular_padding,
            )
            self.norm_skip = Normalization(cfg, in_channels)
        self.conv0 = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=(2, 4),
            stride=(2, 4),
            bias=False,
        )
        self.norm0 = Normalization(cfg, in_channels)
        self.relu = nn.LeakyReLU()
        self.conv1 = CustomConv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
            circular_padding=self.circular_padding,
        )
        self.norm1 = Normalization(cfg, out_channels)

    def forward(self, x, skip=None):
        """Forward pass for upsampling

        Args:
            x (torch.tensor): Input tensor
            skip (bool, optional): Use skip connection. Defaults to None.

        Returns:
            torch.tensor: Upsampled output tensor
        """
        if self.skip:
            x = torch.cat((x, skip), dim=1)
            x = self.conv_skip(x)
            x = self.norm_skip(x)
            x = self.relu(x)
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        return x

class UpBlock3D(nn.Module):
    """Upsamples the input tensor using transposed convolutions"""

    def __init__(
        self, cfg, in_channels, out_channels, temporal_kernel_size, skip=False
    ):
        """Init module"""
        super(UpBlock3D, self).__init__()
        self.skip = skip
        self.circular_padding = cfg["MODEL"]["CIRCULAR_PADDING"]
        if self.skip:
            self.conv_skip = CustomConv3d(
                2 * in_channels,
                in_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False,
                circular_padding=self.circular_padding,
            )
            self.norm_skip = Normalization(cfg, in_channels)
        self.conv0 = nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=(temporal_kernel_size, 2, 4),
            stride=(1, 2, 4),
            bias=False,
        )
        self.norm0 = Normalization3D(cfg, in_channels)
        self.relu = nn.LeakyReLU()
        self.conv1 = CustomConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False,
            circular_padding=self.circular_padding,
        )
        self.norm1 = Normalization3D(cfg, out_channels)

    def forward(self, x, skip=None):
        """Forward pass for upsampling

        Args:
            x (torch.tensor): Input tensor
            skip (bool, optional): Use skip connection. Defaults to None.

        Returns:
            torch.tensor: Upsampled output tensor
        """
        if self.skip:
            x = torch.cat((x, skip), dim=1)
            x = self.conv_skip(x)
            x = self.norm_skip(x)
            x = self.relu(x)
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        return x

class CNN3D_block(nn.Module):
    def __init__(self, cfg):
        """Init all layers needed for range image-based point cloud prediction"""
        super(CNN3D_block, self).__init__()
        self.cfg = cfg
        self.channels_2d = self.cfg["MODEL"]["CHANNELS"]
        self.channels = self.cfg["MODEL"]["3D_CHANNELS"]
        self.skip_if_channel_size = self.cfg["MODEL"]["SKIP_IF_3D_CHANNEL_SIZE"]
        self.temporal_kernel_size = self.cfg["MODEL"]["TEMPORAL_KERNEL_SIZE"]
        self.circular_padding = self.cfg["MODEL"]["CIRCULAR_PADDING"]

        self.input_layer = CustomConv3d(
            self.channels_2d[-1],
            self.channels[0],
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            bias=True,
            circular_padding=self.circular_padding,
        )

        self.DownLayers = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            if self.channels[i + 1] in self.skip_if_channel_size:
                self.DownLayers.append(
                    DownBlock3D(
                        self.cfg,
                        self.channels[i],
                        self.channels[i + 1],
                        self.temporal_kernel_size[i],
                        skip=True,
                    )
                )
            else:
                self.DownLayers.append(
                    DownBlock3D(
                        self.cfg,
                        self.channels[i],
                        self.channels[i + 1],
                        self.temporal_kernel_size[i],
                        skip=False,
                    )
                )

        self.UpLayers = nn.ModuleList()
        for i in reversed(range(len(self.channels) - 1)):
            if self.channels[i + 1] in self.skip_if_channel_size:
                self.UpLayers.append(
                    UpBlock3D(
                        self.cfg,
                        self.channels[i + 1],
                        self.channels[i],
                        self.temporal_kernel_size[i],
                        skip=True,
                    )
                )
            else:
                self.UpLayers.append(
                    UpBlock3D(
                        self.cfg,
                        self.channels[i + 1],
                        self.channels[i],
                        self.temporal_kernel_size[i],
                        skip=False,
                    )
                )

        self.n_outputs = 2
        self.output_layer = CustomConv3d(
            self.channels[0],
            self.channels_2d[-1],
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            bias=True,
            circular_padding=self.circular_padding,
        )

    def forward(self, x):
        """Forward range image-based point cloud prediction

        Args:
            x (torch.tensor): Input tensor of concatenated, unnormalize range images

        Returns:
            dict: Containing the predicted range tensor and mask logits
        """
        # Only select inputs specified in base model
        batch_size, n_past_steps, c, H, W = x.size()

        # Get mask of valid points
        skip_list = []
        x = x.view(batch_size, c, n_past_steps, H, W)
        x = self.input_layer(x)
        for layer in self.DownLayers:
            x = layer(x)
            if layer.skip:
                skip_list.append(x.clone())

        for layer in self.UpLayers:
            if layer.skip:
                x = layer(x, skip_list.pop())
            else:
                x = layer(x)

        x = self.output_layer(x)
        x = x.view(batch_size * n_past_steps, c, H, W)
        return x

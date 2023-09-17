# @brief:     Generic pytorch module for NN
# @author     Kaustab Pal    [kaustab21@gmail.com]

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time
from atppnet.models.base import BasePredictionModel
from atppnet.models.blocks import ConvLSTM, LinearAttentionBlock
from atppnet.models.blocks import DownBlock, UpBlock, CustomConv2d, CNN3D_block
import random
import numpy as np

class ATPPNet(BasePredictionModel):
    def __init__(self, cfg, num_channels, num_kernels, kernel_size, padding, 
    activation, img_size, num_layers, peep=True):
        super(ATPPNet, self).__init__(cfg)
        self.channels = self.cfg["MODEL"]["CHANNELS"]
        self.skip_if_channel_size = self.cfg["MODEL"]["SKIP_IF_CHANNEL_SIZE"]
        self.num_kernels = self.channels[-1]
        self.img_size = img_size
        frame_h = self.img_size[0]//2**(len(self.channels)-1)
        frame_w = self.img_size[1]//2**(len(self.channels)-1)
        self.frame_size = (frame_h, frame_w)
        self.CNN3D_block = CNN3D_block(cfg = self.cfg)

        self.conv_skip = CustomConv2d(
            2 * self.channels[-1],
            self.channels[-1],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
            circular_padding=True,
        )

        self.input_layer = CustomConv2d(
            num_channels,
            self.channels[0],
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
            circular_padding=True,
        )

        self.DownLayers = nn.ModuleList()
        self.UpLayers = nn.ModuleList()

        for i in range(len(self.channels) - 1):
            if self.channels[i + 1] in self.skip_if_channel_size:
                self.DownLayers.append(
                        DownBlock(
                            cfg = self.cfg,
                            in_channels = self.channels[i],
                            out_channels = self.channels[i + 1],
                            skip=True,
                        )
                    )
            else:
                self.DownLayers.append(
                        DownBlock(
                            cfg = self.cfg,
                            in_channels = self.channels[i],
                            out_channels = self.channels[i + 1],
                            skip=False,
                        )
                    )

        for i in reversed(range(len(self.channels) - 1)):
            if self.channels[i + 1] in self.skip_if_channel_size:
                self.UpLayers.append(
                    UpBlock(
                        cfg = self.cfg,
                        in_channels = self.channels[i + 1],
                        out_channels = self.channels[i],
                        skip=True,
                    )
                )
            else:
                self.UpLayers.append(
                    UpBlock(
                        cfg = self.cfg,
                        in_channels = self.channels[i + 1],
                        out_channels = self.channels[i],
                        skip=False,
                    )
                )
        self.output_layer = CustomConv2d(
            self.channels[0],
            2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
            circular_padding=True,
        )

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attention = nn.ModuleList()
        for i in range(len(self.channels)):
            if self.channels[i] in self.skip_if_channel_size:
                self.encoder.append(
                        ConvLSTM(
                            input_dim=self.channels[i],
                            hidden_dim=self.channels[i],
                            kernel_size=kernel_size,
                            padding=padding, activation=activation,
                            frame_size=self.frame_size, num_layers=num_layers,
                            peep=peep, return_all_layers=True)
                        )
                self.attention.append(
                        LinearAttentionBlock(in_features=self.channels[i],
                            normalize_attn=True)
                        )
                self.decoder.append(
                        ConvLSTM(
                            input_dim=2*self.channels[i],
                            hidden_dim=self.channels[i],
                            kernel_size=kernel_size,
                            padding=padding, activation=activation,
                            frame_size=self.frame_size, num_layers=num_layers,
                            peep=peep, return_all_layers=True)
                        )
        self.encoder.append(
                ConvLSTM(
                    input_dim=self.channels[-1],
                    hidden_dim=self.channels[-1],
                    kernel_size=kernel_size,
                    padding=padding, activation=activation,
                    frame_size=self.frame_size, num_layers=num_layers,
                    peep=peep, return_all_layers=True)
                )
        self.attention.append(
                LinearAttentionBlock(in_features=self.channels[-1],
                    normalize_attn=True)
                )
        self.decoder.append(
                ConvLSTM(
                    input_dim=2*self.channels[-1],
                    hidden_dim=self.channels[-1],
                    kernel_size=kernel_size,
                    padding=padding, activation=activation,
                    frame_size=self.frame_size, num_layers=num_layers,
                    peep=peep, return_all_layers=True)
                )

        self.norm = nn.BatchNorm3d(num_features=self.cfg["MODEL"]["N_PAST_STEPS"])

    def forward(self, x):
        x = x[:, :, self.inputs, :, :]
        device = x.device
        batch, seq_length, num_channels, height, width = x.shape
        # if you want to normalize the input uncomment the below lines
        #past_mask = x != -1.0

        ## Standardization and set invalid points to zero
        #mean = self.mean[None, self.inputs, None, None, None]

        #std = self.std[None, self.inputs, None, None, None]
        #x = torch.true_divide(x - mean, std)
        #x = x * past_mask
        #SOS = torch.zeros((batch,num_channels, height, width), device=device)
        skip_layer_encoder = []
        prob_skip_layer_encoder = []
        skip_layer_decoder = []
        attn_list = []
        encoder_output = []
        encoder_hidden = []
        x = x.view(batch*seq_length, num_channels, height, width).to(device)
        x = self.input_layer(x)
        for l in self.DownLayers:
            x = l(x)
            if l.skip:
                skip_layer_encoder.append(x)
        skip_layer_encoder.append(x)
        _,c,h,w = x.shape
        tc_x = x.view(batch,seq_length,c,h,w)
        tc_x = self.CNN3D_block(tc_x)
        for s in range(len(skip_layer_encoder)):
            _,c,h,w = skip_layer_encoder[s].shape
            skip_layer_encoder[s] =\
                    skip_layer_encoder[s].view(batch,seq_length,c,h,w).to(device)

            output, h = self.encoder[s](skip_layer_encoder[s])
            output = output[-1]
            output = self.norm(output)
            g = output[:,-1] # final layer's hidden state
            g_out = torch.zeros((batch, self.cfg["MODEL"]["N_FUTURE_STEPS"],
                    g.shape[1], g.shape[2], g.shape[3]),device=device)
            for i in range(self.cfg["MODEL"]["N_FUTURE_STEPS"]):
                context = torch.zeros((batch, g.shape[1],
                    g.shape[2], g.shape[3]),
                    device=device) # global context vector
                context, maps = self.attention[s](output,g)
                attn_list.append(maps)
                context = torch.cat((context,g),1)
                dec_input = context.unsqueeze(1)
                dec_output, h = self.decoder[s](dec_input, h)
                g = dec_output[-1][:,-1]
                output = output[:,1:]
                output = torch.cat((output,g.unsqueeze(1)),1)
                g_out[:,i] = g
            skip_layer_decoder.append(
                    g_out.view(
                        batch*self.cfg["MODEL"]["N_FUTURE_STEPS"],
                        g_out.shape[2],g_out.shape[3],g_out.shape[4]).to(device)
                    )

        x = skip_layer_decoder.pop()
        x = torch.cat((x, tc_x), dim=1)
        x = self.conv_skip(x)
        for l in self.UpLayers:
            if l.skip:
                x = l(x, skip_layer_decoder.pop())
            else:
                x = l(x)
        x = self.output_layer(x)
        x = x.view(batch, self.cfg["MODEL"]["N_FUTURE_STEPS"], x.shape[1], x.shape[2], x.shape[3]).to(device)
        assert not torch.any(torch.isnan(x))
        output = {}
        output["rv"] = self.min_range + nn.Sigmoid()(x[:, :, 0, :, :]) * (
            self.max_range - self.min_range
        )
        output["mask_logits"] = x[:, :, 1, :, :]
        return output, attn_list

if __name__ == "__main__":
    config_filename = 'config/parameters.yml'# location of config file
    cfg = yaml.safe_load(open(config_filename))
    model = ATPPNet(cfg, num_channels=1, num_kernels=32, 
                    kernel_size=(3, 3), padding=(1, 1), activation="relu", 
                    img_size=(64, 64), num_layers=3, peep=False)
    model = model.to("cuda")
    inp_nuscenes = torch.randn(2,5,1,32,1024)
    inp_kitti = torch.randn(2,5,1,64,2048)
    inp = inp_nuscenes.to("cuda")
    inf_time = []
    for i in range(52):
        start = time.time()
        pred, _ = model(inp)
        inf_time.append((time.time()-start)/inp.shape[0])
    inf_time = inf_time[2:]
    inf_time - np.array(inf_time)
    print("Inference time (sec): ", np.mean(inf_time))











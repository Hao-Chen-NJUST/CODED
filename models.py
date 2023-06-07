import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import build_backbone
from transformer import build_transformer
import math
from util import NestedTensor, nested_tensor_from_tensor_list


class PULT(nn.Module):
    def __init__(self, backbone, transformer, num_classes=2, num_feature_levels=4, featdim=1024):
        super().__init__()

        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.num_feature_levels = num_feature_levels

        self.featdim = featdim
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])

    def forward(self, samples: NestedTensor, proto_memory=None, unlabelled=True, alpha=0.9):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        resnet_fea = []

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        for src in srcs:
            resnet_fea.append(F.adaptive_avg_pool2d(src, 1) + F.adaptive_max_pool2d(src, 1))
        # resnet_fea = torch.squeeze(torch.cat(resnet_fea, dim=1))
        resnet_fea = torch.cat(resnet_fea, dim=2).mean(dim=2).squeeze()
        if self.training:
            if unlabelled:
                proto = resnet_fea[:resnet_fea.size(0)//2].mean(dim=0, keepdim=True)
            else:
                proto = resnet_fea.mean(dim=0, keepdim=True)
            proto_memory = alpha * proto + (1-alpha) * proto_memory
            query_embeds = proto_memory

            hs = self.transformer(srcs, masks, query_embeds, pos)
            outputs_classes = []
            for lvl in range(hs.shape[0]):
                outputs_class = self.class_embed[lvl](hs[lvl])
                outputs_classes.append(outputs_class)
            outputs_class = torch.stack(outputs_classes)
        else:
            query_embeds = proto_memory

            hs = self.transformer(srcs, masks, query_embeds, pos)
            outputs_classes = []
            for lvl in range(hs[0].shape[0]):
                outputs_class = self.class_embed[lvl](hs[0][lvl])
                outputs_classes.append(outputs_class)
            outputs_class = torch.stack(outputs_classes)

        return outputs_class, proto_memory.detach(), hs, resnet_fea


def build_model(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = PULT(
        backbone=backbone,
        transformer=transformer,
        num_classes=2,
        num_feature_levels=args.num_feature_levels,
        featdim=args.featdim)

    return model

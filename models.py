import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import build_backbone
from transformer import build_transformer
import math
from util import NestedTensor, nested_tensor_from_tensor_list
import hyptorch.nn as hypnn
import hyptorch.pmath as pmath


class PULT(nn.Module):
    def __init__(self, backbone, transformer, num_classes=2, num_feature_levels=4, featdim=1024, hyp_c=None,
                 clip_r=None, k=0.1, k_learnable=True):
        super().__init__()

        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.num_feature_levels = num_feature_levels

        self.hyp_c = hyp_c
        self.e2p = hypnn.ToPoincare(c=self.hyp_c, ball_dim=hidden_dim, riemannian=False, clip_r=clip_r)
        self.p2e = hypnn.FromPoincare(c=self.hyp_c, ball_dim=hidden_dim)
        self.hyp_mlr = hypnn.HyperbolicMLR(ball_dim=hidden_dim, n_classes=num_classes, c=self.hyp_c)
        if k == 0:
            self.k = k
        else:
            self.k = torch.nn.Parameter(torch.tensor(k), requires_grad=k_learnable)
            self.k_minmax = {
                "max": 1.0 - 1e-6,
                "min": 0.0 + 1e-6,
            }

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
        resnet_fea = torch.cat(resnet_fea, dim=2).mean(dim=2).squeeze()
        if self.training:
            p_label = None
            self.k.data = torch.clamp(self.k.data, **self.k_minmax)
            if self.hyp_c > 0.0:
                resnet_fea = self.e2p(resnet_fea)

                proto = pmath.poincare_mean(resnet_fea[:resnet_fea.size(0) // 2], dim=0, c=self.e2p.c).unsqueeze(0)
                proto_memory = proto_memory_update(proto, proto_memory, alpha, c=self.hyp_c)

                if self.k is not None:
                    # p_label = (F.cosine_similarity(proto_memory,
                    #                                resnet_fea[resnet_fea.size(0) // 2:]) < self.k).squeeze(
                    #     dim=-1).long()

                    p_label = (1 / (torch.cosh(
                        pmath.dist(proto_memory, resnet_fea[resnet_fea.size(0) // 2:],
                                   c=self.hyp_c))) < self.k.data).squeeze(
                        dim=-1).long()
            else:
                proto = resnet_fea[:resnet_fea.size(0) // 2].mean(dim=0, keepdim=True)
                proto_memory = proto_memory_update(proto, proto_memory, alpha)

                if self.k is not None:
                    p_label = (F.cosine_similarity(proto_memory,
                                                   resnet_fea[resnet_fea.size(0) // 2:]) < self.k).squeeze(
                        dim=-1).long()

        if self.hyp_c > 0.0:
            query_embeds = self.p2e(proto_memory)
        else:
            query_embeds = proto_memory

        hs = self.transformer(srcs, masks, query_embeds, pos)

        if self.training:
            if self.hyp_c > 0.0:
                hs = self.e2p(hs)
                outputs_class = self.hyp_mlr(hs)
            else:
                outputs_class = self.class_embed(hs)
            return outputs_class, proto_memory.detach(), hs, resnet_fea, p_label
        else:
            if self.hyp_c > 0.0:
                hs[0] = self.e2p(hs[0])
                outputs_class = self.hyp_mlr(hs[0])
            else:
                outputs_class = self.class_embed(hs[0])
            return outputs_class, proto_memory.detach(), hs, resnet_fea


def proto_memory_update(proto, proto_memory, alpha, c=0):
    if c > 0:
        return alpha * pmath.poincare_mean(proto, dim=0, c=c).unsqueeze(0) + (1 - alpha) * pmath.poincare_mean(
            proto_memory, dim=0, c=c).unsqueeze(0)
    else:
        return alpha * proto + (1 - alpha) * proto_memory


def build_model(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = PULT(
        backbone=backbone,
        transformer=transformer,
        num_classes=2,
        num_feature_levels=args.num_feature_levels,
        featdim=args.featdim,
        hyp_c=args.hyp_c,
        clip_r=args.clip_r,
        k=args.k,
        k_learnable=args.k_learnable,
    )

    return model

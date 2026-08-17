import torch
import torch.nn as nn
import torch.nn.functional as F

from models.domain_adaption import DomainAdapter
from models.networks import ResnetGenerator, ResnetGeneratorMask


class GlassRemovalPipeline(nn.Module):
    """DA -> GM -> SM -> DS -> DG inference pipeline."""

    def __init__(self, da, gm, sm, ds, dg):
        super().__init__()
        self.da = da
        self.gm = gm
        self.sm = sm
        self.ds = ds
        self.dg = dg

    def forward(self, x):
        gf, sf = self.da(x)

        g_logits = self.gm(gf)
        gmask = g_logits.argmax(1, keepdim=True).float()

        s_logits = self.sm(torch.cat([sf, gmask], dim=1))
        smask = F.softmax(s_logits, dim=1)[:, 1:2] * 1.25

        shadow_free = self.ds(torch.cat([x, smask, gmask], dim=1))
        masked = shadow_free * (1 - gmask)
        final = self.dg(torch.cat([masked, gmask], dim=1))

        return final


def load_pipeline(checkpoint_path, device="cpu"):
    da = DomainAdapter()
    gm = ResnetGeneratorMask(64, 2)
    sm = ResnetGeneratorMask(65, 2)
    ds = ResnetGenerator(5, 3)
    dg = ResnetGenerator(4, 3)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    da.load_state_dict(ckpt["DA"])
    gm.load_state_dict(ckpt["GM"])
    sm.load_state_dict(ckpt["SM"])
    ds.load_state_dict(ckpt["DeShadow"])
    dg.load_state_dict(ckpt["DeGlass"])

    return GlassRemovalPipeline(da, gm, sm, ds, dg).to(device)

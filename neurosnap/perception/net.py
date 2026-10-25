"""GateNet: a small single-stage CenterNet-style gate detector with corner regression.

Input: 192x144 RGB. Output at stride 4 (48x36):
  heatmap (1)     - probability that a gate centre falls in the cell
  offset (2)      - sub-cell centre offset
  corners (8)     - opening corners (TL, TR, BR, BL) relative to the centre, / CORNER_S px
The four corners plus the known opening size give the full gate pose through PnP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4
CORNER_S = 64.0


def conv_bn(cin, cout, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, s, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = conv_bn(c, c)
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(x + self.b(self.a(x)))


def stage(cin, cout, n=1):
    return nn.Sequential(conv_bn(cin, cout, 2), *[ResBlock(cout) for _ in range(n)])


class GateNet(nn.Module):
    def __init__(self, w=(32, 48, 96, 160, 224), head=64):
        super().__init__()
        self.s1 = stage(3, w[0], 1)          # /2
        self.s2 = stage(w[0], w[1], 1)       # /4
        self.s3 = stage(w[1], w[2], 2)       # /8
        self.s4 = stage(w[2], w[3], 2)       # /16
        self.s5 = stage(w[3], w[4], 1)       # /32
        self.l5 = nn.Conv2d(w[4], head, 1)
        self.l4 = nn.Conv2d(w[3], head, 1)
        self.l3 = nn.Conv2d(w[2], head, 1)
        self.l2 = nn.Conv2d(w[1], head, 1)
        self.smooth = conv_bn(head, head)

        def mk(cout):
            return nn.Sequential(conv_bn(head, head), nn.Conv2d(head, cout, 1))

        self.hm = mk(1)
        self.off = mk(2)
        self.cor = mk(8)
        self.hm[-1].bias.data.fill_(-2.19)   # prior p = 0.1

    def forward(self, x):
        c1 = self.s1(x)
        c2 = self.s2(c1)
        c3 = self.s3(c2)
        c4 = self.s4(c3)
        c5 = self.s5(c4)
        p = self.l5(c5)
        p = F.interpolate(p, size=c4.shape[-2:], mode='nearest') + self.l4(c4)
        p = F.interpolate(p, size=c3.shape[-2:], mode='nearest') + self.l3(c3)
        p = F.interpolate(p, size=c2.shape[-2:], mode='nearest') + self.l2(c2)
        p = self.smooth(p)
        return self.hm(p), self.off(p), self.cor(p)


def focal_loss(logits, gt):
    """CenterNet penalty-reduced focal loss."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg
    n = pos.sum().clamp_min(1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n


def gather(feat, ind):
    """feat (B, C, H, W), ind (B, M) flat indices -> (B, M, C)."""
    B, C = feat.shape[:2]
    f = feat.view(B, C, -1).permute(0, 2, 1)
    return f.gather(1, ind[..., None].expand(-1, -1, C))


def detection_loss(out, tgt):
    hm, off, cor = out
    l_hm = focal_loss(hm, tgt['hm'])
    m = tgt['mask'][..., None].float()
    n = m.sum().clamp_min(1.0)
    l_off = (F.l1_loss(gather(off, tgt['ind']), tgt['off'], reduction='none') * m).sum() / (2 * n)
    l_cor = (F.l1_loss(gather(cor, tgt['ind']), tgt['cor'], reduction='none') * m).sum() / (8 * n)
    return l_hm + l_off + 4.0 * l_cor, dict(hm=l_hm.item(), off=l_off.item(), cor=l_cor.item())

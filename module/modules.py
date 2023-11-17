from torch import nn, Tensor
from torch.nn import functional as F

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc = nn.Sequential(
                nn.Linear(input_c, squeeze_c),
                nn.ReLU(inplace=True),
                nn.Linear(squeeze_c, input_c),
        )

    def forward(self, x: Tensor) -> Tensor:
        b,c,_,_ = x.size()
        scale = F.adaptive_avg_pool2d(x, output_size=(1,1)).view(b, c)
        scale = self.fc(scale)
        scores = F.hardsigmoid(scale, inplace=True).view(b, c, 1, 1)
        return scores
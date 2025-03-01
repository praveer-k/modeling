import torch
from torch.nn import Conv1d

def main():
    mps_device = torch.device('mps')
    x = torch.rand(1, 1, 12).to(mps_device)
    print(x)
    y = x * 10
    print(y)
    conv = Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False).to(mps_device)
    conv.weight.data.fill_(10)
    z = conv(x)
    print(z)
    print(torch.allclose(y, z, atol=1e-6))


if __name__ == "__main__":
    main()

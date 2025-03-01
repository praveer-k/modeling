import torch
from torch.nn import Conv1d

def conv_2d_example():
    device = torch.device('mps')
    x = torch.arange(1, 7, 1, dtype=torch.float32).reshape(1, 1, -1).to(device)
    print(f"Input shape: {x.shape}")
    print(f"Input: {x.squeeze()}")
    # Expected output from our mathematical calculation
    y = torch.tensor([6.5, 10.0, 13.5, 17.0]).to(device)
    print(f"Expected output: {y}")

    # Create 1D convolution layer
    conv = Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False
    ).to(device)

    # Set the kernel weights to [1, 2, 0.5]
    conv.weight.data = torch.tensor([1.0, 2.0, 0.5]).reshape(1, 1, -1).to(device)
    print(f"Kernel weights: {conv.weight.data.squeeze()}")
    # Apply convolution
    z = conv(x)
    print(f"Output shape: {z.shape}")
    print(f"Actual output: {z.squeeze()}")
    
    # Verify results match
    print(f"Outputs match: {torch.allclose(y, z.squeeze(), atol=1e-6)}")


if __name__ == "__main__":
    conv_2d_example()

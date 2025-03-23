import torch
from torch.nn import Conv2d


def conv_2d_example():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Create 6x6 input matrix
    x = torch.arange(1, 37, 1, dtype=torch.float32).reshape(1, 1, 6, 6).to(device)
    print(f"Input shape: {x.shape}")
    print(f"Input:\n{x.squeeze()}")

    # Create 3x3 kernel
    kernel = torch.tensor(
        [[1, 2, 0.5], [0.5, 1, 2], [2, 0.5, 1]], dtype=torch.float32
    ).to(device)

    # Create expected output based on mathematical calculation
    expected_output = torch.tensor(
        [
            [84, 94.5, 105, 115.5],
            [147, 157.5, 168, 178.5],
            [210, 220.5, 231, 241.5],
            [273, 283.5, 294, 304.5],
        ],
        dtype=torch.float32,
    ).to(device)

    print(f"\nKernel:\n{kernel}")
    print(f"\nExpected output:\n{expected_output}")

    # Create 2D convolution layer
    conv = Conv2d(
        in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False
    ).to(device)

    # Set the kernel weights
    conv.weight.data = kernel.reshape(1, 1, 3, 3)

    # Apply convolution
    z = conv(x)
    print(f"\nOutput shape: {z.shape}")
    print(f"Actual output:\n{z.squeeze()}")

    # Verify results match
    match = torch.allclose(expected_output, z.squeeze(), atol=1e-6)
    print(f"\nOutputs match: {match}")


if __name__ == "__main__":
    conv_2d_example()

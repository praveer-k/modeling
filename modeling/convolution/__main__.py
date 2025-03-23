from click import command, option, group
from .conv_1d_example import conv_1d_example
from .conv_2d_example import conv_2d_example


@group("convolution")
def cli():
    """Convolutional Neural Network CLI"""
    pass


@cli.command()
def conv_1d():
    """Run a 1D convolution example"""
    conv_1d_example()


@cli.command()
def conv_2d():
    """Run a 2D convolution example"""
    conv_2d_example()


@cli.command()
def train():
    """Train a convolutional neural network"""
    print("Training CNN...")
    pass


@cli.command()
def test():
    """Test a convolutional neural network"""
    print("Testing CNN...")
    pass


@cli.command()
def predict():
    """Predict using a convolutional neural network"""
    print("Predicting using CNN...")
    pass


if __name__ == "__main__":
    cli()

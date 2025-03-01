Convolution
===========================

Convolution is a process of combining two functions to produce a third function. It is a mathematical operation that takes two functions f and g and produces a third function that represents how the shape of one is modified by the other.
For example, if we take two functions f and g, then the convolution of f and g is given by:

.. math::
    (f * g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)d\tau

where `f` and `g` are two functions, `*` denotes the convolution operator, and `t` is the variable of integration.

1D Convolution
----------------
A 1D convolution is a linear operation that involves the multiplication of a kernel with an input to produce an output. The kernel is a small matrix that slides over the input data, performing an element-wise multiplication with the input data and summing the result to produce the output.
Say we have a 1D input array and a scalar kernel, example:

Input array: [1, 2, 3, 4, 5]

Kernel: 2

Then, 1D Convolution is just an element-wise multiplication of the input array with the kernel. The output is produced by multiplying each element of the input array with the kernel.

.. math::
    [1, 2, 3, 4, 5, 6] * 2 = [2, 4, 6, 8, 10, 12]

Similarly, if we have a 1D input array and a 1D kernel, example:
Then, The convolution operation is produced by sliding the kernel over the input array and performing an element-wise multiplication of the kernel and the input array at each step. The results are summed to produce the output.
Here is the break down of the convolution operation:

Input Array :math:`f = [1, 2, 3, 4, 5, 6]`

Kernel or, Weights :math:`g = [1, 2, 0.5]`

.. math::
    \begin{array}{rcl}
    f & = & [1, 2, 3, 4, 5, 6] \\
    g & = & [1, 2, 0.5] \\[1em]
    \text{Position 1:} & & \begin{bmatrix}
    1 & 2 & 3 \\
    1 & 2 & 0.5
    \end{bmatrix} \rightarrow 6.5 \\[1em]
    & & [1\times1 + 2\times2 + 3\times0.5] & = & 6.5 \\[1em]
    \text{Position 2:} & & \begin{bmatrix}
    2 & 3 & 4 \\
    1 & 2 & 0.5
    \end{bmatrix} \rightarrow 10 \\[1em]
    & & [2\times1 + 3\times2 + 4\times0.5] & = & 10 \\[1em]
    \text{Position 3:} & & \begin{bmatrix}
    3 & 4 & 5 \\
    1 & 2 & 0.5
    \end{bmatrix} \rightarrow 13.5 \\[1em]
    & & [3\times1 + 4\times2 + 5\times0.5] & = & 13.5 \\[1em]
    \text{Position 4:} & & \begin{bmatrix}
    4 & 5 & 6 \\
    1 & 2 & 0.5
    \end{bmatrix} \rightarrow 17 \\[1em]
    & & [4\times1 + 5\times2 + 6\times0.5] & = & 17 \\[1em]
    &\qquad \text{output:}\implies [6.5, 10, 13.5, 17]
    \end{array}

example:

.. literalinclude:: ../../convolution/conv_1d_example.py
   :language: python
   :linenos:
   :start-after: def conv_1d_example():
   :end-before: if __name__ == "__main__":
   :dedent: 4

The output of running this code will be:

.. code-block:: text

    Input           : tensor([1., 2., 3., 4., 5., 6.])
    Input shape     : torch.Size([1, 1, 6])

    Expected output : tensor([ 6.5, 10.0, 13.5, 17.0])
    
    Kernel weights  : tensor([1.0, 2.0, 0.5])

    Actual output   : tensor([ 6.5, 10.0, 13.5, 17.0])
    Output shape    : torch.Size([1, 1, 4])

    Outputs match   : True


.. note::
    That the Conv1d module in PyTorch expects:

    * Input tensor to have the shape:
      ``W``
    
    * Kernel tensor to have the shape:
      ``F``
    
    * Output tensor will have the shape:
      ``(W-F+2P)/S + 1``
    
    where,

    :math:`O \implies \text{Output or, Feature map,}`

    :math:`W \implies \text{Input size or, Feature Vector,}`

    :math:`F \implies \text{Kernel size or, Filter size or, Weights,}`

    :math:`P \implies padding,`
    
    :math:`S \implies stride`

The output shape can be calclated using the following formula:

.. math::
    \text{output_size} = \left(\frac{\text{input_size} - \text{kernel_size} + 2 \times \text{padding}}{\text{stride}}\right) + 1

.. math::
    \text{O} = \left(\frac{\text{W} - \text{F} + 2 \times \text{P}}{\text{S}}\right) + 1

where `input_size` is the size of the input tensor, `kernel_size` is the size of the kernel tensor, `padding` is the padding applied to the input tensor, `stride` is the stride of the convolution operation, and `output_size` is the size of the output tensor.

2D Convolution
----------------


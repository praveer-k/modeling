Convolution
===========================

Convolution is a process of combining two functions to produce a third function. It is a mathematical operation that takes two functions f and g and produces a third function that represents how the shape of one is modified by the other.
The idea of convolution steams from the theory of moving averages and is used in signal processing, image processing, and other fields. 

Convolution is a linear operation that involves the multiplication of a kernel with an input to produce an output. The kernel is a small matrix that slides over the input data, performing an element-wise multiplication with the input data and summing the result to produce the output.
For example, if we take two functions :math:`f` and :math:`g`, then the convolution of :math:`f` and :math:`g` is given by:

.. math::
    (f * g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)d\tau

where,
 :math:`f` and, :math:`g` are two functions, 
 
 :math:`*` denotes the convolution operator, and 

 :math:`t` is the variable of integration. The letter :math:`t` is usually used because conventionally convolution is a function of time.

Example:
    * `Moving Averages <https://betterexplained.com/articles/intuitive-convolution/#Moving_averages>`__, where we take the average of a set of numbers and slide the window over another set of numbers to produce the output. The moving average is a convolution operation where the input is the set of numbers and the kernel is the window size. A real world use case would be to take the average of the stock prices over a certain period of time to smooth out the fluctuations in the stock prices.

    * In medical domain (`Hospital Analogy <https://betterexplained.com/articles/intuitive-convolution/#Part_1_Hospital_Analogy>`__) if the doctor has to administer a drug to many patients and each patient is given reducing dosage of the drug over a period of time, then the convolution operation can be used to calculate the total dosage of the drug administered to all the patients over the period of time.


1D Convolution
----------------
A 1D convolution is a linear operation that involves the multiplication of a kernel with an input to produce an output. The kernel is a small matrix that slides over the input data, performing an element-wise multiplication with the input data and summing the result to produce the output.
Say we have a 1D input array and a scalar kernel, example:

Input array: [1, 2, 3, 4, 5, 6]

Kernel: 2

Then, 1D Convolution is just an element-wise multiplication of the input array with the kernel. The output is produced by multiplying each element of the input array with the kernel.

.. math::
    [1, 2, 3, 4, 5, 6] * 2 = [2, 4, 6, 8, 10, 12]

Similarly, if we have a 1D input array and a 1D kernel, example:
Then, The convolution operation is produced by sliding the kernel over the input array and performing an element-wise multiplication of the kernel and the input array at each step. The result at each step are summed to produce the output.
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

.. note::
    That the size of the input array is 6 and the size of the output array is 4. The size of the output array is calculated using the formula:

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

    * In some cases we want the output to have the same size as the input, in such cases we can use padding to achieve that.
    * Padding is the process of adding zeros to the input tensor to increase the size of the output tensor. Padding is 2 times because we add zeros to both sides of the input tensor.
    * Stride is the number of steps the kernel moves before applying the multiplication over the input tensor while sliding over it.

The output shape can be calculated using the following formula:

.. math::
    \text{output_size} = \left(\frac{\text{input_size} - \text{kernel_size} + 2 \times \text{padding}}{\text{stride}}\right) + 1

.. math::
    \text{O} = \left(\frac{\text{W} - \text{F} + 2 \times \text{P}}{\text{S}}\right) + 1

where `input_size` is the size of the input tensor, `kernel_size` is the size of the kernel tensor, `padding` is the padding applied to the input tensor, `stride` is the stride of the convolution operation, and `output_size` is the size of the output tensor.

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


2D Convolution
----------------
2D convolution is a linear operation that involves the multiplication of a kernel with an input to produce an output. The kernel is a small matrix that slides over the input data, performing an element-wise multiplication with the input data and summing the result to produce the output.
Say we have a 2D input array and a 2D kernel, example:

Input array: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

kernel: [[1, 0, 1], [0, 1, 0], [1, 0, 1]]

Then, 2D Convolution is just an element-wise multiplication of the input array with the kernel. The output is produced by multiplying each element of the input array with the kernel and summing the result.

.. math::
    \begin{array}{rcl}
    \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
    \end{bmatrix} * \begin{bmatrix}
    1 & 0 & 1 \\
    0 & 1 & 0 \\
    1 & 0 & 1
    \end{bmatrix} = \sum \begin{bmatrix}
    1 & 0 & 3 \\
    0 & 5 & 0 \\
    7 & 0 & 9
    \end{bmatrix} = \begin{bmatrix}26\end{bmatrix}
    \end{array}

Similarly, if we have a 2D input array with dimensions 6x6 and a 2D kernel with dimension 3x3, example:
Then, Again the convolution operation is produced by sliding the kernel over the input array and performing an element-wise multiplication of the kernel and the input array at each step. The results are summed to produce the output.
Here is the break down of the convolution operation:

Input Array :math:`f = \begin{bmatrix}
1 & 2 & 3 & 4 & 5 & 6 \\
7 & 8 & 9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16 & 17 & 18 \\
19 & 20 & 21 & 22 & 23 & 24 \\
25 & 26 & 27 & 28 & 29 & 30 \\
31 & 32 & 33 & 34 & 35 & 36
\end{bmatrix}`
Kernel or, Weights :math:`g = \begin{bmatrix}
1 & 2 & 0.5 \\
0.5 & 1 & 2 \\
2 & 0.5 & 1
\end{bmatrix}`

The output is produced by sliding the kernel over the input array and performing an element-wise multiplication of the kernel and the input array at each step. The results are summed to produce the output.

.. math::
    \begin{array}{ll}
    \text{Position 1x1:} \begin{bmatrix}
    1 & 2 & 3 \\
    7 & 8 & 9 \\
    13 & 14 & 15
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 84 \\[1em]
    \text{Position 1x2:} \begin{bmatrix}
    2 & 3 & 4 \\
    8 & 9 & 10 \\
    14 & 15 & 16
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 94.5 \\[1em]
    \text{Position 1x3:} \begin{bmatrix}
    3 & 4 & 5 \\
    9 & 10 & 11 \\
    15 & 16 & 17
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 105 \\[1em]
    \text{Position 1x4:} \begin{bmatrix}
    4 & 5 & 6 \\
    10 & 11 & 12 \\
    16 & 17 & 18
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 115.5 \\[1em]
    \text{Position 2x1:} \begin{bmatrix}
    7 & 8 & 9 \\
    13 & 14 & 15 \\
    19 & 20 & 21
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 147 \\[1em]
    \text{Position 2x2:} \begin{bmatrix}
    8 & 9 & 10 \\
    14 & 15 & 16 \\
    20 & 21 & 22
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 157.5 \\[1em]
    \text{Position 2x3:} \begin{bmatrix}
    9 & 10 & 11 \\
    15 & 16 & 17 \\
    21 & 22 & 23
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 168 \\[1em]
    \text{Position 2x4:} \begin{bmatrix}
    10 & 11 & 12 \\
    16 & 17 & 18 \\
    22 & 23 & 24
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 178.5 \\[1em]
    \text{Position 3x1:} \begin{bmatrix}
    13 & 14 & 15 \\
    19 & 20 & 21 \\
    25 & 26 & 27
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 210 \\[1em]
    \text{Position 3x2:} \begin{bmatrix}
    14 & 15 & 16 \\
    20 & 21 & 22 \\
    26 & 27 & 28
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 220.5 \\[1em]
    \text{Position 3x3:} \begin{bmatrix}
    15 & 16 & 17 \\
    21 & 22 & 23 \\
    27 & 28 & 29
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 231 \\[1em]
    \text{Position 3x4:} \begin{bmatrix}
    16 & 17 & 18 \\
    22 & 23 & 24 \\
    28 & 29 & 30
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 241.5 \\[1em]
    \text{Position 4x1:} \begin{bmatrix}
    19 & 20 & 21 \\
    25 & 26 & 27 \\
    31 & 32 & 33
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 273 \\[1em]
    \text{Position 4x2:} \begin{bmatrix}
    20 & 21 & 22 \\
    26 & 27 & 28 \\
    32 & 33 & 34
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 283.5 \\[1em]
    \text{Position 4x3:} \begin{bmatrix}
    21 & 22 & 23 \\
    27 & 28 & 29 \\
    33 & 34 & 35
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 294 \\[1em]
    \text{Position 4x4:} \begin{bmatrix}
    22 & 23 & 24 \\
    28 & 29 & 30 \\
    34 & 35 & 36
    \end{bmatrix} * \begin{bmatrix}
    1 & 2 & 0.5 \\
    0.5 & 1 & 2 \\
    2 & 0.5 & 1
    \end{bmatrix} \rightarrow 304.5 \\[1em]
    \text{output:}\implies \begin{bmatrix}
    84 & 94.5 & 105 & 115.5 \\
    147 & 157.5 & 168 & 178.5 \\
    210 & 220.5 & 231 & 241.5 \\
    273 & 283.5 & 294 & 304.5
    \end{bmatrix}
    \end{array}

.. note::
    That the size of the input array is 3x3 and kernel are the same and hence the two matrices are multiplied element-wise and summed to produce the output. The size of the output array is 1x1.
    Again, The size of the output array is calculated using the formula:

    * Input tensor to have the shape:
      ``W x H``
    
    * Kernel tensor to have the shape:
      ``F x F``
    
    * Output tensor will have the shape:
      ``(W-F+2P)/S + 1 x (H-F+2P)/S + 1``
    
    where,

    :math:`O \implies \text{Output or, Feature map,}`

    :math:`W \implies \text{Input size in x direction or, Feature Vector dimension in x direction,}`

    :math:`H \implies \text{Input size in y direction or, Feature Vector dimension in y direction,}`

    :math:`F \implies \text{Kernel size or, Filter size or, Weights,}`

    :math:`P \implies padding,`
    
    :math:`S \implies stride`

    * In some cases we want the output to have the same size as the input, in such cases we can use padding to achieve that.
    * Padding is the process of adding zeros to the input tensor to increase the size of the output tensor. Padding is 2 times because we add zeros to both sides of the input tensor.
    * Stride is the number of steps the kernel moves before applying the multiplication over the input tensor while sliding over it.

Filters
-------
Filters are the small matrices that are used to extract features from the input data. Filters are also known as kernels or, weights. Filters are used to extract features from the input data by performing an element-wise multiplication with the input data and summing the result to produce the output. Filters are used in convolutional neural networks to extract features from the input data. Filters are learned during the training process and are used to extract features from the input data. Filters are used to extract features such as edges, textures, and shapes from the input data.
There are various types of filters that are used in convolutional neural networks such as edge detection filters, texture filters, and shape filters. Filters are used to extract features from the input data and are used to learn the features that are important for the task at hand.
Based on the documentation, there are several types of filters (kernels) used in convolution operations. 

Here is the list of some of the filters used in convolution operations with their respective kernels
    1. Basic Filters:
        * Edge detection filters
            * Horizontal Edge Detection:
                .. math::
                        \begin{bmatrix}
                        1 & 1 & 1 \\
                        0 & 0 & 0 \\
                        -1 & -1 & -1
                        \end{bmatrix}

            * Vertical Edge Detection:
                .. math::
                        \begin{bmatrix}
                        1 & 0 & -1 \\
                        1 & 0 & -1 \\
                        1 & 0 & -1
                        \end{bmatrix}

            * Diagonal Edge Detection:
                .. math::
                        \begin{bmatrix}
                        -1 & -1 & 2 \\
                        -1 & 2 & -1 \\
                        2 & -1 & -1
                        \end{bmatrix}
        * Texture filters
            * Texture Detection:
                .. math::
                        \begin{bmatrix}
                        -1 & -1 & -1 \\
                        2 & 2 & 2 \\
                        -1 & -1 & -1
                        \end{bmatrix}
                        \quad
                        \begin{bmatrix}
                        -1 & 2 & -1 \\
                        -1 & 2 & -1 \\
                        -1 & 2 & -1
                        \end{bmatrix}

            * Pattern Detection:
                .. math::
                        \begin{bmatrix}
                        1 & -1 & 1 \\
                        -1 & 1 & -1 \\
                        1 & -1 & 1
                        \end{bmatrix}
        * Shape filters
            * Circle Detection:
                .. math::
                        \begin{bmatrix}
                        0 & 1 & 0 \\
                        1 & -4 & 1 \\
                        0 & 1 & 0
                        \end{bmatrix}

            * Square Detection:
                .. math::
                        \begin{bmatrix}
                        1 & 1 & 1 \\
                        1 & -8 & 1 \\
                        1 & 1 & 1
                        \end{bmatrix}
        * Color filters
            * Red Detection:
                .. math::
                        \begin{bmatrix}
                        1 & 0 & 0 \\
                        0 & 0 & 0 \\
                        0 & 0 & 0
                        \end{bmatrix}

            * Green Detection:
                .. math::
                        \begin{bmatrix}
                        0 & 1 & 0 \\
                        1 & 1 & 1 \\
                        0 & 1 & 0
                        \end{bmatrix}

            * Blue Detection:
                .. math::
                        \begin{bmatrix}
                        0 & 0 & 0 \\
                        0 & 0 & 0 \\
                        1 & 0 & 0
                        \end{bmatrix}
        * Blur filters
            .. math::
                \begin{bmatrix}
                1/9 & 1/9 & 1/9 \\
                1/9 & 1/9 & 1/9 \\
                1/9 & 1/9 & 1/9
                \end{bmatrix}
            
        * Sharpen filters
            .. math::
                \begin{bmatrix}
                0 & -1 & 0 \\
                -1 & 5 & -1 \\
                0 & -1 & 0
                \end{bmatrix}
            
        * Emboss filters
            .. math::
                \begin{bmatrix}
                -2 & -1 & 0 \\
                -1 & 1 & 1 \\
                0 & 1 & 2
                \end{bmatrix}

    2. Edge Detection Specific:
        * Sobel filters
            * Horizontal Sobel:
                .. math::
                    \begin{bmatrix}
                    -1 & 0 & 1 \\
                    -2 & 0 & 2 \\
                    -1 & 0 & 1
                    \end{bmatrix}
            * Vertical Sobel:
                .. math::
                    \begin{bmatrix}
                    -1 & -2 & -1 \\
                    0 & 0 & 0 \\
                    1 & 2 & 1
                    \end{bmatrix}

        * Prewitt filters
            * Horizontal Prewitt:
                .. math::
                    \begin{bmatrix}
                    -1 & 0 & 1 \\
                    -1 & 0 & 1 \\
                    -1 & 0 & 1
                    \end{bmatrix}
            * Vertical Prewitt:
                .. math::
                    \begin{bmatrix}
                    -1 & -1 & -1 \\
                    0 & 0 & 0 \\
                    1 & 1 & 1
                    \end{bmatrix}
        * Canny filters
            * Horizontal Canny:
                .. math::
                    \begin{bmatrix}
                    -1 & -1 & -1 \\
                    2 & 2 & 2 \\
                    -1 & -1 & -1
                    \end{bmatrix}
            * Vertical Canny:
                .. math::
                    \begin{bmatrix}
                    -1 & 2 & -1 \\
                    -1 & 2 & -1 \\
                    -1 & 2 & -1
                    \end{bmatrix}

        * Roberts filters
            * Horizontal Roberts:
                .. math::
                    \begin{bmatrix}
                    1 & 0 \\
                    0 & -1
                    \end{bmatrix}
            * Vertical Roberts: 
                .. math::
                    \begin{bmatrix}
                    0 & 1 \\
                    -1 & 0
                    \end{bmatrix}

        * Kirsch filters
            * Horizontal Kirsch:
                .. math::
                    \begin{bmatrix}
                    5 & 5 & 5 \\
                    -3 & 0 & -3 \\
                    -3 & -3 & -3
                    \end{bmatrix}
            * Vertical Kirsch:
                .. math::
                    \begin{bmatrix}
                    5 & -3 & -3 \\
                    5 & 0 & -3 \\
                    5 & -3 & -3
                    \end{bmatrix}
        * Robinson filters
            * Horizontal Robinson:
                .. math::
                    \begin{bmatrix}
                    1 & 1 & 1 \\
                    0 & 0 & 0 \\
                    -1 & -1 & -1
                    \end{bmatrix}
            * Vertical Robinson:
                .. math::
                    \begin{bmatrix}
                    1 & 0 & -1 \\
                    1 & 0 & -1 \\
                    1 & 0 & -1
                    \end{bmatrix}
        * Nevatia-Babu filters
            * Horizontal Nevatia-Babu:
                .. math::
                    \begin{bmatrix}
                    100 & 100 & 100 & 100 & 100 \\
                    100 & 100 & 100 & 100 & 100 \\
                    0 & 0 & 0 & 0 & 0 \\
                    -100 & -100 & -100 & -100 & -100 \\
                    -100 & -100 & -100 & -100 & -100
                    \end{bmatrix}
            * Vertical Nevatia-Babu:
                .. math::
                    \begin{bmatrix}
                    -100 & -100 & 0 & 100 & 100 \\
                    -100 & -100 & 0 & 100 & 100 \\
                    -100 & -100 & 0 & 100 & 100 \\
                    -100 & -100 & 0 & 100 & 100 \\ 
                    -100 & -100 & 0 & 100 & 100
                    \end{bmatrix}
        * Gradient filters
            * Horizontal Gradient:
                .. math::
                    \begin{bmatrix}
                    -1 & 0 & 1 \\
                    -1 & 0 & 1 \\
                    -1 & 0 & 1
                    \end{bmatrix}
            * Vertical Gradient:
                .. math::
                    \begin{bmatrix}
                    -1 & -1 & -1 \\
                    0 & 0 & 0 \\
                    1 & 1 & 1
                    \end{bmatrix}

    3. Gaussian-based:
        * Gaussian filters
            .. math::
                \begin{bmatrix}
                1 & 2 & 1 \\
                2 & 4 & 2 \\
                1 & 2 & 1
                \end{bmatrix}
        * Gaussian derivative filters
            * Gaussian Derivative:
                .. math::
                    \begin{bmatrix}
                    -1 & -2 & -1 \\
                    0 & 0 & 0 \\
                    1 & 2 & 1
                    \end{bmatrix}
            * Gaussian Derivative of Gaussian:
                .. math::
                    \begin{bmatrix}
                    1 & 2 & 1 \\
                    2 & 4 & 2 \\
                    1 & 2 & 1
                    \end{bmatrix}
        * Laplacian filters
            * Laplacian:
                .. math::
                    \begin{bmatrix}
                    0 & 1 & 0 \\
                    1 & -4 & 1 \\
                    0 & 1 & 0
                    \end{bmatrix}
            * Laplacian of Gaussian:
                .. math::
                    \begin{bmatrix}
                    0 & 0 & 1 & 0 & 0 \\
                    0 & 1 & 2 & 1 & 0 \\
                    1 & 2 & -16 & 2 & 1 \\
                    0 & 1 & 2 & 1 & 0 \\
                    0 & 0 & 1 & 0 & 0
                    \end{bmatrix}
        * Difference of Gaussian filters
            * Difference of Gaussian:
                .. math::
                    \begin{bmatrix}
                    1 & 1 & 1 & 1 & 1 \\
                    1 & 2 & 2 & 2 & 1 \\
                    1 & 2 & 0 & 2 & 1 \\
                    1 & 2 & 2 & 2 & 1 \\
                    1 & 1 & 1 & 1 & 1
                    \end{bmatrix}
        * Marr-Hildreth filters
            * Marr-Hildreth:
                .. math::
                    \begin{bmatrix}
                    0 & 0 & -1 & 0 & 0 \\
                    0 & -1 & -2 & -1 & 0 \\
                    -1 & -2 & 16 & -2 & -1 \\
                    0 & -1 & -2 & -1 & 0 \\
                    0 & 0 & -1 & 0 & 0
                    \end{bmatrix}
        
    4. Advanced Filters:
        * Gabor filters
            
        * Frei-Chen filters
        * Homomorphic filters
        * Wiener filters
        * Bilateral filters
        * Non-local means filters
        * Total variation filters

    5. Adaptive Filters:
        * Adaptive median filters
        * Adaptive bilateral filters
        * Adaptive Wiener filters
        * Adaptive anisotropic diffusion filters
        * Adaptive total variation filters
        * Adaptive non-local means filters

These filters serve different purposes in image processing and computer vision tasks, from simple edge detection to complex noise reduction and feature extraction.
To implement any of these filters in PyTorch, you would define the kernel weights accordingly. 
For example:

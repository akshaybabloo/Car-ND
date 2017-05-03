Sobel Operator
==============

The Sobel operator is at the heart of the Canny edge detection algorithm you used in the Introductory Lesson. Applying the Sobel operator to an image is a way of taking the derivative of the image in the :math:`x` or :math:`y` direction. The operators for :math:`sobel_{x}` and :math:`sobel_{y}`  respectively, look like this:

.. math::

    S_{x} = \begin{bmatrix}
            -1 & 0 & 1\\
            -2 & 0 & 2\\
            -1 & 0 & 1
            \end{bmatrix} \\
    S_{y} = \begin{bmatrix}
            -1 & -2 & -1\\
            0 & 0 & 0\\
            1 & 2 & 1
            \end{bmatrix}

These are examples of Sobel operators with a kernel size of 3 (implying a 3 x 3 operator in each case). This is the minimum size, but the kernel size can be any odd number. A larger kernel implies taking the gradient over a larger region of the image, or, in other words, a smoother gradient.

To understand how these operators take the derivative, you can think of overlaying either one on a 3 x 3 region of an image. If the image is flat across that region, then the result (summing the element-wise product of the operator and corresponding image pixels) will be zero. If, instead, for example, you apply the :math:`S_{x}` operator to a region of the image where values are rising from left to right, then the result will be positive, implying a positive derivative.

Derivative Example
------------------

If we apply the Sobel x and y operators to this image:

.. figure:: curved-lane.jpg
   :align: center

   Source: Udacity

And then we take the absolute value, we get the result:

.. figure:: screen-shot-2016-12-01-at-4.50.36-pm.png
   :align: center

   Source: Udacity

Magnitude of Gradient
---------------------

With the result of the last quiz, you can now take the gradient in x or y and set thresholds to identify pixels within a certain gradient range. If you play around with the thresholds a bit, you'll find the x-gradient does a cleaner job of picking up the lane lines, but you can see the lines in the y-gradient as well.

In this next exercise, your goal is to apply a threshold to the overall magnitude of the gradient, in both x and y.

The magnitude, or absolute value, of the gradient is just the square root of the squares of the individual x and y gradients. For a gradient in both the x and y directions, the magnitude is the square root of the sum of the squares.

.. math::

    abs\_sobelx = \sqrt{\left ( sobel_{x} \right )^{2}} \\
    abs\_sobely = \sqrt{\left ( sobel_{y} \right )^{2}} \\
    abs\_sobelxy = \sqrt{\left ( sobel_{x} + sobel_{y} \right )^{2}}

It's also worth considering the size of the region in the image over which you'll be taking the gradient. You can modify the kernel size for the Sobel operator to change the size of this region. Taking the gradient over larger regions can smooth over noisy intensity fluctuations on small scales. The default Sobel kernel size is 3, but here you'll define a new function that takes kernel size as a parameter (must be an odd number!)

The function you'll define for the exercise below should take in an image and optional Sobel kernel size, as well as thresholds for gradient magnitude. Next, you'll compute the gradient magnitude, apply a threshold, and create a binary output image showing where thresholds were met.

**Steps to take in this exercise:**

1. Fill out the function in the editor below to return a thresholded gradient magnitude. Again, you can apply exclusive ``(<, >)`` or inclusive ``(<=, >=)`` thresholds.
2. Test that your function returns output similar to the example below for ``sobel_kernel=9, mag_thresh=(30, 100)``.

Direction of the Gradient
-------------------------

When you play around with the thresholding for the gradient magnitude in the previous exercise, you find what you might expect, namely, that it picks up the lane lines well, but with a lot of other stuff detected too. Gradient magnitude is at the heart of Canny edge detection, and is why Canny works well for picking up all edges.

In the case of lane lines, we're interested only in edges of a particular orientation. So now we will explore the direction, or orientation, of the gradient.

The direction of the gradient is simply the inverse tangent (arctangent) of the y gradient divided by the x gradient:

.. math::

    arctan \left ( sobel_{y} / sobel_{x} \right )

Each pixel of the resulting image contains a value for the angle of the gradient away from horizontal in units of radians, covering a range of :math:`-\frac{\pi }{2}` to :math:`\frac{\pi }{2}`. An orientation of 0 implies a horizontal line and orientations of :math:`\pm \frac{\pi }{2}`

In this next exercise, you'll write a function to compute the direction of the gradient and apply a threshold. The direction of the gradient is much noisier than the gradient magnitude, but you should find that you can pick out particular features by orientation.

**Steps to take in this exercise:**

1. Fill out the function in the editor below to return a thresholded absolute value of the gradient direction. Use Boolean operators, again with exclusive ``(<, >)`` or inclusive ``(<=, >=)`` thresholds.
2. Test that your function returns output similar to the example below for ``sobel_kernel=15, thresh=(0.7, 1.3)``.

Combining them together
------------------------

Combining them together using the following condition:

.. code-block:: python

   combined = np.zeros_like(dir_binary)
   combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

Code for Combination
^^^^^^^^^^^^^^^^^^^^

.. automodule:: Term_1.advanced_lane_finding_10.combining_threshold_10_7
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
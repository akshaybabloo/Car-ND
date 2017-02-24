def python_run():
    """
    Pythonic way of getting the ``Parameter Sharing``
    """
    input_shape = [32, 32, 3]  # HxWxD
    filter_shape = [20, 8, 8, 3]  # number_of_filtersxHxWxD
    stride = 2  # S
    valid_padding = 1  # P

    new_height = (input_shape[0] - filter_shape[1] + 2 * valid_padding) / stride + 1
    new_width = (input_shape[1] - filter_shape[2] + 2 * valid_padding) / stride + 1
    new_depth = filter_shape[0]  # number of filters is the depth

    parameter_sharing = (filter_shape[1] * filter_shape[2] * filter_shape[3] + 1) * filter_shape[0]
    parameter_sharing += new_depth

    print("Parameter Sharing ", parameter_sharing)

if __name__ == '__main__':
    python_run()

def python_run():
    """
    Pythonic way of getting the ``new width`` and ``new height``
    """
    input_shape = [32, 32, 3]  # HxWxD
    filter_shape = [20, 8, 8, 3]  # number_of_filtersxHxWxD
    stride = 2  # S
    valid_padding = 1  # P

    new_height = (input_shape[0] - filter_shape[1] + 2 * valid_padding) / stride + 1
    new_width = (input_shape[1] - filter_shape[2] + 2 * valid_padding) / stride + 1
    new_depth = filter_shape[0]  # number of filters is the depth

    print("{}x{}x{}".format(new_height, new_width, new_depth))

    total_parameters = (filter_shape[1] * filter_shape[2] * filter_shape[3] + 1) * (new_height * new_width * new_depth)

    print("Total parameters ", total_parameters)

if __name__ == '__main__':
    python_run()

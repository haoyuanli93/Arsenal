import numpy as np
import holoviews as hv


########################################################################################################
# Assemble and show the patterns
########################################################################################################
def assemble_patterns_image(data_holder, row_num, col_num, index, value_range, pattern_shape):
    """
    After the program has obtained the index of the patterns in the selected region,
    this function randomly choose several of the patterns to show in a grid-space.

    Specifically, each pattern in this function is a holoview image object.


    :param data_holder: The holder containing all the data shown in the diagram
    :param row_num: The row number of the grid space
    :param col_num: The column number of the grid space
    :param index: The index of all the data in the selected region
    :param value_range: The range of values to show a numpy array as RGB image.
    :param pattern_shape: The pattern shape
    :return: hv.GridSpace
    """
    index = np.array(index)
    index_num = index.shape[0]
    if index_num >= row_num * col_num:
        np.random.shuffle(index)
        sampled_index = index[:row_num * col_num]
        sampled_index = sampled_index.reshape((row_num, col_num))

        image_holder = {(x, y): hv.Image(data_holder[sampled_index[x, y]]).redim.range(z=(value_range[0],
                                                                                          value_range[1]))
                        for x in range(row_num) for y in range(col_num)}
    else:
        # When we do not have so many patterns, first layout
        # all the patterns available and then fill the other
        # positions with patterns of zeros.
        index_list = [(x, y) for x in range(row_num) for y in range(col_num)]
        image_holder = {index_list[l]: hv.Image(data_holder[index[l]]).redim.range(z=(value_range[0],
                                                                                      value_range[1]))
                        for l in range(index_num)}
        image_holder.update({index_list[l]: hv.Image(np.zeros(pattern_shape,
                                                              dtype=np.float64))
                             for l in range(index_num, row_num * col_num)})

    return hv.GridSpace(image_holder)


def assemble_patterns_curve(y_data_array, y_range, x_data, x_range, row_num, col_num, index):
    """
    After the program has obtained the index of the patterns in the selected region,
    this function randomly choose several of the patterns to show in a grid-space.

    Specifically, each pattern in this function is a holoview curve object. This function also
    assume that all curves share the same x-axis.

    :param y_data_array: 2D numpy array of the shape [n, curve_length] where n is the pattern number in total.
    :param y_range: The y range of the pattern.
    :param x_data: The x coordinate of the curve.
    :param x_range: The x range of the pattern.
    :param row_num: The row number of the grid space
    :param col_num: The column number of the grid space
    :param index: The index of all the data in the selected region
    :return: hv.GridSpace
    """
    index = np.array(index)
    index_num = index.shape[0]

    # Get the span of x
    x_span = (x_data.max() - x_data.min())

    if y_range == 'auto' and x_range == 'auto':
        if index_num >= row_num * col_num:

            # Extract some samples from all the selected ones.
            np.random.shuffle(index)
            sampled_index = index[:row_num * col_num]
            sampled_index = sampled_index.reshape((row_num, col_num))

            # Assemble the patterns
            image_holder = {}
            for x in range(row_num):
                for y in range(col_num):
                    # Get the span of the y values
                    y_data = y_data_array[sampled_index[x, y]]
                    y_span = y_data.max() - y_data.min()

                    image_holder.update({(x, y): hv.Curve((x_data, y_data)).redim.range(
                        x=(x_data.min() - x_span * 0.05,
                           x_data.max() + x_span * 0.05),
                        y=(y_data.min() - y_span * 0.05,
                           y_data.max() + y_span * 0.05))})
        else:
            # When we do not have so many patterns, first layout
            # all the patterns available and then fill the other
            # positions with patterns of zeros.
            index_list = [(x, y) for x in range(row_num) for y in range(col_num)]

            # Assemble the True patterns
            image_holder = {}
            for l in range(index_num):
                # Get the span of the y values
                y_data = y_data_array[index[l]]
                y_span = y_data.max() - y_data.min()

                image_holder.update({index_list[l]: hv.Curve((x_data, y_data)).redim.range(
                    x=(x_data.min() - x_span * 0.05,
                       x_data.max() + x_span * 0.05),
                    y=(y_data.min() - y_span * 0.05,
                       y_data.max() + y_span * 0.05))})

            # Assemble the psudo-image
            image_holder.update({index_list[l]: hv.Curve((x_data, np.zeros_like(x_data))).redim.range(
                x=(x_data.min() - x_span * 0.05, x_data.max() + x_span * 0.05), y=(0, 1))
                for l in range(index_num, row_num * col_num)})

    elif y_range == 'auto' and x_range != 'auto':  # If y_range is auto while x is specified

        if index_num >= row_num * col_num:

            # Extract some samples from all the selected ones.
            np.random.shuffle(index)
            sampled_index = index[:row_num * col_num]
            sampled_index = sampled_index.reshape((row_num, col_num))

            # Assemble the patterns
            image_holder = {}
            for x in range(row_num):
                for y in range(col_num):
                    # Get the span of the y values
                    y_data = y_data_array[sampled_index[x, y]]
                    y_span = y_data.max() - y_data.min()

                    image_holder.update({(x, y): hv.Curve((x_data, y_data)).redim.range(
                        x=(x_range[0], x_range[1]),
                        y=(y_data.min() - y_span * 0.05,
                           y_data.max() + y_span * 0.05))})
        else:
            # When we do not have so many patterns, first layout
            # all the patterns available and then fill the other
            # positions with patterns of zeros.
            index_list = [(x, y) for x in range(row_num) for y in range(col_num)]

            # Assemble the True patterns
            image_holder = {}
            for l in range(index_num):
                # Get the span of the y values
                y_data = y_data_array[index[l]]
                y_span = y_data.max() - y_data.min()

                image_holder.update({index_list[l]: hv.Curve((x_data, y_data)).redim.range(
                    x=(x_range[0], x_range[1]),
                    y=(y_data.min() - y_span * 0.05,
                       y_data.max() + y_span * 0.05))})

            # Assemble the psudo-image
            image_holder.update({index_list[l]: hv.Curve((x_data, np.zeros_like(x_data))).redim.range(
                x=(x_range[0], x_range[1]), y=(0, 1))
                for l in range(index_num, row_num * col_num)})

    elif y_range != 'auto' and x_range == 'auto':

        if index_num >= row_num * col_num:

            # Extract some samples from all the selected ones.
            np.random.shuffle(index)
            sampled_index = index[:row_num * col_num]
            sampled_index = sampled_index.reshape((row_num, col_num))

            # Assemble the patterns
            image_holder = {}
            for x in range(row_num):
                for y in range(col_num):
                    # Get the span of the y values
                    y_data = y_data_array[sampled_index[x, y]]

                    image_holder.update({(x, y): hv.Curve((x_data, y_data)).redim.range(
                        x=(x_data.min() - x_span * 0.05,
                           x_data.max() + x_span * 0.05),
                        y=(y_range[0], y_range[1]))})
        else:
            # When we do not have so many patterns, first layout
            # all the patterns available and then fill the other
            # positions with patterns of zeros.
            index_list = [(x, y) for x in range(row_num) for y in range(col_num)]

            # Assemble the True patterns
            image_holder = {}
            for l in range(index_num):
                # Get the span of the y values
                y_data = y_data_array[index[l]]

                image_holder.update({index_list[l]: hv.Curve((x_data, y_data)).redim.range(
                    x=(x_data.min() - x_span * 0.05,
                       x_data.max() + x_span * 0.05),
                    y=(y_range[0], y_range[1]))})

            # Assemble the psudo-image
            image_holder.update({index_list[l]: hv.Curve((x_data, np.zeros_like(x_data))).redim.range(
                x=(x_data.min() - x_span * 0.05, x_data.max() + x_span * 0.05), y=(y_range[0], y_range[1]))
                for l in range(index_num, row_num * col_num)})

    else:  # If neither x_range or y_range is specified
        if index_num >= row_num * col_num:

            # Extract some samples from all the selected ones.
            np.random.shuffle(index)
            sampled_index = index[:row_num * col_num]
            sampled_index = sampled_index.reshape((row_num, col_num))

            # Assemble the patterns
            image_holder = {}
            for x in range(row_num):
                for y in range(col_num):
                    # Get the span of the y values
                    y_data = y_data_array[sampled_index[x, y]]

                    image_holder.update({(x, y): hv.Curve((x_data, y_data)).redim.range(
                        x=(x_range[0], x_range[1]), y=(y_range[0], y_range[1]))})
        else:
            # When we do not have so many patterns, first layout
            # all the patterns available and then fill the other
            # positions with patterns of zeros.
            index_list = [(x, y) for x in range(row_num) for y in range(col_num)]

            # Assemble the True patterns
            image_holder = {}
            for l in range(index_num):
                # Get the span of the y values
                y_data = y_data_array[index[l]]

                image_holder.update({index_list[l]: hv.Curve((x_data, y_data)).redim.range(
                    x=(x_range[0], x_range[1]), y=(y_range[0], y_range[1]))})

            # Assemble the psudo-image
            image_holder.update({index_list[l]: hv.Curve((x_data, np.zeros_like(x_data))).redim.range(
                x=(x_range[0], x_range[1]), y=(y_range[0], y_range[1]))
                for l in range(index_num, row_num * col_num)})

    return hv.GridSpace(image_holder).options(shared_xaxis=False, shared_yaxis=False)

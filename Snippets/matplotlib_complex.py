import matplotlib.pyplot as plt
import numpy as np

##############################################################################
# Plot many patterns together
##############################################################################
"""
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %      Many Line plots
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        This section consider the following case.
        Assume that one has some distributions, some fitted functions
        and the fitting region. One wants to show them all
        in a single plot.
        
        Notice that in this plot. Only 
        
                x_data and y_data 
        
        controls the plotting parameters
        
"""
# [USER] Specify Meta Data
nrow = 3  # Number of rows
ncol = 3  # Number of columns
plot_num = 64  # How many patterns in total
index_to_show = np.random.permutation(plot_num)[: nrow * ncol]  # select some data to show

# [USER] Specify Data
x_data_main = np.arange(128)  # The x data to show on the axis
y_data_main = np.random.rand(plot_num, 128)  # The y data to show on the axis

param_main = np.random.rand(plot_num)  # The first parameter associated with the plot
legend_main = "Data main"

# Specify some other data to show
x_data_1 = np.arange(128)
y_data_1 = np.random.rand(128)  # The second class of y data

legend_1 = "Data 1"

# Specify parameters associated with the data
param_1 = np.random.rand(plot_num)  # The second parameter associated with the the plot
param_2 = np.random.rand(plot_num)  # The third parameter associated with the the plot


# TODO Allow the user to use functions to customize the dynamical y data
# Perhaps one needs to calculate some plot according to the
def test_function(*args):
    if args:
        pass
    pass


legend_function = "Data Function"

# [USER] Specify format parameters

# Figure size
fig_height = 16
fig_width = 16

# Set range range
xlim_flag = True
x_lim = (0, np.max(x_data_main))

# TODO Allow the user to use lambda function to customize the dynamical y range
ylim_flag = False  # Whether not to use the default y lim
ylim_static_flag = True  # Whether to use the same y lim
y_lim = (0, 0.2)

# X axis label
xname = "Name of x axis"
x_unit = "Unit"

xtick_flag = True  # Whether not to use the default x tick
xticks_span = 8
xticks = x_data_main[::xticks_span]
xticks_labels = ["{:.1f}".format(1 / x) for x in xticks]

# Y axis label
yname = "Name of y axis"
y_unit = "Unit"

# TODO Allow the user to use lambda function to customize the dynamical y ticks
yticks_flag = False
ytick_static_flag = False  # Whether to use the same y ticks
yticks_num = 15
yticks = np.linspace(np.min(y_data_main), np.max(y_data_main), num=yticks_num)
yticks_labels = ["{:.1f}".format(x) for x in yticks]

# Specify the font sizes
title_font_size = 15

xlabel_font = 15
xtick_font_size = 10

ylabel_font = 15
ytick_font_size = 10

# [AUTO] Set up the canvas
fig, axes = plt.subplots(nrows=nrow, ncols=ncol)
fig.set_figheight(fig_height)
fig.set_figwidth(fig_width)

# [AUTO] Plot the patterns
for l in range(nrow):
    for m in range(ncol):

        # Get the ax object
        ax = axes[l, m]

        # Get the index to show
        idx = index_to_show[l * nrow + m]

        # Plot
        ax.plot(x_data_main, y_data_main[idx], '-o', label=legend_main)

        # Plot some other data
        ax.plot(x_data_main, test_function(x_data_main), label=legend_function)

        # Set title
        ax.set_title("Index {}, Param_1 {}, Param_2 {}".format(idx,
                                                               param_1[idx],
                                                               param_2[idx]),
                     fontsize=title_font_size)

        # Set axis labels
        ax.set_xlabel("{} ({})".format(xname, x_unit), fontsize=xlabel_font)
        ax.set_ylabel("{} ({})".format(yname, y_unit), fontsize=ylabel_font)

        # Set plot range
        if ylim_flag:
            if ylim_static_flag:
                ax.set_ylim(y_lim)
            else:
                # Calculate the y_lim
                ylim_low = np.min(y_data_main[idx])
                ylim_high = 1 * ylim_low + 1. * (np.max(y_data_main[idx]) - ylim_low)

                y_lim = (ylim_low, ylim_high)
                ax.set_ylim(y_lim)

        if xlim_flag:
            ax.set_xlim(x_lim)

        # Set axis ticks
        if xtick_flag:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks_labels[::xticks_span], fontsize=xtick_font_size)

        if yticks_flag:
            if ytick_static_flag:
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks_labels, fontsize=ytick_font_size)
            else:

                # Get dynamic label
                yticks = np.linspace(np.min(y_data_main[idx]),
                                     np.max(y_data_main[idx]),
                                     num=yticks_num)
                yticks_labels = ["{:.1f}".format(x) for x in yticks]

                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks_labels, fontsize=ytick_font_size)

        ##############################################################################
        # Add your personal code below
        ##############################################################################
        pass
        ##############################################################################

# Show the canvas
plt.show()

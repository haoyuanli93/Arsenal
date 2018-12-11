import matplotlib.pyplot as plt
import numpy as np

##############################################################################
# Plot many patterns together
##############################################################################
pattern_num = 100
data = np.random.rand(100, 64, 64)

nrow = 3
ncol = 3

index_to_show = np.random.permutation(pattern_num)[: nrow * ncol]
index_to_show = index_to_show.reshape((nrow, ncol))

v_max = 10

# Set up the canvas
fig, axes = plt.subplots(nrows=nrow, ncols=ncol)
fig.set_figheight(16)
fig.set_figwidth(16)

for l in range(nrow):
    for m in range(ncol):
        # Show the polar pattern
        ax = axes[l, m]
        ax.imshow(np.flipud(data[index_to_show[l, m], :, :].T), vmax=v_max, cmap='jet')
        ax.set_title("Local index {}".format(index_to_show[l, m]))

# Show the canvas
plt.show()

##############################################################################
# Plot many patterns together
##############################################################################
"""
The following code plots a 3 by 3 grid pattern
"""
x_data = np.arange(100)
y_data = np.random.rand(100, 8)
y_data_mean = np.mean(y_data, axis=0)

# Set up the canvas
fig, axes = plt.subplots(nrows=3, ncols=3)
fig.set_figheight(16)
fig.set_figwidth(16)

# Show the average pattern
axes[0, 0].plot(x_data, y_data_mean)
axes[0, 0].set_title("Mean distribution")

# Plot position index
plot_pos = [[l, m] for l in range(3) for m in range(3)]
plot_pos.remove([0, 0])

# Plot the patterns
for l in range(8):
    axes[plot_pos[l][0], plot_pos[l][1]].plot(x_data, y_data[l])
    axes[plot_pos[l][0], plot_pos[l][1]].set_title("Index {}".format(l))

# Show the canvas
plt.show()

##############################################################################
# Plot many patterns together
##############################################################################

nrow = 3
ncol = 3

index_to_show = np.random.permutation(pattern_num)[: nrow * ncol]

# Set up the canvas
fig, axes = plt.subplots(nrows=nrow, ncols=ncol)
fig.set_figheight(16)
fig.set_figwidth(16)

# Set y range
# y_lim = (0, 0.2)

# Plot the patterns
for l in range(nrow):
    for m in range(ncol):
        ax = axes[l, m]

        idx = index_to_show[l * nrow + m]

        # Plot
        ax.plot(momentum_step, holder[idx], '-o', label='Data')
        ax.plot(momentum_step, test_func(momentum_step,
                                         intensities[idx],
                                         radius[idx],
                                         constants[idx]), label='Fitted function')

        # Set axis value
        ax.set_xticks(momentum_step[::8])
        ax.set_xticklabels(resolution_step[::8], fontsize=10)
        #    item = ax.get_yticklabels()
        #    ax.set_yticklabels(item, fontsize =15)

        # Set title
        ax.set_title("Index {}, radius {:.2f}".format(idx, radius[idx]), fontsize=15)
        ax.set_xlabel('Resolution (nm)', fontsize=15)
        ax.set_ylabel("Intensity", fontsize=15)

        # restrict
        # ax.set_ylim(y_lim)
        ax.set_xlim([0, np.max(momentum_step)])

# Show the canvas
plt.show()

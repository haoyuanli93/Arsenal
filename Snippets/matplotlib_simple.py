import matplotlib.pyplot as plt
import numpy as np

##############################################################################
# Plot many patterns with colorbar
##############################################################################
diffraction = np.ones((128, 128), dtype=np.complex128)
magnitude = np.ones((128, 128), dtype=np.float64)

# Set up the canvas
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(16)
fig.set_figwidth(20)

im = axes[0, 0].imshow(np.angle(diffraction), cmap='jet')
fig.colorbar(im, ax=axes[0, 0])
axes[0, 0].set_title("Phase")

im = axes[0, 1].imshow(magnitude, vmax=50, cmap='jet')
fig.colorbar(im, ax=axes[0, 1])
axes[0, 1].set_title("Magnitude")

im = axes[1, 0].imshow(np.abs(diffraction.real), vmax=10, cmap='jet')
fig.colorbar(im, ax=axes[1, 0])
axes[1, 0].set_title("Diffraction Real")

im = axes[1, 1].imshow(np.abs(diffraction.imag), vmax=10, cmap='jet')
fig.colorbar(im, ax=axes[1, 1])
axes[1, 1].set_title("Diffraction Complex")

# Show the canvas
plt.show()

##############################################################################
# Plot many patterns together
##############################################################################
"""

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %      Many images
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        This section consider the following case.
        One wants to shows many images together.
"""

pattern_num = 100
data = np.random.rand(100, 64, 64)

nrow = 3
ncol = 3

index_to_show = np.random.permutation(pattern_num)[: nrow * ncol]
index_to_show = index_to_show.reshape((nrow, ncol))

default_vmax_flag = False  # Whether to use the default vmax setting
static_vmax_flag = True  # Whether to use a single vmax for all images
v_max = 10

cmap = 'jet'

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
x_data_main = np.arange(100)
y_data = np.random.rand(100, 8)
y_data_mean = np.mean(y_data, axis=0)

# Set up the canvas
fig, axes = plt.subplots(nrows=3, ncols=3)
fig.set_figheight(16)
fig.set_figwidth(16)

# Show the average pattern
axes[0, 0].plot(x_data_main, y_data_mean)
axes[0, 0].set_title("Mean distribution")

# Plot position index
plot_pos = [[l, m] for l in range(3) for m in range(3)]
plot_pos.remove([0, 0])

# Plot the patterns
for l in range(8):
    axes[plot_pos[l][0], plot_pos[l][1]].plot(x_data_main, y_data[l])
    axes[plot_pos[l][0], plot_pos[l][1]].set_title("Index {}".format(l))

# Show the canvas
plt.show()

##############################################################################
# Show a pattern without boundary lines and labels
##############################################################################

pattern = np.zeros((128, 128))
threshold = 10
output_folder = '~/Downloads'

fig = plt.figure(frameon=False)
fig.set_size_inches(w=1, h=1)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

# Render the image
ax.imshow(pattern, vmax=threshold, aspect='auto', cmap='jet')

# Save the data. Notice that VGG16 only works for 224 by 224 images.
fig.savefig(output_folder + '/image.png', dpi=224)

# Close the figure
plt.close(fig)

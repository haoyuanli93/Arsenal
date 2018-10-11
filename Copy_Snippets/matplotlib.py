import matplotlib.pyplot as plt
import numpy as np

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

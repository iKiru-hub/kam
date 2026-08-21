import matplotlib.pyplot as plt
import numpy as np

print("Starting loop...")
fig, ax = plt.subplots(1, 1)

d = np.random.randn(1, 2)
im = ax.imshow(d, vmin=-2, vmax=2)
for i in range(5):
    ax.set_title(f"{i=}")
    d = np.random.randn(1, 2)
    im.set_data(d)
    print(f"Iteration {i}: title={ax.get_title()}, data={d}")

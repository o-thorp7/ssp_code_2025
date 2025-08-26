import numpy as np, matplotlib.pyplot as plt, math
from scipy.ndimage import gaussian_filter1d

# Constants
INNER_RAD = 1e-3 # r1
OUTER_RAD = 1  # r2
DISK_RES = 300 # pixels per metre
PIX_RADIUS = OUTER_RAD * DISK_RES
NUM_BINS = int(56 * PIX_RADIUS ** (1/3)) # Freedman rule
U_MAX = 5.0 / math.sqrt(OUTER_RAD)
BIN_WIDTH = 2 * U_MAX / NUM_BINS
ALPHA = 1.5

def get_u(x, y):
    r2 = x**2 + y**2
    # set u = inf for small radius so that it is excluded from the plot
    if r2 < INNER_RAD**2:
        return 0
    if r2 > OUTER_RAD**2:
        return 0
    return x * (r2 ** -0.75)

def get_brightness_shifted(x: float, y: float) -> float:
    r = math.sqrt(x**2 + y**2)
    r_dash = (r + OUTER_RAD) / OUTER_RAD
    if (r > OUTER_RAD):
        return 0.0
    
    # Option 0: smiley face :)
    # if (
    #     (x-OUTER_RAD/3)**2 + (-y-OUTER_RAD/2)**2 <= (OUTER_RAD**2)/35 or
    #     (x- 2*OUTER_RAD/3)**2 + (-y-OUTER_RAD/2)**2 <= (OUTER_RAD**2)/35 or
    #     (
    #         (-y <= OUTER_RAD/8) and 
    #         ((x-OUTER_RAD/2)**2 + (-y-OUTER_RAD/8)**2 <= (OUTER_RAD**2)/8) and
    #         ((x-OUTER_RAD/2)**2 + (-y-OUTER_RAD/8)**2 >= (OUTER_RAD**2)/10)
    #     )
    # ):
    #     return 1
    # return 0.5

    # Option 1: constant
    # return 1

    # Option 2: linear in x
    # return (x + 2*OUTER_RAD) / (4*OUTER_RAD)

    # Option 3: inverse power law:
    # return (r_dash**(-ALPHA))

    # Option 4: combination
    return (((x + 2*OUTER_RAD) / (4*OUTER_RAD)) + (r_dash**(-ALPHA)))/2

def get_brightness(x: float, y: float) -> float:
    r = math.sqrt(x**2 + y**2)
    if (r > OUTER_RAD or r < INNER_RAD):
        return 0.0
    
    # Option 1: constant
    # return 1

    # Option 2: linear in x
    # return (x + 2*OUTER_RAD) / (4*OUTER_RAD)

    # Option 3: inverse power law:
    return (r**(-ALPHA))

    # Option 4: combination
    # return (((x + 2*OUTER_RAD) / (4*OUTER_RAD)) + (r_dash**(-ALPHA)))/2


def make_brightness_grid(is_shifted: bool = False) -> list[float, float]:
    # each coordinate represents the corresponding brightness coordinate on the disk
    brightness_grid = np.zeros((PIX_RADIUS*2, PIX_RADIUS*2))

    for i in range(-PIX_RADIUS, PIX_RADIUS):
        for j in range(-PIX_RADIUS, PIX_RADIUS):
            x = j / DISK_RES
            y = i / DISK_RES
            if is_shifted:
                brightness_grid[i+PIX_RADIUS, j+PIX_RADIUS] = get_brightness_shifted(x, y)
            else:
                brightness_grid[i+PIX_RADIUS, j+PIX_RADIUS] = get_brightness(x, y)

    return brightness_grid

def make_u_grid(is_shifted: bool = False) -> list[float, float]:
    # each coordinate represents the corresponding brightness coordinate on the disk
    u_grid = np.zeros((PIX_RADIUS*2, PIX_RADIUS*2))

    for i in range(-PIX_RADIUS, PIX_RADIUS):
        for j in range(-PIX_RADIUS, PIX_RADIUS):
            x = j / DISK_RES
            y = i / DISK_RES
            u_grid[i+PIX_RADIUS, j+PIX_RADIUS] = get_u(x, y)

    return u_grid

def generate_disk_img(data: list[float, float], adjust_contrast: bool = True) -> None:
    if (adjust_contrast):
        vmin_percentile = np.percentile(data, 5)
        vmax_percentile = np.percentile(data, 95)
        plt.imshow(data, cmap='plasma', vmin=vmin_percentile, vmax=vmax_percentile)
    else:
        plt.imshow(data, cmap=plt.cm.viridis)
    plt.colorbar()
    plt.show()

def make_u_intensities(brightness_grid: list[float, float]) -> list[float]:
    bin_starts = np.linspace(-U_MAX, U_MAX, NUM_BINS, endpoint=False)
    u_intensities = np.zeros((NUM_BINS, 2))
    u_intensities[:, 0] = bin_starts
    for i in range(-PIX_RADIUS, PIX_RADIUS):
        for j in range(-PIX_RADIUS, PIX_RADIUS):
            x = j / DISK_RES
            y = i / DISK_RES
            if (x**2 + y**2 > OUTER_RAD**2):
                continue
            u = get_u(x, y)
            if abs(u) > U_MAX:
                continue
            brightness = brightness_grid[i+PIX_RADIUS, j+PIX_RADIUS]
            bin_index = int((u + U_MAX) / BIN_WIDTH)
            if bin_index < 0 or bin_index >= NUM_BINS:
                continue
            u_intensities[bin_index, 1] += brightness

    return u_intensities

def plot_u_intensities(u_intensities: list[float, float]) -> None:
    bin_starts = np.linspace(-U_MAX, U_MAX, NUM_BINS, endpoint=False)
    #TODO: remove
    # plt.xkcd()
    #
    plt.plot(bin_starts, u_intensities[:, 1])
    plt.grid(True)
    plt.show()

def flatten_spike(u_intens: list[float], n: int) -> list[float]:
    # average the middle n bins
    indices = np.arange(int((NUM_BINS-n)/2), int((NUM_BINS+n)/2), 1)
    # print(f"middle bit: {u_intens[indices,1]}")
    av = np.mean(u_intens[indices, 1])
    # print(f"indices, av: {indices}, {av}")
    flattened = np.copy(u_intens)
    for i in indices:
        flattened[i] = av
    return flattened

def perf_gauss_smooth(u_intens: list[float], sigma: float) -> list[float]:
    old_brightness = u_intens[:,1]
    new_brightness = gaussian_filter1d(old_brightness, sigma)
    smoothed_intens = np.copy(u_intens)
    smoothed_intens[:,1] = new_brightness
    return smoothed_intens

bg = make_brightness_grid(is_shifted=False)
generate_disk_img(bg, adjust_contrast=True)

# bg = make_brightness_grid(is_shifted=True)
# generate_disk_img(bg, adjust_contrast=False)


generate_disk_img(make_u_grid(), adjust_contrast=True)


u_intens = make_u_intensities(bg)
plot_u_intensities(u_intens)

# s_u_intens = perf_gauss_smooth(u_intens, sigma=5)
# plot_u_intensities(s_u_intens)

# flattened_u_i = flatten_spike(u_intens, 4)
# plot_u_intensities(flattened_u_i)
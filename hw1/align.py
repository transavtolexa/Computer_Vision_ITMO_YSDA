import numpy as np

# Read the implementation of the align_image function in pipeline.py
# to see, how these functions will be used for image alignment.
def extract_channel_plates(raw_img, crop):
    # Your code here
    H, W = raw_img.shape
    plate_h = H // 3  # Высота одной пластины

    blue = raw_img[0:plate_h, :]
    green = raw_img[plate_h:2 * plate_h, :]
    red = raw_img[2 * plate_h:3 * plate_h, :]

    blue_coord = np.array([0, 0])
    green_coord = np.array([plate_h, 0])
    red_coord = np.array([2 * plate_h, 0])

    if crop:
        crop_y = plate_h // 10
        crop_x = W // 10

        blue = blue[crop_y:plate_h - crop_y, crop_x:W - crop_x]
        green = green[crop_y:plate_h - crop_y, crop_x:W - crop_x]
        red = red[crop_y:plate_h - crop_y, crop_x:W - crop_x]

        crop_offset = np.array([crop_y, crop_x], dtype=np.int64)

        blue_coord += crop_offset
        green_coord += crop_offset
        red_coord += crop_offset

    unaligned_rgb = (np.array(red), np.array(green), np.array(blue))
    coords = (red_coord, green_coord, blue_coord)

    return unaligned_rgb, coords


def find_relative_shift_pyramid(img_a, img_b):
    # Your code here
    if np.array_equal(img_a, img_b):
        return np.array([0, 0])

    def mse(img1, img2):
        return np.mean((img1 - img2) ** 2) ## sum(array) / array.size, а в нашем случае array.size == height * width

    def ncc(img1, img2):
        numerator = np.sum(img1 * img2)
        denomenator = np.sqrt(np.sum(img1 ** 2) * np.sum(img2 ** 2))

        if denomenator == 0:
            return -np.inf

        return numerator / denomenator

    def get_overlapping_regions(img1, img2, dy, dx):
        # Для уже заданного сдвига [dy, dx] выбрать из img_a и img_b две области, 
        # содержащие соответствующие друг другу пиксели.
        # b съехало на dy, dx отн-но a: b = y + dy, x + dx
        height, width = img1.shape[0], img1.shape[1]
        if dy >= 0:
            a_y_slice = slice(0, height - dy) # Чтобы не выйти за границы массива
            b_y_slice = slice(dy, height)
        else:
            a_y_slice = slice(-dy, height)
            b_y_slice = slice(0, height + dy)

        if dx >= 0:
            a_x_slice = slice(0, width - dx)
            b_x_slice = slice(dx, width)
        else:
            a_x_slice = slice(-dx, width)
            b_x_slice = slice(0, width + dx)

        a_overlap = img1[a_y_slice, a_x_slice]
        b_overlap = img2[b_y_slice, b_x_slice]

        return a_overlap, b_overlap

    def optimum_shift(img1, img2, dy_values, dx_values):
        min_mse, best_shift = np.inf, np.array([0, 0])
        height, width = img1.shape

        for dy in dy_values:
            for dx in dx_values:
                # При таком сдвиге изображения выходим за границы
                if abs(dy) >= height or abs(dx) >= width:
                    continue

                img1_overlap, img2_overlap = get_overlapping_regions(img1, img2, dy, dx)
                msee = mse(img1_overlap, img2_overlap)

                if msee < min_mse:
                    min_mse = msee
                    best_shift = np.array([dy, dx])

        return best_shift

    def pyramid_shift(img1, img2):
        pyramid_a = [img1]
        pyramid_b = [img2]

        # Пока хотя бы одна сторона больше 500,
        # уменьшаем оба изображения в два раза.
        while max(pyramid_a[-1].shape) > 500:
            smaller_img_a = pyramid_a[-1][::2, ::2]
            smaller_img_b = pyramid_b[-1][::2, ::2]

            pyramid_a.append(smaller_img_a)
            pyramid_b.append(smaller_img_b)

        # Начинаем с последних элементов списков —
        # самых маленьких изображений.
        shift = optimum_shift(
            pyramid_a[-1],
            pyramid_b[-1],
            dy_values=range(-15, 16),
            dx_values=range(-15, 16),
        )

        # Идём от предпоследнего уровня к исходному.
        for level in range(len(pyramid_a) - 2, -1, -1):
            approximate_shift = shift * 2

            approximate_dy = int(approximate_shift[0])
            approximate_dx = int(approximate_shift[1])

            shift = optimum_shift(
                pyramid_a[level],
                pyramid_b[level],
                dy_values=range(
                    approximate_dy - 1,
                    approximate_dy + 2,
                ),
                dx_values=range(
                    approximate_dx - 1,
                    approximate_dx + 2,
                ),
            )

        return shift
    return pyramid_shift(img_a, img_b)


def find_absolute_shifts(
    crops,
    crop_coords,
    find_relative_shift_fn,
):
    # Your code here
    red, green, blue = crops
    red_coord, green_coord, blue_coord = crop_coords


    r_to_g = np.array(find_relative_shift_fn(red, green) + green_coord - red_coord)
    b_to_g = np.array(find_relative_shift_fn(blue, green) + green_coord - blue_coord)
    return r_to_g, b_to_g


def create_aligned_image(
    channels,
    channel_coords,
    r_to_g,
    b_to_g,
):
    # Your code here
    red, green, blue = channels
    red_coord, green_coord, blue_coord = channel_coords

    r_to_g_relative = r_to_g + red_coord - green_coord
    b_to_g_relative = b_to_g + blue_coord - green_coord

    r_dy, r_dx = r_to_g_relative
    b_dy, b_dx = b_to_g_relative


    height, width = green.shape

    start_y = max(0, r_dy, b_dy)
    end_y = min(height, height + r_dy, height + b_dy)

    start_x = max(0, r_dx, b_dx)
    end_x = min(width, width + r_dx, width + b_dx)

    # Вырезаем соответствующие друг другу области
    green_aligned = green[start_y:end_y, start_x:end_x]
    red_aligned = red[start_y - r_dy:end_y - r_dy, start_x - r_dx:end_x - r_dx]
    blue_aligned = blue[start_y - b_dy:end_y - b_dy, start_x - b_dx:end_x - b_dx]

    aligned_img = np.stack([red_aligned, green_aligned, blue_aligned], axis=-1)

    return aligned_img


def find_relative_shift_fourier(img_a, img_b):
    # Your code here
    a_to_b = np.array([0, 0])
    fourier_a = np.fft.fft2(img_a)
    fourier_b = np.fft.fft2(img_b)

    corr = np.fft.ifft2(np.conj(fourier_a) * fourier_b).real
    peak = np.array(np.unravel_index(np.argmax(corr), corr.shape))

    shape = np.array(corr.shape)

    # перевод координат циклического сдвига в знаковые значения
    wrapped = peak >= (shape + 1) // 2
    peak[wrapped] -= shape[wrapped]

    return peak


if __name__ == "__main__":
    import common
    import pipeline

    # Read the source image and the corresponding ground truth information
    test_path = "tests/05_unittest_align_image_pyramid_img_small_input/00"
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_path)

    # Draw the same point on each channel in the original
    # raw image using the ground truth coordinates
    visualized_img = pipeline.visualize_point(raw_img, r_point, g_point, b_point)
    common.save_image(f"gt_visualized.png", visualized_img)

    for method in ["pyramid", "fourier"]:
        # Run the whole alignment pipeline
        r_to_g, b_to_g, aligned_img = pipeline.align_image(raw_img, method)
        common.save_image(f"{method}_aligned.png", aligned_img)

        # Draw the same point on each channel in the original
        # raw image using the predicted r->g and b->g shifts
        # (Compare with gt_visualized for debugging purposes)
        r_pred = g_point - r_to_g
        b_pred = g_point - b_to_g
        visualized_img = pipeline.visualize_point(raw_img, r_pred, g_point, b_pred)

        r_error = abs(r_pred - r_point)
        b_error = abs(b_pred - b_point)
        print(f"{method}: {r_error = }, {b_error = }")

        common.save_image(f"{method}_visualized.png", visualized_img)

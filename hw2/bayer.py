import numpy as np

def get_bayer_masks(n_rows, n_cols):
    """
    :param n_rows: `int`, number of rows
    :param n_cols: `int`, number of columns

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.bool_`
        containing red, green and blue Bayer masks
    """

    red = np.array([[False, True], [False, False]])
    green = np.array([[True, False], [False, True]])
    blue = np.array([[False, False], [True, False]])

    red_mask = np.tile(red, ((n_rows + 1) // 2, (n_cols + 1) // 2))[:n_rows, :n_cols]
    green_mask = np.tile(green, ((n_rows + 1) // 2, (n_cols + 1) // 2))[:n_rows, :n_cols]
    blue_mask = np.tile(blue, ((n_rows + 1) // 2, (n_cols + 1) // 2))[:n_rows, :n_cols]

    return np.stack([red_mask, green_mask, blue_mask], axis=-1)


def get_colored_img(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        each channel contains known color values or zeros
        depending on Bayer masks
    """
    bayer_masks = get_bayer_masks(*raw_img.shape)
    colored_img = raw_img[..., np.newaxis] * bayer_masks

    return colored_img


def get_raw_img(colored_img):
    """
    :param colored_img:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        colored image

    :return:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image as captured by camera
    """
    masks = get_bayer_masks(*colored_img.shape[:2])

    return np.sum(colored_img * masks, axis=-1)


def bilinear_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)`, and dtype `np.uint8`,
        result of bilinear interpolation
    """
    n_rows, n_cols = raw_img.shape
    masks = get_bayer_masks(n_rows, n_cols)
    colored_img = get_colored_img(raw_img)
    colored_img_values = colored_img.astype(np.float32)

    inner_shape = (n_rows - 2, n_cols - 2, 3)

    channel_sums = np.zeros(inner_shape, dtype=np.float32)
    channel_counts = np.zeros(inner_shape, dtype=np.uint8)

    for window_row in range(3):
        for window_col in range(3):
            channel_sums += colored_img_values[window_row:window_row + n_rows - 2, window_col:window_col + n_cols - 2]
            channel_counts += masks[window_row:window_row + n_rows - 2, window_col:window_col + n_cols - 2]

    pixel_filled = channel_sums / channel_counts

    result = colored_img_values.copy()
    result[1:-1, 1:-1] = np.where(
        masks[1:-1, 1:-1],
        colored_img_values[1:-1, 1:-1],
        pixel_filled,
    )

    return result.astype(np.uint8)

def improved_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        result of improved interpolation
    """
    n_rows, n_cols = raw_img.shape
    raw_values = raw_img.astype(np.float32)
    masks = get_bayer_masks(n_rows, n_cols)
    result = raw_values[..., np.newaxis] * masks

    # Фильтр для восстановления G в позициях R и B.
    g_filter = np.array([
        [ 0, 0, -1, 0,  0],
        [ 0, 0,  2, 0,  0],
        [-1, 2,  4, 2, -1],
        [ 0, 0,  2, 0,  0],
        [ 0, 0, -1, 0,  0],
    ], dtype=np.float32)

    # Фильтр для восстановления цвета в зелёной позиции,
    # когда известные значения нужного цвета находятся
    # преимущественно слева и справа.
    hor_filter = np.array([
        [ 0,  0, 0.5,  0,  0],
        [ 0, -1,   0, -1,  0],
        [-1,  4,   5,  4, -1],
        [ 0, -1,   0, -1,  0],
        [ 0,  0, 0.5,  0,  0],
    ], dtype=np.float32)

    # Фильтр для восстановления R в позиции B
    # или B в позиции R.
    opp_filter = np.array([
        [   0, 0, -1.5, 0,    0],
        [   0, 2,    0, 2,    0],
        [-1.5, 0,    6, 0, -1.5],
        [   0, 2,    0, 2,    0],
        [   0, 0, -1.5, 0,    0],
    ], dtype=np.float32)

    # Нормализация фильтров
    g_filter /= g_filter.sum()
    hor_filter /= hor_filter.sum()
    opp_filter /= opp_filter.sum()

    # Так как вертикальный случай симметричен горизонтальному, то достаточно транспонировать матрицу коэффициентов
    ver_filter = hor_filter.T

    def apply_filter(kernel):
        """
        Применяет один фильтр 5 × 5 сразу ко всем внутренним
        пикселям изображения.

        Возвращаемая форма:
            (n_rows - 4, n_cols - 4)
        """
        filtered = np.zeros(
            (n_rows - 4, n_cols - 4),
            dtype=np.float32,
        )

        for window_row in range(5):
            for window_col in range(5):
                coefficient = kernel[window_row, window_col]

                if coefficient != 0:
                    filtered += coefficient * raw_values[
                        window_row:window_row + n_rows - 4,
                        window_col:window_col + n_cols - 4,
                    ]

        return filtered

    green_estimate = apply_filter(g_filter)
    horizontal_estimate = apply_filter(hor_filter)
    vertical_estimate = apply_filter(ver_filter)
    opposite_estimate = apply_filter(opp_filter)

    inner = result[2:-2, 2:-2]

    inner[0::2, 1::2, 1] = green_estimate[0::2, 1::2]
    inner[0::2, 1::2, 2] = opposite_estimate[0::2, 1::2]
    inner[1::2, 0::2, 0] = opposite_estimate[1::2, 0::2]
    inner[1::2, 0::2, 1] = green_estimate[1::2, 0::2]
    inner[0::2, 0::2, 0] = horizontal_estimate[0::2, 0::2]
    inner[0::2, 0::2, 2] = vertical_estimate[0::2, 0::2]
    inner[1::2, 1::2, 0] = vertical_estimate[1::2, 1::2]
    inner[1::2, 1::2, 2] = horizontal_estimate[1::2, 1::2]

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)


def compute_psnr(img_pred, img_gt):
    """
    :param img_pred:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        predicted image
    :param img_gt:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        ground truth image

    :return:
        `float`, PSNR metric
    """
    pred = img_pred.astype(np.float32)
    gt = img_gt.astype(np.float32)

    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        raise ValueError("MSE is zero")

    max_gt = np.max(gt)
    psnr = 10 * np.log10(max_gt ** 2 / mse)

    return float(psnr)


if __name__ == "__main__":
    import common

    for img_name, img_raw, img_gt in common.get_test_images():
        print(f"Processing {img_name}...")
        img_bilinear = bilinear_interpolation(img_raw)
        img_improved = improved_interpolation(img_raw)
        if img_bilinear is not None:
            print("PSNR (bilinear):", compute_psnr(img_bilinear, img_gt))
            common.save_image(f"imgs_bilinear/{img_name}", img_bilinear)
        if img_improved is not None:
            print("PSNR (improved):", compute_psnr(img_improved, img_gt))
            common.save_image(f"imgs_improved/{img_name}", img_improved)
        print()

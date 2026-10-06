from numpy import array, zeros

from bayer import bilinear_interpolation
from common import assert_ndarray_equal


def test_bilinear_interpolation_1():
    raw_img = array(
        [
            [1, 2, 3],
            [8, 9, 4],
            [7, 6, 5],
        ],
        dtype="uint8",
    )

    img = bilinear_interpolation(raw_img)
    pr = img[1, 1]
    gt = array([4, 9, 6], dtype="uint8")
    assert_ndarray_equal(
        actual=pr,
        correct=gt,
        atol=1,
    )


def test_bilinear_interpolation_2():
    raw_img = array(
        [
            [1, 2, 3, 4],
            [4, 5, 6, 5],
            [7, 8, 9, 6],
            [7, 8, 9, 10],
        ],
        dtype="uint8",
    )
    gt_img = zeros((4, 4, 3), "uint8")
    r = slice(1, -1), slice(1, -1)
    gt_img[r + (0,)] = array(
        [
            [5, 5],
            [8, 7],
        ]
    )
    gt_img[r + (1,)] = array(
        [
            [5, 5],
            [7, 9],
        ]
    )
    gt_img[r + (2,)] = array(
        [
            [5, 6],
            [6, 7],
        ]
    )

    img = bilinear_interpolation(raw_img)
    assert_ndarray_equal(
        actual=img[r],
        correct=gt_img[r],
        atol=1,
    )


def test_bilinear_interpolation_3():
    raw_img = array(
        [
            # fmt: off
            [202, 150, 137, 121, 195],
            [ 94,   0, 217,  68, 248],
            [208, 170, 109,  67,  22],
            [ 20,  93, 222,  54,  50],
            [254, 252,  10, 187, 203],
            # fmt: on
        ],
        dtype="uint8",
    )
    gt_img = zeros((5, 5, 3), "uint8")
    r = slice(1, -1), slice(1, -1)
    gt_img[r + (0,)] = array(
        [
            [160, 127, 94],
            [170, 118, 67],
            [211, 169, 127],
        ]
    )
    gt_img[r + (1,)] = array(
        [
            # fmt: off
            [  0,  78, 68],
            [102, 109, 63],
            [ 93,  66, 54],
            # fmt: on
        ]
    )
    gt_img[r + (2,)] = array(
        [
            [155, 217, 232],
            [138, 219, 184],
            [121, 222, 136],
        ]
    )

    img = bilinear_interpolation(raw_img)
    assert_ndarray_equal(
        actual=img[r],
        correct=gt_img[r],
        atol=1,
    )


def test_bilinear_interpolation_4():
    raw_img = array(
        [
            # fmt: off
            [141, 160, 172, 165, 168],
            [ 65,  99,  91, 132, 107],
            [115, 112,  93, 129,  99],
            [ 69,  88,  80, 109,  82],
            [ 72,  88,  73, 113,  81],
            [ 39,  36,  25,  28,  29],
            # fmt: on
        ],
        dtype="uint8",
    )
    gt_img = zeros((6, 5, 3), "uint8")
    r = slice(1, -1), slice(1, -1)
    gt_img[r + (0,)] = array(
        [
            [136, 141, 147],
            [112, 120, 129],
            [100, 110, 121],
            [88, 100, 113],
        ]
    )
    gt_img[r + (1,)] = array(
        [
            [99, 124, 132],
            [98, 93, 108],
            [88, 90, 109],
            [67, 73, 72],
        ]
    )
    gt_img[r + (2,)] = array(
        [
            [78, 91, 99],
            [76, 85, 90],
            [74, 80, 81],
            [53, 52, 54],
        ]
    )

    img = bilinear_interpolation(raw_img)
    assert_ndarray_equal(
        actual=img[r],
        correct=gt_img[r],
        atol=1,
    )

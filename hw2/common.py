import json
import os
import pathlib

import numpy as np
import PIL.Image


def read_image(img_path, mode):
    error = (
        "Couldn't read the {!r} image required for tests. "
        "Did you completely extract the test archive?"
    )

    try:
        img = PIL.Image.open(img_path)
        assert img.mode == mode
        return np.asarray(img)
    except (AssertionError, OSError) as err:
        assert False, error.format(str(img_path))


def get_test_images(test=None):
    raw_root = pathlib.Path("imgs_raw").resolve()
    if test is None:
        gt_root = pathlib.Path("imgs_gt").resolve()
    else:
        gt_root = pathlib.Path(test).resolve().parent

    for idx in range(10):
        name = f"{idx + 1:02}.png"
        raw_img = read_image(raw_root / name, "L")
        gt_img = read_image(gt_root / name, "RGB")
        yield f"{idx:02}.png", raw_img, gt_img


def save_image(out_path, img_u8):
    if img_u8 is None:
        print(f"No image data provided for {out_path!r}, skipping...")
        return

    pathlib.Path(out_path).resolve().parent.mkdir(exist_ok=True)
    PIL.Image.fromarray(img_u8).save(out_path)


def assert_value_is_ndarray(value):
    __tracebackhide__ = True
    error = f"Value should be an instance of np.ndarray, but it is {type(value)}."
    assert isinstance(value, (np.ndarray, np.generic)), error


def assert_dtypes_compatible(actual_dtype, correct_dtype):
    __tracebackhide__ = True
    error = (
        "The dtypes of actual value and correct value are not the same "
        "and can't be safely converted.\n"
        f"actual.dtype={actual_dtype}, correct.dtype={correct_dtype}"
    )
    assert np.can_cast(actual_dtype, correct_dtype, casting="same_kind"), error
    assert np.can_cast(correct_dtype, actual_dtype, casting="same_kind"), error


def assert_shapes_match(actual_shape, correct_shape):
    __tracebackhide__ = True
    error = (
        "The shapes of actual value and correct value are not the same.\n"
        f"actual.shape={actual_shape}, correct.shape={correct_shape}"
    )
    assert len(actual_shape) == len(correct_shape), error
    assert actual_shape == correct_shape, error


def assert_ndarray_equal(*, actual, correct, rtol=0, atol=1e-6, err_msg=""):
    __tracebackhide__ = True
    assert_value_is_ndarray(actual)
    assert_dtypes_compatible(actual.dtype, correct.dtype)
    assert_shapes_match(actual.shape, correct.shape)
    np.testing.assert_allclose(
        actual,
        correct,
        atol=atol,
        rtol=rtol,
        verbose=True,
        err_msg=err_msg,
    )


def assert_time_limit(*, actual, limit):
    __tracebackhide__ = True
    filename = "calibration.json"

    if os.environ.get("CHECKER"):
        extra = ""
    else:
        try:
            with open(filename, "r") as f:
                coefficient = json.load(f)["coefficient"]

            extra = f" (based on server time limit of {limit:.2f}s)"
            limit /= coefficient
        except (OSError, ValueError, ZeroDivisionError):
            assert False, (
                "Couldn't read the local calibration coefficient. "
                "Did you forget to run the calibrate.py script?"
            )

    assert actual <= limit, (
        "This test must complete within the specified time limit.\n"
        f"actual_time={actual:.2f}s, time_limit={limit:.2f}s{extra}"
    )

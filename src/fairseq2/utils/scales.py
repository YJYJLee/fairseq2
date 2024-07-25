# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Optional
import math

def slice_str_to_array(slice_str: str, length: int):
    """
    Converts a slice string to a boolean array where True indicates the index is included in the slice.
    :param slice_str:
        A string representing a slice. The format should be "start:end:step", where
        each part is optional. Examples include "1:5", ":5", "::2".
    :param length:
        The length of the resulting boolean array. This is the total number of elements
        in the array, and should be a non-negative integer.
    :return:
        A list of boolean values where each element is True if its index falls within the specified slice.
    :raises ValueError:
        If any part of `slice_str` is not convertible to an integer.
    Examples:
    >>> slice_str_to_array("1:5", 10)
    [False, True, True, True, True, False, False, False, False, False]
    >>> slice_str_to_array("::2", 5)
    [True, False, True, False, True]
    >>> slice_str_to_array("3:", 5)
    [False, False, False, True, True]
    """
    # Parse the slice string
    parts = slice_str.split(':')
    start, end, step = None, None, None

    if len(parts) == 1 and parts[0] != '':
        start = int(parts[0])
    elif len(parts) == 2:
        start = int(parts[0]) if parts[0] != '' else None
        end = int(parts[1]) if parts[1] != '' else None
    elif len(parts) == 3:
        start = int(parts[0]) if parts[0] != '' else None
        end = int(parts[1]) if parts[1] != '' else None
        step = int(parts[2]) if parts[2] != '' else None

    # Create a boolean array based on the slice
    result = [False] * length
    slice_indices = range(start if start is not None else 0,
                          end if end is not None else length,
                          step if step is not None else 1)

    for i in slice_indices:
        if 0 <= i < length:
            result[i] = True

    return result

class ScaleType(str, Enum):
    UNIFORM = "uniform"
    EXP = "exp"
    LINEAR = "linear"
    LOG = "log"
    SIN = "sin"
    SIGMOID = "sigmoid"
    STEP = "step"

def get_scale(scale_type: ScaleType, scale_period: int, idx: int):
    """
    Calculates a scaling factor based on the specified scale type, scale period, and value.
    :param scale_type:
        A member of the :class:`ScaleType` enum that specifies the type of scaling to apply.
    :param scale_period:
        An integer representing the period over which the scaling is applied. This is used
        as the denominator in scaling calculations to normalize the `val`.
    :param idx:
        An integer representing the current index for which the scaling factor is calculated.
        This value should be within the range [0, scale_period].
    :return:
        A float representing the scaling factor. This factor is calculated based on the `scale_type`.
        The scaling factor is designed to be 0 when `val` is 0 and 1 when `val` is `scale_period`,
        except for `ScaleType.UNIFORM` where it is always 1.
    :raises ValueError:
        If `scale_period` is 0, as division by zero in scaling calculations is not allowed.
    Examples:
    >>> get_scale(ScaleType.LINEAR, 10, 5)
    0.5
    >>> get_scale(ScaleType.EXP, 10, 3)
    0.2362900883445226
    >>> get_scale(ScaleType.LOG, 10, 2)
    0.3562071871080222
    >>> get_scale(ScaleType.SIN, 10, 5)
    1.0
    >>> get_scale(ScaleType.SIGMOID, 10, 5)
    0.5
    """
    if scale_period == 0:
        return 1

    # all the equations below aim to make scale = 0 when val=0, and scale = 1 when val=scale_period
    return {
        ScaleType.UNIFORM: 1,
        ScaleType.EXP: math.exp(idx * math.log(2) / scale_period) - 1,
        ScaleType.LINEAR: idx / scale_period,
        ScaleType.LOG: math.log(idx + 1) / math.log(scale_period + 1),
        ScaleType.SIN: math.sin(0.5 * math.pi * idx / scale_period),
        ScaleType.SIGMOID: 1 / (1 + math.exp(-10 * (idx / scale_period - 0.5))),
    }[scale_type]

def get_values(scale_type: ScaleType, scale_period: int, max_val: float= 0.0, slice_str: Optional[str] = None):
    """
    Generates a list of values scaled according to the specified scale type and period, optionally filtered by a slice string.
    :param scale_type:
        A member of the :class:`ScaleType` enum that specifies the type of scaling to apply.
    :param scale_period:
        An integer representing the period over which the scaling is applied. This is used
        to determine the number of values in the result list.
    :param max_val:
        A float representing the maximum possible value in the result list. Defaults to 0.0.
    :param slice_str:
        An optional string representing a slice of indices to include in the scaling. If provided,
        only indices that fall within the slice are scaled; others are set to 0.0. If None, all
        indices are included. Defaults to None.
    :return:
        A list of floats where each element is a scaled value based on `scale_type`. The scaling
        is applied only to indices specified by `slice_str`, and all values are guaranteed to be
        between 0 and `max_val`.
    :raises AssertionError:
        If any calculated value is not within the range [0, `max_val`].
    Examples:
    >>> get_values(ScaleType.LINEAR, 5, 10)
    [0.0, 2.5, 5.0, 7.5, 10.0]
    >>> get_values(ScaleType.EXP, 5, 10, "1:3")
    [0.0, 2.371373705661655, 4.894348370484656, 0.0, 0.0]
    >>> get_values(ScaleType.LOG, 5, 10, "0:5:2")
    [0.0, 0.0, 5.0, 0.0, 10.0]
    """
    vals = []
    has_val = slice_str_to_array(slice_str, scale_period) if slice_str else [True] * scale_period

    for idx in range(scale_period):
        val = max_val * get_scale(
            scale_type = scale_type,
            scale_period = scale_period - 1,
            idx = idx,
        ) if has_val[idx] else 0.0
        assert val >= 0.0 and val <= max_val, f"val={val} should be between 0 and {max_val}"
        vals.append(val)

    return vals

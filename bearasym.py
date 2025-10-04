"""Utilities for bearing matrices in asymmetrical rotor models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class BearingModel:
    """Simple container mirroring the MATLAB ``model`` structure.

    Parameters
    ----------
    node:
        Array-like describing the model nodes. Only the number of nodes is
        required by :func:`bearasym`.
    bearing:
        Array-like describing the bearings. Each row corresponds to one
        bearing definition and follows the conventions from the original
        MATLAB implementation.
    """

    node: Sequence[Sequence[float]]
    bearing: Sequence[Sequence[float]]


def _as_array(data: Iterable[Iterable[float]]) -> np.ndarray:
    """Convert ``data`` to a 2-D NumPy array."""

    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def bearasym(model: BearingModel | object):
    """Replicate the MATLAB ``bearasym`` function in Python.

    Parameters
    ----------
    model:
        Object with ``node`` and ``bearing`` attributes (or keys) describing
        the rotor model.

    Returns
    -------
    Cb, Kb, K1b, zero_dof:
        Bearing damping matrix, stiffness matrix, cross-coupled stiffness
        matrix, and a list of constrained degrees of freedom, respectively.
    """

    node_def = getattr(model, "node", None)
    if node_def is None and isinstance(model, dict):
        node_def = model["node"]
    bearing_def = getattr(model, "bearing", None)
    if bearing_def is None and isinstance(model, dict):
        bearing_def = model["bearing"]

    node_def = _as_array(node_def)
    bearing_def = _as_array(bearing_def)

    no_node = node_def.shape[0]
    ndof = 4 * no_node
    nbearing, ncol_bearing = bearing_def.shape

    Cb = np.zeros((ndof, ndof))
    Kb = np.zeros((ndof, ndof))
    K1b = np.zeros((ndof, ndof))
    zero_dof: list[int] = []

    for i in range(nbearing):
        bearing_type = int(np.rint(bearing_def[i, 0]))
        if bearing_type < 1 or bearing_type > 4:
            print(
                f">>>> Error - bearing type {bearing_type} not implemented - bearing ignored"
            )
            continue

        Kb1 = np.zeros((4, 4))
        K1b1 = np.zeros((4, 4))
        Cb1 = np.zeros((4, 4))

        if bearing_type == 1:
            n1 = int(bearing_def[i, 1])
            zero_dof.extend([4 * n1 - 3, 4 * n1 - 2])

        if bearing_type == 2:
            n1 = int(bearing_def[i, 1])
            zero_dof.extend(range(4 * n1 - 3, 4 * n1 + 1))

        if bearing_type == 3:
            if ncol_bearing < 6:
                raise ValueError(
                    ">>>> Error - too few columns in bearing definition matrix for bearing type 3"
                )
            if bearing_def[i, 2] != bearing_def[i, 3]:
                raise ValueError(">>>> Error - bearings must be symmetric")
            if bearing_def[i, 4] != bearing_def[i, 5]:
                raise ValueError(">>>> Error - bearings must be symmetric")

            Kb1 = np.diag([bearing_def[i, 2], bearing_def[i, 3], 0.0, 0.0])
            Cb1 = np.diag([bearing_def[i, 4], bearing_def[i, 5], 0.0, 0.0])
            K1b1[0, 1] = -bearing_def[i, 4]
            K1b1[1, 0] = bearing_def[i, 4]

        if bearing_type == 4:
            if ncol_bearing < 10:
                raise ValueError(
                    ">>>> Error - too few columns in bearing definition matrix for bearing type 4"
                )
            if bearing_def[i, 2] != bearing_def[i, 3]:
                raise ValueError(">>>> Error - bearings must be symmetric")
            if bearing_def[i, 4] != bearing_def[i, 5]:
                raise ValueError(">>>> Error - bearings must be symmetric")
            if bearing_def[i, 6] != bearing_def[i, 7]:
                raise ValueError(">>>> Error - bearings must be symmetric")
            if bearing_def[i, 8] != bearing_def[i, 9]:
                raise ValueError(">>>> Error - bearings must be symmetric")

            Kb1 = np.diag(
                [
                    bearing_def[i, 2],
                    bearing_def[i, 3],
                    bearing_def[i, 4],
                    bearing_def[i, 5],
                ]
            )
            Cb1 = np.diag(
                [
                    bearing_def[i, 6],
                    bearing_def[i, 7],
                    bearing_def[i, 8],
                    bearing_def[i, 9],
                ]
            )
            K1b1[0, 1] = -bearing_def[i, 6]
            K1b1[1, 0] = bearing_def[i, 6]
            K1b1[2, 3] = -bearing_def[i, 8]
            K1b1[1, 0] = bearing_def[i, 8]

        nnode = int(bearing_def[i, 1])
        start = 4 * (nnode - 1)
        end = start + 4

        Kb[start:end, start:end] += Kb1
        Cb[start:end, start:end] += Cb1
        K1b[start:end, start:end] += K1b1

    return Cb, Kb, K1b, zero_dof

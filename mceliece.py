#!/usr/bin/env python3
"""
McEliece command-line interface.

Usage:
  mceliece.py [options] enc PUB_KEY_FILE [FILE]
  mceliece.py [options] dec PRIV_KEY_FILE [FILE]
  mceliece.py [options] gen M N T PRIV_KEY_FILE PUB_KEY_FILE
  mceliece.py (-h | --help)
  mceliece.py --version

Options:
  -b, --block        Interpret input/output as a block stream.
  -i, --poly-input   Interpret input as an integer-array literal.
  -o, --poly-output  Print output as an integer array.
  -h, --help         Show this screen.
  --version          Show version.
  -d, --debug        Debug mode.
  -v, --verbose      Verbose mode.

Phase 2C key format
-------------------

Private key:

    format_version
    m, n, t, k
    G
    H
    R
    right_inverse_pivots
    S
    S_inv
    P
    P_inv
    g_poly
    irr_poly

Public key:

    format_version
    m, n, t, k
    Gp

The secret generator right inverse satisfies

    G R = I_k.

The right inverse has dimensions

    R : n x k.

The vector

    right_inverse_pivots : shape (k,)

contains the k independent columns of G used to construct R.

Binary matrices are serialized as primitive uint8 NumPy arrays.

The secret McEliece permutation is serialized as

    P     : int64[n]
    P_inv : int64[n]

rather than dense n x n permutation matrices.

No Python objects are stored in the NPZ files. Keys are therefore
loaded using

    allow_pickle=False.
"""

import ast
import logging
import sys

import numpy as np
from docopt import docopt

from mceliece.mathutils import GF2Matrix
from mceliece.mceliececipher import McElieceCipher
from padding.padding import (
    padding_decode,
    padding_encode,
)


log = logging.getLogger("mceliece")


# ================================================================
# Key-format version
# ================================================================

KEY_FORMAT_VERSION = 3


# ================================================================
# GF(2) serialization helpers
# ================================================================

def gf2_matrix_to_uint8(matrix):
    """
    Convert a GF2Matrix into an ordinary uint8 NumPy array while
    preserving its shape.
    """

    if not isinstance(
        matrix,
        GF2Matrix,
    ):
        raise TypeError(
            "gf2_matrix_to_uint8 expects GF2Matrix."
        )

    return np.array(
        [
            int(entry.n)
            for entry in matrix.arr.flat
        ],
        dtype=np.uint8,
    ).reshape(
        matrix.arr.shape
    )


def uint8_to_gf2_matrix(array):
    """
    Convert a binary NumPy array into GF2Matrix.
    """

    array = np.asarray(
        array,
        dtype=np.uint8,
    )

    if not np.all(
        (array == 0)
        |
        (array == 1)
    ):
        raise ValueError(
            "Serialized GF(2) matrix contains "
            "entries outside {0,1}."
        )

    return GF2Matrix.from_list(
        array
    )


def gf2_matrix_equal(
    A,
    B,
):
    """
    Compare two GF2Matrix objects.
    """

    if not isinstance(
        A,
        GF2Matrix,
    ):
        raise TypeError(
            "A must be GF2Matrix."
        )

    if not isinstance(
        B,
        GF2Matrix,
    ):
        raise TypeError(
            "B must be GF2Matrix."
        )

    if A.arr.shape != B.arr.shape:
        return False

    return np.array_equal(
        gf2_matrix_to_uint8(
            A
        ),
        gf2_matrix_to_uint8(
            B
        ),
    )


def gf2_identity(size):
    """
    Construct I_size over GF(2).
    """

    return GF2Matrix.from_list(
        np.eye(
            size,
            dtype=np.uint8,
        )
    )


# ================================================================
# Permutation helpers
# ================================================================

def validate_permutation(
    permutation,
    n,
    name,
):
    """
    Validate a length-n permutation vector.
    """

    permutation = np.asarray(
        permutation,
        dtype=np.int64,
    ).reshape(-1)

    if permutation.shape != (
        n,
    ):
        raise ValueError(
            f"{name} must have shape ({n},), "
            f"received {permutation.shape}."
        )

    identity = np.arange(
        n,
        dtype=np.int64,
    )

    if not np.array_equal(
        np.sort(
            permutation
        ),
        identity,
    ):
        raise ValueError(
            f"{name} is not a valid permutation "
            f"of 0,...,{n - 1}."
        )

    return permutation


def validate_permutation_pair(
    P,
    P_inv,
    n,
):
    """
    Validate P and P_inv and verify

        P[P_inv] = identity,

        P_inv[P] = identity.
    """

    P = validate_permutation(
        P,
        n,
        "P",
    )

    P_inv = validate_permutation(
        P_inv,
        n,
        "P_inv",
    )

    identity = np.arange(
        n,
        dtype=np.int64,
    )

    if not np.array_equal(
        P[
            P_inv
        ],
        identity,
    ):
        raise ValueError(
            "P[P_inv] != identity."
        )

    if not np.array_equal(
        P_inv[
            P
        ],
        identity,
    ):
        raise ValueError(
            "P_inv[P] != identity."
        )

    return (
        P,
        P_inv,
    )


# ================================================================
# Phase 2C right-inverse validation
# ================================================================

def validate_right_inverse_pivots(
    pivots,
    n,
    k,
):
    """
    Validate the pivot indices associated with the sparse right
    inverse construction.

    The vector must contain k distinct indices from

        {0, ..., n-1}.
    """

    pivots = np.asarray(
        pivots,
        dtype=np.int64,
    ).reshape(-1)

    if pivots.shape != (
        k,
    ):
        raise ValueError(
            "right_inverse_pivots must have "
            f"shape ({k},), "
            f"received {pivots.shape}."
        )

    if np.any(
        pivots < 0
    ):
        raise ValueError(
            "right_inverse_pivots contains "
            "a negative index."
        )

    if np.any(
        pivots >= n
    ):
        raise ValueError(
            "right_inverse_pivots contains "
            "an index outside the code length."
        )

    if len(
        np.unique(
            pivots
        )
    ) != k:
        raise ValueError(
            "right_inverse_pivots must contain "
            "k distinct indices."
        )

    return pivots


def validate_right_inverse(
    G,
    R,
    pivots,
):
    """
    Validate a serialized Phase 2C right inverse.

    Required properties:

        G in GF(2)^{k x n},

        R in GF(2)^{n x k},

        G R = I_k.

    For the current sparse construction, rows outside the selected
    pivot positions are required to be zero.

    Parameters
    ----------
    G : GF2Matrix
        Generator matrix.

    R : GF2Matrix
        Proposed right inverse.

    pivots : array-like
        Selected independent columns of G.

    Returns
    -------
    numpy.ndarray
        Validated pivot vector.
    """

    if not isinstance(
        G,
        GF2Matrix,
    ):
        raise TypeError(
            "G must be GF2Matrix."
        )

    if not isinstance(
        R,
        GF2Matrix,
    ):
        raise TypeError(
            "R must be GF2Matrix."
        )

    if G.arr.ndim != 2:
        raise ValueError(
            "G must be two-dimensional."
        )

    if R.arr.ndim != 2:
        raise ValueError(
            "R must be two-dimensional."
        )

    k, n = G.arr.shape

    if R.arr.shape != (
        n,
        k,
    ):
        raise ValueError(
            f"R has shape {R.arr.shape}; "
            f"expected ({n},{k})."
        )

    pivots = validate_right_inverse_pivots(
        pivots,
        n,
        k,
    )

    # ------------------------------------------------------------
    # Verify G R = I_k.
    # ------------------------------------------------------------

    GR = (
        G
        * R
    )

    identity = gf2_identity(
        k
    )

    if not gf2_matrix_equal(
        GR,
        identity,
    ):
        raise ValueError(
            "Invalid right inverse: G R != I_k."
        )

    # ------------------------------------------------------------
    # Verify sparse-selector representation.
    #
    # Rows outside the pivot positions should be zero.
    # ------------------------------------------------------------

    R_numpy = gf2_matrix_to_uint8(
        R
    )

    nonpivot_mask = np.ones(
        n,
        dtype=bool,
    )

    nonpivot_mask[
        pivots
    ] = False

    if np.any(
        R_numpy[
            nonpivot_mask,
            :
        ]
    ):
        raise ValueError(
            "Serialized R contains nonzero rows "
            "outside right_inverse_pivots."
        )

    # ------------------------------------------------------------
    # Verify the pivot submatrix relation directly:
    #
    #     G[:,J] R[J,:] = I_k.
    # ------------------------------------------------------------

    B = GF2Matrix(
        G.arr[
            :,
            pivots
        ]
    )

    R_selected = GF2Matrix(
        R.arr[
            pivots,
            :
        ]
    )

    local_identity = (
        B
        * R_selected
    )

    if not gf2_matrix_equal(
        local_identity,
        identity,
    ):
        raise ValueError(
            "Pivot representation is inconsistent: "
            "G[:,pivots] R[pivots,:] != I_k."
        )

    return pivots


# ================================================================
# Key generation and serialization
# ================================================================

def generate(
    m,
    n,
    t,
    priv_key_file,
    pub_key_file,
):
    """
    Generate a McEliece key pair and serialize it using primitive
    NumPy numerical arrays.

    Phase 2C serializes the precomputed generator right inverse R.
    """

    m = int(
        m
    )

    n = int(
        n
    )

    t = int(
        t
    )

    mc = McElieceCipher(
        m,
        n,
        t,
    )

    mc.generate_random_keys()

    k = int(
        mc.k
    )

    # ------------------------------------------------------------
    # Validate Phase 2B permutation representation.
    # ------------------------------------------------------------

    (
        P,
        P_inv,
    ) = validate_permutation_pair(
        mc.P,
        mc.P_inv,
        n,
    )

    # ------------------------------------------------------------
    # Validate Phase 2C right inverse.
    # ------------------------------------------------------------

    if mc.R is None:
        raise RuntimeError(
            "Key generation did not construct R."
        )

    if mc.right_inverse_pivots is None:
        raise RuntimeError(
            "Key generation did not construct "
            "right_inverse_pivots."
        )

    validated_pivots = (
        validate_right_inverse(
            mc.G,
            mc.R,
            mc.right_inverse_pivots,
        )
    )

    # ------------------------------------------------------------
    # Convert GF(2) objects to primitive uint8 arrays.
    # ------------------------------------------------------------

    G = gf2_matrix_to_uint8(
        mc.G
    )

    H = gf2_matrix_to_uint8(
        mc.H
    )

    R = gf2_matrix_to_uint8(
        mc.R
    )

    S = gf2_matrix_to_uint8(
        mc.S
    )

    S_inv = gf2_matrix_to_uint8(
        mc.S_inv
    )

    Gp = gf2_matrix_to_uint8(
        mc.Gp
    )

    g_poly = np.asarray(
        mc.g_poly,
        dtype=np.uint8,
    )

    irr_poly = np.asarray(
        mc.irr_poly,
        dtype=np.uint8,
    )

    # ------------------------------------------------------------
    # Private key
    # ------------------------------------------------------------

    np.savez_compressed(
        priv_key_file,

        format_version=np.array(
            KEY_FORMAT_VERSION,
            dtype=np.int64,
        ),

        m=np.array(
            m,
            dtype=np.int64,
        ),

        n=np.array(
            n,
            dtype=np.int64,
        ),

        t=np.array(
            t,
            dtype=np.int64,
        ),

        k=np.array(
            k,
            dtype=np.int64,
        ),

        G=G,

        H=H,

        R=R,

        right_inverse_pivots=(
            validated_pivots.astype(
                np.int64
            )
        ),

        S=S,

        S_inv=S_inv,

        P=P.astype(
            np.int64
        ),

        P_inv=P_inv.astype(
            np.int64
        ),

        g_poly=g_poly,

        irr_poly=irr_poly,
    )

    log.info(
        "Private key saved to %s",
        priv_key_file,
    )

    # ------------------------------------------------------------
    # Public key
    # ------------------------------------------------------------

    np.savez_compressed(
        pub_key_file,

        format_version=np.array(
            KEY_FORMAT_VERSION,
            dtype=np.int64,
        ),

        m=np.array(
            m,
            dtype=np.int64,
        ),

        n=np.array(
            n,
            dtype=np.int64,
        ),

        t=np.array(
            t,
            dtype=np.int64,
        ),

        k=np.array(
            k,
            dtype=np.int64,
        ),

        Gp=Gp,
    )

    log.info(
        "Public key saved to %s",
        pub_key_file,
    )

    return mc


# ================================================================
# Public-key loading
# ================================================================

def load_public_key(
    pub_key_file,
):
    """
    Load a Phase 2C public key using allow_pickle=False.
    """

    with np.load(
        pub_key_file,
        allow_pickle=False,
    ) as key:

        required = {
            "format_version",
            "m",
            "n",
            "t",
            "k",
            "Gp",
        }

        missing = (
            required
            - set(
                key.files
            )
        )

        if missing:
            raise ValueError(
                "Public key is missing fields: "
                f"{sorted(missing)}"
            )

        version = int(
            np.asarray(
                key[
                    "format_version"
                ]
            ).item()
        )

        if version != KEY_FORMAT_VERSION:
            raise ValueError(
                "Unsupported public-key format "
                f"version {version}; "
                f"expected {KEY_FORMAT_VERSION}."
            )

        m = int(
            np.asarray(
                key["m"]
            ).item()
        )

        n = int(
            np.asarray(
                key["n"]
            ).item()
        )

        t = int(
            np.asarray(
                key["t"]
            ).item()
        )

        k = int(
            np.asarray(
                key["k"]
            ).item()
        )

        Gp_array = np.asarray(
            key["Gp"],
            dtype=np.uint8,
        )

    if Gp_array.shape != (
        k,
        n,
    ):
        raise ValueError(
            "Serialized public generator has "
            f"shape {Gp_array.shape}; "
            f"expected ({k},{n})."
        )

    mc = McElieceCipher(
        m,
        n,
        t,
    )

    mc.k = k

    mc.Gp = uint8_to_gf2_matrix(
        Gp_array
    )

    return mc


# ================================================================
# Private-key loading
# ================================================================

def load_private_key(
    priv_key_file,
):
    """
    Load a Phase 2C private key using allow_pickle=False.

    The stored generator right inverse is checked once during loading:

        G R = I_k.

    After successful loading, normal decryption can use R directly
    without reconstructing it.
    """

    with np.load(
        priv_key_file,
        allow_pickle=False,
    ) as key:

        required = {
            "format_version",
            "m",
            "n",
            "t",
            "k",
            "G",
            "H",
            "R",
            "right_inverse_pivots",
            "S",
            "S_inv",
            "P",
            "P_inv",
            "g_poly",
            "irr_poly",
        }

        missing = (
            required
            - set(
                key.files
            )
        )

        if missing:
            raise ValueError(
                "Private key is missing fields: "
                f"{sorted(missing)}"
            )

        version = int(
            np.asarray(
                key[
                    "format_version"
                ]
            ).item()
        )

        if version != KEY_FORMAT_VERSION:
            raise ValueError(
                "Unsupported private-key format "
                f"version {version}; "
                f"expected {KEY_FORMAT_VERSION}."
            )

        m = int(
            np.asarray(
                key["m"]
            ).item()
        )

        n = int(
            np.asarray(
                key["n"]
            ).item()
        )

        t = int(
            np.asarray(
                key["t"]
            ).item()
        )

        k = int(
            np.asarray(
                key["k"]
            ).item()
        )

        G_array = np.asarray(
            key["G"],
            dtype=np.uint8,
        )

        H_array = np.asarray(
            key["H"],
            dtype=np.uint8,
        )

        R_array = np.asarray(
            key["R"],
            dtype=np.uint8,
        )

        right_inverse_pivots = np.asarray(
            key[
                "right_inverse_pivots"
            ],
            dtype=np.int64,
        ).reshape(-1)

        S_array = np.asarray(
            key["S"],
            dtype=np.uint8,
        )

        S_inv_array = np.asarray(
            key["S_inv"],
            dtype=np.uint8,
        )

        P = np.asarray(
            key["P"],
            dtype=np.int64,
        ).reshape(-1)

        P_inv = np.asarray(
            key["P_inv"],
            dtype=np.int64,
        ).reshape(-1)

        g_poly = np.asarray(
            key["g_poly"],
            dtype=np.uint8,
        )

        irr_poly = np.asarray(
            key["irr_poly"],
            dtype=np.uint8,
        )

    # ------------------------------------------------------------
    # Dimension checks
    # ------------------------------------------------------------

    if G_array.shape != (
        k,
        n,
    ):
        raise ValueError(
            "Serialized G has shape "
            f"{G_array.shape}; "
            f"expected ({k},{n})."
        )

    if H_array.ndim != 2:
        raise ValueError(
            "Serialized H must be two-dimensional."
        )

    if H_array.shape[1] != n:
        raise ValueError(
            "Serialized H has incorrect "
            "number of columns."
        )

    if R_array.shape != (
        n,
        k,
    ):
        raise ValueError(
            "Serialized R has shape "
            f"{R_array.shape}; "
            f"expected ({n},{k})."
        )

    if right_inverse_pivots.shape != (
        k,
    ):
        raise ValueError(
            "Serialized right_inverse_pivots "
            f"has shape {right_inverse_pivots.shape}; "
            f"expected ({k},)."
        )

    if S_array.shape != (
        k,
        k,
    ):
        raise ValueError(
            "Serialized S has shape "
            f"{S_array.shape}; "
            f"expected ({k},{k})."
        )

    if S_inv_array.shape != (
        k,
        k,
    ):
        raise ValueError(
            "Serialized S_inv has shape "
            f"{S_inv_array.shape}; "
            f"expected ({k},{k})."
        )

    # ------------------------------------------------------------
    # Validate permutation representation.
    # ------------------------------------------------------------

    (
        P,
        P_inv,
    ) = validate_permutation_pair(
        P,
        P_inv,
        n,
    )

    # ------------------------------------------------------------
    # Convert binary matrices.
    # ------------------------------------------------------------

    G = uint8_to_gf2_matrix(
        G_array
    )

    H = uint8_to_gf2_matrix(
        H_array
    )

    R = uint8_to_gf2_matrix(
        R_array
    )

    S = uint8_to_gf2_matrix(
        S_array
    )

    S_inv = uint8_to_gf2_matrix(
        S_inv_array
    )

    # ------------------------------------------------------------
    # Phase 2C right-inverse validation.
    #
    # This is the one-time load-time check:
    #
    #     G R = I_k.
    # ------------------------------------------------------------

    validated_pivots = (
        validate_right_inverse(
            G,
            R,
            right_inverse_pivots,
        )
    )

    # ------------------------------------------------------------
    # Verify scrambling inverse once while loading.
    # ------------------------------------------------------------

    SS_inv = (
        S
        * S_inv
    )

    if not gf2_matrix_equal(
        SS_inv,
        gf2_identity(
            k
        ),
    ):
        raise ValueError(
            "Serialized scrambling matrices are "
            "inconsistent: S S_inv != I_k."
        )

    # ------------------------------------------------------------
    # Reconstruct McElieceCipher.
    # ------------------------------------------------------------

    mc = McElieceCipher(
        m,
        n,
        t,
    )

    mc.k = k

    mc.G = G

    mc.H = H

    mc.R = R

    mc.right_inverse_pivots = (
        validated_pivots
    )

    mc.S = S

    mc.S_inv = S_inv

    mc.P = P

    mc.P_inv = P_inv

    mc.g_poly = g_poly

    mc.irr_poly = irr_poly

    return mc


# ================================================================
# Encryption using serialized public key
# ================================================================

def encrypt(
    pub_key_file,
    input_arr,
    block=False,
):
    """
    Encrypt using a public key loaded from disk.
    """

    mc = load_public_key(
        pub_key_file
    )

    k = int(
        mc.Gp.arr.shape[0]
    )

    n = int(
        mc.Gp.arr.shape[1]
    )

    input_arr = np.asarray(
        input_arr,
        dtype=np.uint8,
    ).reshape(-1)

    if not np.all(
        (input_arr == 0)
        |
        (input_arr == 1)
    ):
        raise ValueError(
            "Input must be binary."
        )

    # ------------------------------------------------------------
    # Single-block encryption
    # ------------------------------------------------------------

    if not block:

        if len(
            input_arr
        ) != k:
            raise ValueError(
                "Single-block plaintext must contain "
                f"exactly {k} bits; "
                f"received {len(input_arr)}."
            )

        return (
            mc.encrypt(
                input_arr
            )
            .to_numpy()
            .astype(
                np.uint8
            )
        )

    # ------------------------------------------------------------
    # Multi-block encryption
    # ------------------------------------------------------------

    padded = padding_encode(
        input_arr,
        k,
    )

    padded = np.asarray(
        padded,
        dtype=np.uint8,
    )

    if (
        len(
            padded
        )
        % k
        != 0
    ):
        raise RuntimeError(
            "Padding did not produce a multiple "
            "of the plaintext block size."
        )

    blocks = padded.reshape(
        -1,
        k,
    )

    ciphertext_blocks = []

    for index, block_data in enumerate(
        blocks,
        start=1,
    ):

        log.info(
            "Encrypting block %d of %d",
            index,
            len(
                blocks
            ),
        )

        encrypted = (
            mc.encrypt(
                block_data
            )
            .to_numpy()
            .astype(
                np.uint8
            )
        )

        if encrypted.shape != (
            n,
        ):
            raise RuntimeError(
                "Unexpected ciphertext block length."
            )

        ciphertext_blocks.append(
            encrypted
        )

    if not ciphertext_blocks:
        return np.array(
            [],
            dtype=np.uint8,
        )

    return np.concatenate(
        ciphertext_blocks
    ).astype(
        np.uint8
    )


# ================================================================
# Decryption using serialized private key
# ================================================================

def decrypt(
    priv_key_file,
    input_arr,
    block=False,
):
    """
    Decrypt using a Phase 2C private key loaded from disk.

    The loaded private key already contains R, so normal decryption
    does not need to reconstruct the generator right inverse.
    """

    mc = load_private_key(
        priv_key_file
    )

    n = int(
        mc.H.arr.shape[1]
    )

    k = int(
        mc.G.arr.shape[0]
    )

    if mc.R is None:
        raise RuntimeError(
            "Loaded Phase 2C private key "
            "does not contain R."
        )

    input_arr = np.asarray(
        input_arr,
        dtype=np.uint8,
    ).reshape(-1)

    if not np.all(
        (input_arr == 0)
        |
        (input_arr == 1)
    ):
        raise ValueError(
            "Ciphertext must be binary."
        )

    # ------------------------------------------------------------
    # Single-block decryption
    # ------------------------------------------------------------

    if not block:

        if len(
            input_arr
        ) != n:
            raise ValueError(
                "Single-block ciphertext must contain "
                f"exactly {n} bits; "
                f"received {len(input_arr)}."
            )

        return np.asarray(
            mc.decrypt(
                input_arr
            ),
            dtype=np.uint8,
        ).reshape(-1)

    # ------------------------------------------------------------
    # Multi-block decryption
    # ------------------------------------------------------------

    if (
        len(
            input_arr
        )
        % n
        != 0
    ):
        raise ValueError(
            "Block ciphertext length must be "
            f"a multiple of {n}."
        )

    ciphertext_blocks = (
        input_arr.reshape(
            -1,
            n,
        )
    )

    plaintext_blocks = []

    for index, block_data in enumerate(
        ciphertext_blocks,
        start=1,
    ):

        log.info(
            "Decrypting block %d of %d",
            index,
            len(
                ciphertext_blocks
            ),
        )

        decoded = np.asarray(
            mc.decrypt(
                block_data
            ),
            dtype=np.uint8,
        ).reshape(-1)

        if decoded.shape != (
            k,
        ):
            raise RuntimeError(
                "Unexpected plaintext block length."
            )

        plaintext_blocks.append(
            decoded
        )

    if not plaintext_blocks:
        return np.array(
            [],
            dtype=np.uint8,
        )

    padded_plaintext = (
        np.concatenate(
            plaintext_blocks
        )
        .astype(
            np.uint8
        )
    )

    decoded = padding_decode(
        padded_plaintext,
        k,
    )

    return np.asarray(
        decoded,
        dtype=np.uint8,
    ).reshape(-1)


# ================================================================
# CLI input handling
# ================================================================

def read_cli_input(
    filename,
    poly_input,
):
    """
    Read binary or integer-array input from a file/stdin.
    """

    if (
        filename is None
        or
        filename == "-"
    ):

        if poly_input:
            raw = sys.stdin.read()

        else:
            raw = sys.stdin.buffer.read()

    else:

        if poly_input:

            with open(
                filename,
                "r",
                encoding="utf-8",
            ) as file:
                raw = file.read()

        else:

            with open(
                filename,
                "rb",
            ) as file:
                raw = file.read()

    if poly_input:

        parsed = ast.literal_eval(
            raw
        )

        result = np.asarray(
            parsed,
            dtype=np.uint8,
        ).reshape(-1)

    else:

        result = np.unpackbits(
            np.frombuffer(
                raw,
                dtype=np.uint8,
            )
        ).astype(
            np.uint8
        )

    if not np.all(
        (result == 0)
        |
        (result == 1)
    ):
        raise ValueError(
            "Input contains non-binary values."
        )

    return result


# ================================================================
# CLI output handling
# ================================================================

def write_cli_output(
    output,
    poly_output,
):
    """
    Write output either as an integer list or packed bytes.
    """

    output = np.asarray(
        output,
        dtype=np.uint8,
    ).reshape(-1)

    if poly_output:

        print(
            [
                int(x)
                for x in output
            ]
        )

    else:

        sys.stdout.buffer.write(
            np.packbits(
                output
            ).tobytes()
        )


# ================================================================
# Main command-line entry point
# ================================================================

def main():

    args = docopt(
        __doc__,
        version="McEliece Phase 2C",
    )

    root = logging.getLogger()

    root.setLevel(
        logging.DEBUG
    )

    handler = logging.StreamHandler(
        sys.stderr
    )

    if args[
        "--debug"
    ]:
        handler.setLevel(
            logging.DEBUG
        )

    elif args[
        "--verbose"
    ]:
        handler.setLevel(
            logging.INFO
        )

    else:
        handler.setLevel(
            logging.WARNING
        )

    root.addHandler(
        handler
    )

    poly_input = bool(
        args[
            "--poly-input"
        ]
    )

    poly_output = bool(
        args[
            "--poly-output"
        ]
    )

    block = bool(
        args[
            "--block"
        ]
    )

    # ------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------

    if args[
        "gen"
    ]:

        generate(
            int(
                args["M"]
            ),
            int(
                args["N"]
            ),
            int(
                args["T"]
            ),
            args[
                "PRIV_KEY_FILE"
            ],
            args[
                "PUB_KEY_FILE"
            ],
        )

        return

    # ------------------------------------------------------------
    # Read input for encryption/decryption.
    # ------------------------------------------------------------

    input_arr = read_cli_input(
        args[
            "FILE"
        ],
        poly_input,
    )

    # ------------------------------------------------------------
    # Encryption
    # ------------------------------------------------------------

    if args[
        "enc"
    ]:

        output = encrypt(
            args[
                "PUB_KEY_FILE"
            ],
            input_arr,
            block=block,
        )

    # ------------------------------------------------------------
    # Decryption
    # ------------------------------------------------------------

    elif args[
        "dec"
    ]:

        output = decrypt(
            args[
                "PRIV_KEY_FILE"
            ],
            input_arr,
            block=block,
        )

    else:
        raise RuntimeError(
            "No operation selected."
        )

    write_cli_output(
        output,
        poly_output,
    )


if __name__ == "__main__":
    main()
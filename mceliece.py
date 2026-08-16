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

Phase 2D private-key format:

    format_version
    m, n, t, k
    G
    H
    right_inverse_pivots
    right_inverse_core
    S
    S_inv
    P
    P_inv
    g_poly
    irr_poly

where

    J = right_inverse_pivots,
    C = right_inverse_core,

and

    G[:,J] C = I_k.

No dense n x k right inverse is serialized.
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


KEY_FORMAT_VERSION = 4


# ================================================================
# GF(2) helpers
# ================================================================

def gf2_matrix_to_uint8(matrix):

    if not isinstance(
        matrix,
        GF2Matrix,
    ):
        raise TypeError(
            "Expected GF2Matrix."
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
            "GF(2) data contains "
            "entries outside {0,1}."
        )

    return GF2Matrix.from_list(
        array
    )


def gf2_matrix_equal(
    A,
    B,
):

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


def gf2_identity(k):

    return GF2Matrix.from_list(
        np.eye(
            k,
            dtype=np.uint8,
        )
    )


# ================================================================
# Permutations
# ================================================================

def validate_permutation(
    permutation,
    n,
    name,
):

    permutation = np.asarray(
        permutation,
        dtype=np.int64,
    ).reshape(-1)

    if permutation.shape != (
        n,
    ):
        raise ValueError(
            f"{name} must have shape ({n},)."
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
            f"{name} is not a valid permutation."
        )

    return permutation


def validate_permutation_pair(
    P,
    P_inv,
    n,
):

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
# Phase 2D factorized right-inverse validation
# ================================================================

def validate_right_inverse_factorization(
    G,
    pivots,
    core,
):

    if not isinstance(
        G,
        GF2Matrix,
    ):
        raise TypeError(
            "G must be GF2Matrix."
        )

    if not isinstance(
        core,
        GF2Matrix,
    ):
        raise TypeError(
            "core must be GF2Matrix."
        )

    k, n = G.arr.shape

    pivots = np.asarray(
        pivots,
        dtype=np.int64,
    ).reshape(-1)

    if pivots.shape != (
        k,
    ):
        raise ValueError(
            "right_inverse_pivots has "
            f"shape {pivots.shape}; "
            f"expected ({k},)."
        )

    if len(
        np.unique(
            pivots
        )
    ) != k:
        raise ValueError(
            "right_inverse_pivots contains duplicates."
        )

    if (
        np.any(
            pivots < 0
        )
        or
        np.any(
            pivots >= n
        )
    ):
        raise ValueError(
            "right_inverse_pivots contains "
            "invalid coordinates."
        )

    if core.arr.shape != (
        k,
        k,
    ):
        raise ValueError(
            "right_inverse_core has shape "
            f"{core.arr.shape}; "
            f"expected ({k},{k})."
        )

    B = GF2Matrix(
        G.arr[
            :,
            pivots
        ]
    )

    identity = gf2_identity(
        k
    )

    if not gf2_matrix_equal(
        B * core,
        identity,
    ):
        raise ValueError(
            "Invalid factorization: "
            "G[:,J] C != I_k."
        )

    if not gf2_matrix_equal(
        core * B,
        identity,
    ):
        raise ValueError(
            "Invalid factorization: "
            "C G[:,J] != I_k."
        )

    return pivots


# ================================================================
# Key generation
# ================================================================

def generate(
    m,
    n,
    t,
    priv_key_file,
    pub_key_file,
):

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

    (
        P,
        P_inv,
    ) = validate_permutation_pair(
        mc.P,
        mc.P_inv,
        n,
    )

    if mc.right_inverse_core is None:
        raise RuntimeError(
            "right_inverse_core was not generated."
        )

    if mc.right_inverse_pivots is None:
        raise RuntimeError(
            "right_inverse_pivots was not generated."
        )

    pivots = (
        validate_right_inverse_factorization(
            mc.G,
            mc.right_inverse_pivots,
            mc.right_inverse_core,
        )
    )

    G = gf2_matrix_to_uint8(
        mc.G
    )

    H = gf2_matrix_to_uint8(
        mc.H
    )

    core = gf2_matrix_to_uint8(
        mc.right_inverse_core
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

    # Private key.
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

        right_inverse_pivots=(
            pivots.astype(
                np.int64
            )
        ),

        right_inverse_core=core,

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

    # Public key.
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
            "G_pub has invalid dimensions."
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
            "right_inverse_pivots",
            "right_inverse_core",
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

        pivots = np.asarray(
            key[
                "right_inverse_pivots"
            ],
            dtype=np.int64,
        ).reshape(-1)

        core_array = np.asarray(
            key[
                "right_inverse_core"
            ],
            dtype=np.uint8,
        )

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

    if G_array.shape != (
        k,
        n,
    ):
        raise ValueError(
            "Serialized G has invalid dimensions."
        )

    if (
        H_array.ndim != 2
        or
        H_array.shape[1] != n
    ):
        raise ValueError(
            "Serialized H has invalid dimensions."
        )

    if pivots.shape != (
        k,
    ):
        raise ValueError(
            "Serialized pivot vector has "
            "invalid dimensions."
        )

    if core_array.shape != (
        k,
        k,
    ):
        raise ValueError(
            "Serialized right-inverse core "
            "has invalid dimensions."
        )

    if S_array.shape != (
        k,
        k,
    ):
        raise ValueError(
            "Serialized S has invalid dimensions."
        )

    if S_inv_array.shape != (
        k,
        k,
    ):
        raise ValueError(
            "Serialized S_inv has invalid dimensions."
        )

    (
        P,
        P_inv,
    ) = validate_permutation_pair(
        P,
        P_inv,
        n,
    )

    G = uint8_to_gf2_matrix(
        G_array
    )

    H = uint8_to_gf2_matrix(
        H_array
    )

    core = uint8_to_gf2_matrix(
        core_array
    )

    S = uint8_to_gf2_matrix(
        S_array
    )

    S_inv = uint8_to_gf2_matrix(
        S_inv_array
    )

    pivots = (
        validate_right_inverse_factorization(
            G,
            pivots,
            core,
        )
    )

    if not gf2_matrix_equal(
        S * S_inv,
        gf2_identity(
            k
        ),
    ):
        raise ValueError(
            "S S_inv != I_k."
        )

    mc = McElieceCipher(
        m,
        n,
        t,
    )

    mc.k = k

    mc.G = G

    mc.H = H

    mc.right_inverse_pivots = (
        pivots
    )

    mc.right_inverse_core = (
        core
    )

    mc.S = S

    mc.S_inv = S_inv

    mc.P = P

    mc.P_inv = P_inv

    mc.g_poly = g_poly

    mc.irr_poly = irr_poly

    return mc


# ================================================================
# Encryption
# ================================================================

def encrypt(
    pub_key_file,
    input_arr,
    block=False,
):

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

    if not block:

        if len(
            input_arr
        ) != k:
            raise ValueError(
                f"Plaintext must contain {k} bits."
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

    padded = np.asarray(
        padding_encode(
            input_arr,
            k,
        ),
        dtype=np.uint8,
    )

    if len(
        padded
    ) % k != 0:
        raise RuntimeError(
            "Invalid padded plaintext length."
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
# Decryption
# ================================================================

def decrypt(
    priv_key_file,
    input_arr,
    block=False,
):

    mc = load_private_key(
        priv_key_file
    )

    n = int(
        mc.H.arr.shape[1]
    )

    k = int(
        mc.G.arr.shape[0]
    )

    if mc.right_inverse_core is None:
        raise RuntimeError(
            "Loaded key does not contain "
            "right_inverse_core."
        )

    if mc.right_inverse_pivots is None:
        raise RuntimeError(
            "Loaded key does not contain "
            "right_inverse_pivots."
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

    if not block:

        if len(
            input_arr
        ) != n:
            raise ValueError(
                f"Ciphertext must contain {n} bits."
            )

        return np.asarray(
            mc.decrypt(
                input_arr
            ),
            dtype=np.uint8,
        ).reshape(-1)

    if len(
        input_arr
    ) % n != 0:
        raise ValueError(
            "Block ciphertext length must "
            f"be a multiple of {n}."
        )

    blocks = input_arr.reshape(
        -1,
        n,
    )

    plaintext_blocks = []

    for index, block_data in enumerate(
        blocks,
        start=1,
    ):

        log.info(
            "Decrypting block %d of %d",
            index,
            len(
                blocks
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
# CLI input/output
# ================================================================

def read_cli_input(
    filename,
    poly_input,
):

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


def write_cli_output(
    output,
    poly_output,
):

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
# Main
# ================================================================

def main():

    args = docopt(
        __doc__,
        version="McEliece Phase 2D",
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

    input_arr = read_cli_input(
        args[
            "FILE"
        ],
        bool(
            args[
                "--poly-input"
            ]
        ),
    )

    if args[
        "enc"
    ]:

        output = encrypt(
            args[
                "PUB_KEY_FILE"
            ],
            input_arr,
            block=bool(
                args[
                    "--block"
                ]
            ),
        )

    elif args[
        "dec"
    ]:

        output = decrypt(
            args[
                "PRIV_KEY_FILE"
            ],
            input_arr,
            block=bool(
                args[
                    "--block"
                ]
            ),
        )

    else:
        raise RuntimeError(
            "No operation selected."
        )

    write_cli_output(
        output,
        bool(
            args[
                "--poly-output"
            ]
        ),
    )


if __name__ == "__main__":
    main()
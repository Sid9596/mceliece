"""
Phase 2E detailed scaling and bottleneck profiler.

This experiment profiles the binary-Goppa McEliece implementation
stage by stage.

Measured components
-------------------

1. Extension-field, support, and Goppa-polynomial generation.
2. Parity-check construction:
       extension-field H
       binary expansion H_bin
3. Nullspace computation for the generator G.
4. Factorized right inverse:
       J = right_inverse_pivots
       C = right_inverse_core
       G[:,J] C = I_k
5. Dense scrambling matrix:
       generation of S
       inversion of S
6. Public-generator construction:
       S G
       column permutation
7. Encryption.
8. Patterson correction.
9. Factorized information extraction:
       c[J] C
10. Total decryption.
11. Memory/storage measurements for
       G, H, S, S_inv, G_pub, J, C.

Phase 2D representation
-----------------------

The generator right inverse is represented by

    J = right_inverse_pivots

and

    C = right_inverse_core = (G[:,J])^{-1}.

The full n x k right inverse R is never required.

For a clean codeword

    c = a G,

we recover

    a = c[J] C.

Output
------

Results are appended incrementally to

    phase2e_scaling_results.csv

so successful smaller parameter sets are preserved even if a larger
parameter set fails or is interrupted.
"""

import argparse
import csv
import gc
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import traceback

import numpy as np

from goppa.goppacodegenerator import GoppaCodeGenerator
from mceliece.mceliececipher import McElieceCipher
from mceliece.mathutils import (
    GF2Matrix,
    get_binary_from_alpha,
    random_inv_matrix,
)


# ======================================================================
# Parameter progression
# ======================================================================

PARAMETER_SETS = [
    # index, m, n, t, trials
    (0, 7, 127, 4, 3),
    (1, 8, 255, 8, 3),
    (2, 9, 511, 16, 2),
    (3, 10, 1023, 32, 1),
]


DEFAULT_CSV = "phase2e_scaling_results.csv"

BASE_SEED = 20260816


# ======================================================================
# CSV columns
# ======================================================================

CSV_FIELDS = [
    # Run information
    "run_time",
    "parameter_index",
    "status",
    "error_stage",
    "error_message",
    "seed",

    # Parameters
    "m",
    "n",
    "t",
    "k",
    "mt",
    "dimension_lower_bound",
    "rate",

    # Stage 1
    "field_generation_sec",
    "support_generation_sec",
    "goppa_polynomial_sec",
    "goppa_storage_conversion_sec",
    "goppa_support_total_sec",

    # Stage 2
    "H_extension_sec",
    "H_binary_expansion_sec",
    "H_total_sec",

    # Stage 3
    "nullspace_sec",
    "generator_finalize_sec",
    "generator_total_sec",
    "rank_H",

    # Stage 4
    "right_inverse_factorization_sec",

    # Stage 5
    "S_generation_sec",
    "S_inversion_sec",
    "S_total_sec",

    # Additional permutation timing
    "permutation_generation_sec",

    # Stage 6
    "SG_multiplication_sec",
    "Gpub_permutation_sec",
    "Gpub_total_sec",

    # Total construction
    "construction_wall_sec",

    # Stages 7-10
    "encryption_avg_sec",
    "encryption_min_sec",
    "encryption_max_sec",

    "syndrome_avg_sec",

    "patterson_avg_sec",
    "patterson_min_sec",
    "patterson_max_sec",

    "factorized_extraction_avg_sec",
    "factorized_extraction_min_sec",
    "factorized_extraction_max_sec",

    "unscramble_avg_sec",

    "total_decryption_avg_sec",
    "total_decryption_min_sec",
    "total_decryption_max_sec",

    "trials",
    "successful_trials",

    # Matrix dimensions
    "G_rows",
    "G_cols",
    "H_rows",
    "H_cols",
    "S_rows",
    "S_cols",
    "S_inv_rows",
    "S_inv_cols",
    "Gpub_rows",
    "Gpub_cols",
    "J_entries",
    "C_rows",
    "C_cols",

    # Logical binary storage
    "G_logical_packed_bytes",
    "H_logical_packed_bytes",
    "S_logical_packed_bytes",
    "S_inv_logical_packed_bytes",
    "Gpub_logical_packed_bytes",
    "C_logical_packed_bytes",

    # uint8 serialized storage
    "G_uint8_bytes",
    "H_uint8_bytes",
    "S_uint8_bytes",
    "S_inv_uint8_bytes",
    "Gpub_uint8_bytes",
    "C_uint8_bytes",
    "J_int64_bytes",

    # NumPy object-array pointer buffers
    "G_object_buffer_bytes",
    "H_object_buffer_bytes",
    "S_object_buffer_bytes",
    "S_inv_object_buffer_bytes",
    "Gpub_object_buffer_bytes",
    "C_object_buffer_bytes",

    # Approximate resident memory of GF2 object representation
    "G_estimated_python_bytes",
    "H_estimated_python_bytes",
    "S_estimated_python_bytes",
    "S_inv_estimated_python_bytes",
    "Gpub_estimated_python_bytes",
    "C_estimated_python_bytes",

    # Aggregate storage
    "total_binary_packed_bytes",
    "total_uint8_plus_J_bytes",
    "total_object_buffer_plus_J_bytes",
    "total_estimated_python_plus_J_bytes",

    # Actual compressed NPZ sizes
    "private_npz_bytes",
    "public_npz_bytes",

    # Process memory diagnostics
    "rss_after_goppa_mib",
    "rss_after_H_mib",
    "rss_after_G_mib",
    "rss_after_right_inverse_mib",
    "rss_after_S_mib",
    "rss_after_Gpub_mib",
    "rss_before_trials_mib",
    "rss_after_trials_mib",
    "peak_rss_mib",

    # Correctness
    "GHt_zero",
    "right_inverse_identity",
    "scrambling_inverse_identity",
    "public_generator_identity",
    "factorized_clean_extraction",
    "patterson_success",
    "decryption_success",
]


# ======================================================================
# Basic helpers
# ======================================================================

def now_string():
    return time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def seconds_since(start):
    return time.perf_counter() - start


def gf2_numpy(matrix):
    """
    Convert GF2Matrix to a uint8 NumPy array while preserving shape.
    """

    if not isinstance(
        matrix,
        GF2Matrix,
    ):
        raise TypeError(
            "gf2_numpy expects GF2Matrix."
        )

    return np.array(
        [
            int(x.n)
            for x in matrix.arr.flat
        ],
        dtype=np.uint8,
    ).reshape(
        matrix.arr.shape
    )


def matrix_equal(A, B):
    if A.arr.shape != B.arr.shape:
        return False

    return np.array_equal(
        gf2_numpy(A),
        gf2_numpy(B),
    )


def matrix_is_zero(M):
    return not np.any(
        gf2_numpy(M)
    )


def gf2_identity(k):
    return GF2Matrix.from_list(
        np.eye(
            k,
            dtype=np.uint8,
        )
    )


def vector_equal(A, B):
    return np.array_equal(
        np.asarray(
            A,
            dtype=np.uint8,
        ).reshape(-1),
        np.asarray(
            B,
            dtype=np.uint8,
        ).reshape(-1),
    )


def timing_stats(values):
    """
    Return avg, min, max.

    Empty input returns NaN.
    """

    if not values:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    return (
        float(np.mean(values)),
        float(np.min(values)),
        float(np.max(values)),
    )


# ======================================================================
# Process memory
# ======================================================================

def current_rss_mib():
    """
    Current resident-set size on Linux using /proc/self/status.

    Returns NaN on unsupported platforms.
    """

    try:
        with open(
            "/proc/self/status",
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if line.startswith(
                    "VmRSS:"
                ):
                    kb = int(
                        line.split()[1]
                    )

                    return (
                        kb / 1024.0
                    )

    except Exception:
        pass

    return float("nan")


def peak_rss_mib():
    """
    Process peak resident memory.

    On Linux ru_maxrss is reported in KiB.
    """

    try:
        usage = resource.getrusage(
            resource.RUSAGE_SELF
        )

        return (
            float(
                usage.ru_maxrss
            )
            / 1024.0
        )

    except Exception:
        return float("nan")


# ======================================================================
# Matrix storage measurements
# ======================================================================

def logical_packed_bytes(entries):
    """
    Minimum byte count required to bit-pack `entries` binary values.
    """

    return (
        int(entries) + 7
    ) // 8


def matrix_memory_stats(matrix):
    """
    Return several views of GF2Matrix storage.

    logical_packed_bytes:
        theoretical bit-packed representation.

    uint8_bytes:
        storage if each binary coefficient is serialized as uint8.

    object_buffer_bytes:
        NumPy object-array pointer buffer only. This does not include
        the GF2 objects pointed to by the array.

    estimated_python_bytes:
        object-buffer bytes plus sys.getsizeof() for one GF2 object
        per matrix entry.

        This remains an estimate; Python allocator overhead is not
        included.
    """

    if not isinstance(
        matrix,
        GF2Matrix,
    ):
        raise TypeError(
            "matrix_memory_stats expects GF2Matrix."
        )

    entries = int(
        matrix.arr.size
    )

    pointer_buffer = int(
        matrix.arr.nbytes
    )

    if entries > 0:
        one_object_size = int(
            sys.getsizeof(
                matrix.arr.flat[0]
            )
        )
    else:
        one_object_size = 0

    estimated_python = (
        pointer_buffer
        +
        entries * one_object_size
    )

    return {
        "entries": entries,
        "packed": logical_packed_bytes(
            entries
        ),
        "uint8": entries,
        "object_buffer": pointer_buffer,
        "estimated_python": (
            estimated_python
        ),
    }


def record_matrix_stats(
    row,
    prefix,
    matrix,
):
    stats = matrix_memory_stats(
        matrix
    )

    shape = matrix.arr.shape

    row[
        f"{prefix}_rows"
    ] = int(
        shape[0]
    )

    row[
        f"{prefix}_cols"
    ] = int(
        shape[1]
    )

    row[
        f"{prefix}_logical_packed_bytes"
    ] = stats[
        "packed"
    ]

    row[
        f"{prefix}_uint8_bytes"
    ] = stats[
        "uint8"
    ]

    row[
        f"{prefix}_object_buffer_bytes"
    ] = stats[
        "object_buffer"
    ]

    row[
        f"{prefix}_estimated_python_bytes"
    ] = stats[
        "estimated_python"
    ]

    return stats


# ======================================================================
# Incremental CSV writer
# ======================================================================

def append_csv_row(
    csv_path,
    row,
):
    csv_path = Path(
        csv_path
    )

    file_exists = (
        csv_path.exists()
        and
        csv_path.stat().st_size > 0
    )

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        if not file_exists:
            writer.writeheader()

        complete_row = {
            field: row.get(
                field,
                ""
            )
            for field in CSV_FIELDS
        }

        writer.writerow(
            complete_row
        )

        file.flush()

        os.fsync(
            file.fileno()
        )


# ======================================================================
# Goppa-polynomial primitive storage conversion
# ======================================================================

def convert_polynomials_for_cipher(
    g_poly,
    irr_poly,
):
    """
    Convert the SymPy extension-field polynomial representation to
    the primitive arrays used by McElieceCipher.
    """

    g_coeffs = []

    for coefficient in (
        g_poly
        .all_coeffs()[::-1]
    ):
        bits = get_binary_from_alpha(
            coefficient,
            irr_poly,
            2,
        )

        g_coeffs.append(
            [
                int(bit.n)
                for bit in bits
            ]
        )

    g_storage = np.asarray(
        g_coeffs,
        dtype=np.uint8,
    )

    irr_storage = np.asarray(
        [
            int(c) % 2
            for c
            in irr_poly
            .all_coeffs()[::-1]
        ],
        dtype=np.uint8,
    )

    return (
        g_storage,
        irr_storage,
    )


# ======================================================================
# Actual compressed storage measurement
# ======================================================================

def measure_npz_sizes(
    mc,
    g_storage,
    irr_storage,
):
    """
    Measure actual compressed Phase-2D-style NPZ sizes without
    regenerating the key.
    """

    with tempfile.TemporaryDirectory(
        prefix="phase2e_storage_"
    ) as temp_dir:

        temp_dir = Path(
            temp_dir
        )

        private_path = (
            temp_dir
            / "private.npz"
        )

        public_path = (
            temp_dir
            / "public.npz"
        )

        np.savez_compressed(
            private_path,

            m=np.asarray(
                mc.m,
                dtype=np.int64,
            ),

            n=np.asarray(
                mc.n,
                dtype=np.int64,
            ),

            t=np.asarray(
                mc.t,
                dtype=np.int64,
            ),

            k=np.asarray(
                mc.k,
                dtype=np.int64,
            ),

            G=gf2_numpy(
                mc.G
            ),

            H=gf2_numpy(
                mc.H
            ),

            right_inverse_pivots=(
                np.asarray(
                    mc.right_inverse_pivots,
                    dtype=np.int64,
                )
            ),

            right_inverse_core=(
                gf2_numpy(
                    mc.right_inverse_core
                )
            ),

            S=gf2_numpy(
                mc.S
            ),

            S_inv=gf2_numpy(
                mc.S_inv
            ),

            P=np.asarray(
                mc.P,
                dtype=np.int64,
            ),

            P_inv=np.asarray(
                mc.P_inv,
                dtype=np.int64,
            ),

            g_poly=np.asarray(
                g_storage,
                dtype=np.uint8,
            ),

            irr_poly=np.asarray(
                irr_storage,
                dtype=np.uint8,
            ),
        )

        np.savez_compressed(
            public_path,

            m=np.asarray(
                mc.m,
                dtype=np.int64,
            ),

            n=np.asarray(
                mc.n,
                dtype=np.int64,
            ),

            t=np.asarray(
                mc.t,
                dtype=np.int64,
            ),

            k=np.asarray(
                mc.k,
                dtype=np.int64,
            ),

            Gp=gf2_numpy(
                mc.Gp
            ),
        )

        return (
            int(
                private_path.stat().st_size
            ),
            int(
                public_path.stat().st_size
            ),
        )


# ======================================================================
# Parameter-set profiler
# ======================================================================

def profile_parameter_set(
    index,
    m,
    n,
    t,
    trials,
    seed,
):
    """
    Profile one parameter set.

    Results are returned as a dictionary suitable for CSV output.
    """

    row = {
        "run_time": now_string(),
        "parameter_index": index,
        "status": "RUNNING",
        "error_stage": "",
        "error_message": "",
        "seed": seed,
        "m": m,
        "n": n,
        "t": t,
        "mt": m * t,
        "dimension_lower_bound": (
            n - m * t
        ),
        "trials": trials,
    }

    stage = "initialization"

    print()
    print("=" * 88)

    print(
        f"PHASE 2E PROFILE "
        f"index={index}: "
        f"m={m}, n={n}, t={t}, "
        f"trials={trials}"
    )

    print("=" * 88)

    np.random.seed(
        seed
    )

    construction_start = (
        time.perf_counter()
    )

    try:
        # ==========================================================
        # 1. Field, support, Goppa polynomial
        # ==========================================================

        stage = "field_generation"

        generator = (
            GoppaCodeGenerator(
                m,
                n,
                t,
            )
        )

        start = time.perf_counter()

        (
            irr_poly,
            ring,
        ) = (
            generator
            ._generate_primitive_field_polynomial()
        )

        row[
            "field_generation_sec"
        ] = seconds_since(
            start
        )

        print()
        print(
            "[1a] Extension field"
        )

        print(
            "  time = "
            f"{row['field_generation_sec']:.6f} s"
        )

        # ----------------------------------------------------------
        # Support
        # ----------------------------------------------------------

        stage = "support_generation"

        start = time.perf_counter()

        support_exponents = list(
            range(
                n
            )
        )

        support = [
            generator.support_exponents
            for _ in []
        ]

        # Explicit support used by the current implementation.
        support = [
            exponent
            for exponent in support_exponents
        ]

        generator.support_exponents = (
            support_exponents
        )

        # The actual generator stores symbolic alpha powers.
        # We do not need a second symbolic list for the profiler;
        # support_exponents completely specifies the current support.
        generator.support = (
            support
        )

        row[
            "support_generation_sec"
        ] = seconds_since(
            start
        )

        print()
        print(
            "[1b] Support"
        )

        print(
            f"  length = {len(support_exponents)}"
        )

        print(
            "  time   = "
            f"{row['support_generation_sec']:.6f} s"
        )

        # ----------------------------------------------------------
        # Goppa polynomial
        # ----------------------------------------------------------

        stage = "goppa_polynomial"

        start = time.perf_counter()

        g_poly = (
            generator
            ._generate_goppa_polynomial(
                irr_poly,
                ring,
                support_exponents,
            )
        )

        row[
            "goppa_polynomial_sec"
        ] = seconds_since(
            start
        )

        print()
        print(
            "[1c] Goppa polynomial"
        )

        print(
            f"  degree = {g_poly.degree()}"
        )

        print(
            "  time   = "
            f"{row['goppa_polynomial_sec']:.6f} s"
        )

        # ----------------------------------------------------------
        # Primitive storage conversion used by McElieceCipher.
        # ----------------------------------------------------------

        stage = "goppa_storage_conversion"

        start = time.perf_counter()

        (
            g_storage,
            irr_storage,
        ) = (
            convert_polynomials_for_cipher(
                g_poly,
                irr_poly,
            )
        )

        row[
            "goppa_storage_conversion_sec"
        ] = seconds_since(
            start
        )

        row[
            "goppa_support_total_sec"
        ] = (
            row[
                "field_generation_sec"
            ]
            +
            row[
                "support_generation_sec"
            ]
            +
            row[
                "goppa_polynomial_sec"
            ]
            +
            row[
                "goppa_storage_conversion_sec"
            ]
        )

        row[
            "rss_after_goppa_mib"
        ] = current_rss_mib()

        # ==========================================================
        # 2. H construction
        # ==========================================================

        stage = "H_extension"

        print()
        print(
            "[2] Parity-check construction"
        )

        start = time.perf_counter()

        H_ext = (
            generator
            ._build_extension_parity_check(
                g_poly,
                irr_poly,
                ring,
                support_exponents,
            )
        )

        row[
            "H_extension_sec"
        ] = seconds_since(
            start
        )

        print(
            "  extension H = "
            f"{row['H_extension_sec']:.6f} s"
        )

        stage = "H_binary_expansion"

        start = time.perf_counter()

        H = (
            generator
            ._binary_expand_parity_check(
                H_ext,
                irr_poly,
            )
        )

        row[
            "H_binary_expansion_sec"
        ] = seconds_since(
            start
        )

        row[
            "H_total_sec"
        ] = (
            row[
                "H_extension_sec"
            ]
            +
            row[
                "H_binary_expansion_sec"
            ]
        )

        print(
            "  binary expansion = "
            f"{row['H_binary_expansion_sec']:.6f} s"
        )

        print(
            "  H total          = "
            f"{row['H_total_sec']:.6f} s"
        )

        print(
            "  H shape          = "
            f"{H.arr.shape}"
        )

        row[
            "rss_after_H_mib"
        ] = current_rss_mib()

        # H_ext is no longer needed.
        del H_ext

        gc.collect()

        # ==========================================================
        # 3. Nullspace -> G
        # ==========================================================

        stage = "nullspace"

        print()
        print(
            "[3] Nullspace / generator"
        )

        start = time.perf_counter()

        (
            H_nullspace,
            nullity,
        ) = H.nullspace()

        row[
            "nullspace_sec"
        ] = seconds_since(
            start
        )

        stage = "generator_finalize"

        start = time.perf_counter()

        G = GF2Matrix(
            H_nullspace.T()[
                :nullity
            ]
        )

        row[
            "generator_finalize_sec"
        ] = seconds_since(
            start
        )

        row[
            "generator_total_sec"
        ] = (
            row[
                "nullspace_sec"
            ]
            +
            row[
                "generator_finalize_sec"
            ]
        )

        k = int(
            nullity
        )

        row[
            "k"
        ] = k

        row[
            "rate"
        ] = (
            float(k)
            /
            float(n)
        )

        row[
            "rank_H"
        ] = (
            n - k
        )

        print(
            f"  k                  = {k}"
        )

        print(
            "  nullspace          = "
            f"{row['nullspace_sec']:.6f} s"
        )

        print(
            "  generator finalize = "
            f"{row['generator_finalize_sec']:.6f} s"
        )

        print(
            f"  G shape            = {G.arr.shape}"
        )

        # Verify G H^T = 0.
        GHt = (
            G
            * H.T()
        )

        GHt_zero = matrix_is_zero(
            GHt
        )

        row[
            "GHt_zero"
        ] = GHt_zero

        if not GHt_zero:
            raise RuntimeError(
                "G H^T != 0."
            )

        del H_nullspace
        del GHt

        gc.collect()

        row[
            "rss_after_G_mib"
        ] = current_rss_mib()

        # ==========================================================
        # Build the McElieceCipher object from the measured pieces.
        # ==========================================================

        mc = McElieceCipher(
            m,
            n,
            t,
        )

        mc.G = G
        mc.H = H
        mc.k = k

        mc.g_poly = (
            g_storage.copy()
        )

        mc.irr_poly = (
            irr_storage.copy()
        )

        # ==========================================================
        # 4. Factorized right inverse J,C
        # ==========================================================

        stage = "right_inverse_factorization"

        print()
        print(
            "[4] Factorized right inverse"
        )

        start = time.perf_counter()

        (
            mc.right_inverse_pivots,
            mc.right_inverse_core,
        ) = (
            mc
            ._construct_right_inverse_factorization()
        )

        row[
            "right_inverse_factorization_sec"
        ] = seconds_since(
            start
        )

        J = np.asarray(
            mc.right_inverse_pivots,
            dtype=np.int64,
        )

        C = (
            mc.right_inverse_core
        )

        B = GF2Matrix(
            G.arr[
                :,
                J
            ]
        )

        JC_identity = matrix_equal(
            B * C,
            gf2_identity(
                k
            ),
        )

        row[
            "right_inverse_identity"
        ] = JC_identity

        if not JC_identity:
            raise RuntimeError(
                "G[:,J] C != I_k."
            )

        print(
            f"  J shape = {J.shape}"
        )

        print(
            f"  C shape = {C.arr.shape}"
        )

        print(
            "  time    = "
            f"{row['right_inverse_factorization_sec']:.6f} s"
        )

        del B

        gc.collect()

        row[
            "rss_after_right_inverse_mib"
        ] = current_rss_mib()

        # ==========================================================
        # 5. S generation and inversion
        # ==========================================================

        stage = "S_generation"

        print()
        print(
            "[5] Dense scrambling matrix S"
        )

        start = time.perf_counter()

        S_numpy = random_inv_matrix(
            k
        )

        mc.S = GF2Matrix.from_list(
            S_numpy
        )

        row[
            "S_generation_sec"
        ] = seconds_since(
            start
        )

        del S_numpy

        gc.collect()

        stage = "S_inversion"

        start = time.perf_counter()

        mc.S_inv = (
            mc.S.inv()
        )

        row[
            "S_inversion_sec"
        ] = seconds_since(
            start
        )

        row[
            "S_total_sec"
        ] = (
            row[
                "S_generation_sec"
            ]
            +
            row[
                "S_inversion_sec"
            ]
        )

        S_inverse_ok = matrix_equal(
            mc.S * mc.S_inv,
            gf2_identity(
                k
            ),
        )

        row[
            "scrambling_inverse_identity"
        ] = S_inverse_ok

        if not S_inverse_ok:
            raise RuntimeError(
                "S S_inv != I_k."
            )

        print(
            "  generation = "
            f"{row['S_generation_sec']:.6f} s"
        )

        print(
            "  inversion  = "
            f"{row['S_inversion_sec']:.6f} s"
        )

        print(
            "  total      = "
            f"{row['S_total_sec']:.6f} s"
        )

        row[
            "rss_after_S_mib"
        ] = current_rss_mib()

        # ==========================================================
        # Permutation
        # ==========================================================

        stage = "permutation_generation"

        start = time.perf_counter()

        mc.P = np.random.permutation(
            n
        ).astype(
            np.int64
        )

        mc.P_inv = np.argsort(
            mc.P
        ).astype(
            np.int64
        )

        row[
            "permutation_generation_sec"
        ] = seconds_since(
            start
        )

        identity_positions = np.arange(
            n,
            dtype=np.int64,
        )

        if not np.array_equal(
            mc.P[
                mc.P_inv
            ],
            identity_positions,
        ):
            raise RuntimeError(
                "Permutation inverse is invalid."
            )

        # ==========================================================
        # 6. G_pub
        # ==========================================================

        stage = "SG_multiplication"

        print()
        print(
            "[6] Public generator"
        )

        start = time.perf_counter()

        SG = (
            mc.S
            * mc.G
        )

        row[
            "SG_multiplication_sec"
        ] = seconds_since(
            start
        )

        stage = "Gpub_permutation"

        start = time.perf_counter()

        mc.Gp = (
            mc._permute_columns(
                SG,
                mc.P,
            )
        )

        row[
            "Gpub_permutation_sec"
        ] = seconds_since(
            start
        )

        row[
            "Gpub_total_sec"
        ] = (
            row[
                "SG_multiplication_sec"
            ]
            +
            row[
                "Gpub_permutation_sec"
            ]
        )

        expected_Gp = GF2Matrix(
            SG.arr[
                :,
                mc.P
            ]
        )

        Gpub_ok = matrix_equal(
            expected_Gp,
            mc.Gp,
        )

        row[
            "public_generator_identity"
        ] = Gpub_ok

        if not Gpub_ok:
            raise RuntimeError(
                "G_pub != (S G)[:,P]."
            )

        print(
            "  S G       = "
            f"{row['SG_multiplication_sec']:.6f} s"
        )

        print(
            "  permutation = "
            f"{row['Gpub_permutation_sec']:.6f} s"
        )

        print(
            "  total       = "
            f"{row['Gpub_total_sec']:.6f} s"
        )

        del SG
        del expected_Gp

        gc.collect()

        row[
            "rss_after_Gpub_mib"
        ] = current_rss_mib()

        row[
            "construction_wall_sec"
        ] = seconds_since(
            construction_start
        )

        # ==========================================================
        # 11. Object memory/storage
        #
        # Record before ciphertext trials so these measurements are
        # not affected by Patterson's conversion of polynomial data.
        # ==========================================================

        stage = "memory_measurement"

        print()
        print(
            "[11] Matrix memory/storage"
        )

        G_stats = record_matrix_stats(
            row,
            "G",
            mc.G,
        )

        H_stats = record_matrix_stats(
            row,
            "H",
            mc.H,
        )

        S_stats = record_matrix_stats(
            row,
            "S",
            mc.S,
        )

        S_inv_stats = record_matrix_stats(
            row,
            "S_inv",
            mc.S_inv,
        )

        Gpub_stats = record_matrix_stats(
            row,
            "Gpub",
            mc.Gp,
        )

        C_stats = record_matrix_stats(
            row,
            "C",
            mc.right_inverse_core,
        )

        row[
            "J_entries"
        ] = int(
            J.size
        )

        row[
            "J_int64_bytes"
        ] = int(
            J.nbytes
        )

        binary_stats = [
            G_stats,
            H_stats,
            S_stats,
            S_inv_stats,
            Gpub_stats,
            C_stats,
        ]

        row[
            "total_binary_packed_bytes"
        ] = sum(
            stats[
                "packed"
            ]
            for stats
            in binary_stats
        )

        row[
            "total_uint8_plus_J_bytes"
        ] = (
            sum(
                stats[
                    "uint8"
                ]
                for stats
                in binary_stats
            )
            +
            int(
                J.nbytes
            )
        )

        row[
            "total_object_buffer_plus_J_bytes"
        ] = (
            sum(
                stats[
                    "object_buffer"
                ]
                for stats
                in binary_stats
            )
            +
            int(
                J.nbytes
            )
        )

        row[
            "total_estimated_python_plus_J_bytes"
        ] = (
            sum(
                stats[
                    "estimated_python"
                ]
                for stats
                in binary_stats
            )
            +
            int(
                J.nbytes
            )
        )

        (
            private_npz_bytes,
            public_npz_bytes,
        ) = measure_npz_sizes(
            mc,
            g_storage,
            irr_storage,
        )

        row[
            "private_npz_bytes"
        ] = private_npz_bytes

        row[
            "public_npz_bytes"
        ] = public_npz_bytes

        print(
            "  packed binary total = "
            f"{row['total_binary_packed_bytes'] / 2**20:.3f} MiB"
        )

        print(
            "  uint8 + J total      = "
            f"{row['total_uint8_plus_J_bytes'] / 2**20:.3f} MiB"
        )

        print(
            "  object-buffer + J    = "
            f"{row['total_object_buffer_plus_J_bytes'] / 2**20:.3f} MiB"
        )

        print(
            "  estimated Python     = "
            f"{row['total_estimated_python_plus_J_bytes'] / 2**20:.3f} MiB"
        )

        print(
            "  compressed private   = "
            f"{private_npz_bytes / 2**20:.3f} MiB"
        )

        print(
            "  compressed public    = "
            f"{public_npz_bytes / 2**20:.3f} MiB"
        )

        row[
            "rss_before_trials_mib"
        ] = current_rss_mib()

        # ==========================================================
        # 7-10. Encryption / Patterson / extraction / decryption
        # ==========================================================

        stage = "ciphertext_trials"

        print()
        print(
            "[7-10] Ciphertext-operation profiling"
        )

        encryption_times = []
        syndrome_times = []
        patterson_times = []
        extraction_times = []
        unscramble_times = []
        total_decryption_times = []

        successful_trials = 0

        all_extraction_ok = True
        all_patterson_ok = True
        all_decryption_ok = True

        for trial in range(
            1,
            trials + 1,
        ):
            print()
            print(
                f"  Trial {trial}/{trials}"
            )

            message = np.random.randint(
                0,
                2,
                size=k,
                dtype=np.uint8,
            )

            # ------------------------------------------------------
            # 7. Encryption.
            # ------------------------------------------------------

            start = time.perf_counter()

            ciphertext = mc.encrypt(
                message
            )

            encryption_time = seconds_since(
                start
            )

            encryption_times.append(
                encryption_time
            )

            print(
                "    encryption          = "
                f"{encryption_time:.6f} s"
            )

            # ------------------------------------------------------
            # Secret-coordinate received word.
            # ------------------------------------------------------

            secret_ct = (
                mc._permute_vector(
                    ciphertext,
                    mc.P_inv,
                )
            )

            # ------------------------------------------------------
            # Syndrome timing.
            # ------------------------------------------------------

            start = time.perf_counter()

            syndrome = (
                secret_ct
                * mc.H.T()
            )

            syndrome_time = seconds_since(
                start
            )

            syndrome_times.append(
                syndrome_time
            )

            # ------------------------------------------------------
            # 8. Patterson correction.
            #
            # Reset polynomial representation for each trial. This
            # includes the same conversion path a freshly loaded key
            # would encounter.
            # ------------------------------------------------------

            mc.g_poly = (
                g_storage.copy()
            )

            mc.irr_poly = (
                irr_storage.copy()
            )

            working_word = GF2Matrix(
                secret_ct.arr.copy()
            )

            start = time.perf_counter()

            corrected = mc.repair_errors(
                working_word,
                syndrome,
            )

            patterson_time = seconds_since(
                start
            )

            patterson_times.append(
                patterson_time
            )

            repaired_syndrome = (
                corrected
                * mc.H.T()
            )

            patterson_ok = matrix_is_zero(
                repaired_syndrome
            )

            all_patterson_ok = (
                all_patterson_ok
                and
                patterson_ok
            )

            print(
                "    syndrome            = "
                f"{syndrome_time:.6f} s"
            )

            print(
                "    Patterson           = "
                f"{patterson_time:.6f} s "
                f"(valid={patterson_ok})"
            )

            if not patterson_ok:
                raise RuntimeError(
                    "Patterson correction failed."
                )

            # ------------------------------------------------------
            # 9. Factorized extraction.
            #
            #     a = corrected[J] C.
            # ------------------------------------------------------

            start = time.perf_counter()

            extracted = (
                mc._extract_information_vector(
                    corrected
                )
            )

            extraction_time = seconds_since(
                start
            )

            extraction_times.append(
                extraction_time
            )

            expected_scrambled = (
                GF2Matrix.from_list(
                    message
                )
                * mc.S
            )

            extraction_ok = matrix_equal(
                extracted,
                expected_scrambled,
            )

            all_extraction_ok = (
                all_extraction_ok
                and
                extraction_ok
            )

            print(
                "    factor extraction   = "
                f"{extraction_time:.6f} s "
                f"(valid={extraction_ok})"
            )

            if not extraction_ok:
                raise RuntimeError(
                    "Factorized extraction failed."
                )

            # ------------------------------------------------------
            # S^{-1} multiplication.
            # ------------------------------------------------------

            start = time.perf_counter()

            manual_plaintext = (
                extracted
                * mc.S_inv
            )

            unscramble_time = seconds_since(
                start
            )

            unscramble_times.append(
                unscramble_time
            )

            manual_ok = vector_equal(
                manual_plaintext
                .to_numpy()
                .astype(
                    np.uint8
                ),
                message,
            )

            if not manual_ok:
                raise RuntimeError(
                    "S^{-1} message recovery failed."
                )

            # ------------------------------------------------------
            # 10. Total decryption.
            #
            # Reset polynomial storage before timing so every trial
            # represents the cold/private-key-array state.
            # ------------------------------------------------------

            mc.g_poly = (
                g_storage.copy()
            )

            mc.irr_poly = (
                irr_storage.copy()
            )

            start = time.perf_counter()

            decoded = mc.decrypt(
                ciphertext
            )

            total_decryption_time = (
                seconds_since(
                    start
                )
            )

            total_decryption_times.append(
                total_decryption_time
            )

            decryption_ok = vector_equal(
                decoded,
                message,
            )

            all_decryption_ok = (
                all_decryption_ok
                and
                decryption_ok
            )

            if decryption_ok:
                successful_trials += 1

            print(
                "    S^-1 multiply       = "
                f"{unscramble_time:.6f} s"
            )

            print(
                "    total decryption    = "
                f"{total_decryption_time:.6f} s "
                f"(valid={decryption_ok})"
            )

            if not decryption_ok:
                raise RuntimeError(
                    "Total decryption failed."
                )

            del message
            del ciphertext
            del secret_ct
            del syndrome
            del working_word
            del corrected
            del repaired_syndrome
            del extracted
            del expected_scrambled
            del manual_plaintext
            del decoded

            gc.collect()

        # ==========================================================
        # Aggregate operation timings
        # ==========================================================

        (
            row[
                "encryption_avg_sec"
            ],
            row[
                "encryption_min_sec"
            ],
            row[
                "encryption_max_sec"
            ],
        ) = timing_stats(
            encryption_times
        )

        (
            row[
                "patterson_avg_sec"
            ],
            row[
                "patterson_min_sec"
            ],
            row[
                "patterson_max_sec"
            ],
        ) = timing_stats(
            patterson_times
        )

        (
            row[
                "factorized_extraction_avg_sec"
            ],
            row[
                "factorized_extraction_min_sec"
            ],
            row[
                "factorized_extraction_max_sec"
            ],
        ) = timing_stats(
            extraction_times
        )

        (
            row[
                "total_decryption_avg_sec"
            ],
            row[
                "total_decryption_min_sec"
            ],
            row[
                "total_decryption_max_sec"
            ],
        ) = timing_stats(
            total_decryption_times
        )

        row[
            "syndrome_avg_sec"
        ] = float(
            np.mean(
                syndrome_times
            )
        )

        row[
            "unscramble_avg_sec"
        ] = float(
            np.mean(
                unscramble_times
            )
        )

        row[
            "successful_trials"
        ] = successful_trials

        row[
            "factorized_clean_extraction"
        ] = all_extraction_ok

        row[
            "patterson_success"
        ] = all_patterson_ok

        row[
            "decryption_success"
        ] = all_decryption_ok

        row[
            "rss_after_trials_mib"
        ] = current_rss_mib()

        row[
            "peak_rss_mib"
        ] = peak_rss_mib()

        row[
            "status"
        ] = "PASS"

        print()
        print("-" * 88)

        print(
            "PARAMETER-SET SUMMARY"
        )

        print("-" * 88)

        print(
            f"  (m,n,t,k)              = "
            f"({m},{n},{t},{k})"
        )

        print(
            f"  rate                   = "
            f"{row['rate']:.6f}"
        )

        print(
            "  construction wall      = "
            f"{row['construction_wall_sec']:.6f} s"
        )

        print(
            "  encryption avg         = "
            f"{row['encryption_avg_sec']:.6f} s"
        )

        print(
            "  Patterson avg          = "
            f"{row['patterson_avg_sec']:.6f} s"
        )

        print(
            "  extraction avg         = "
            f"{row['factorized_extraction_avg_sec']:.6f} s"
        )

        print(
            "  total decryption avg   = "
            f"{row['total_decryption_avg_sec']:.6f} s"
        )

        print(
            "  current RSS            = "
            f"{row['rss_after_trials_mib']:.2f} MiB"
        )

        print(
            "  peak RSS               = "
            f"{row['peak_rss_mib']:.2f} MiB"
        )

        print(
            f"  correctness            = "
            f"{successful_trials}/{trials} PASS"
        )

        return row

    except Exception as exc:
        row[
            "status"
        ] = "FAIL"

        row[
            "error_stage"
        ] = stage

        row[
            "error_message"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

        row[
            "peak_rss_mib"
        ] = peak_rss_mib()

        print()
        print("!" * 88)

        print(
            "PARAMETER SET FAILED"
        )

        print("!" * 88)

        print(
            f"  stage = {stage}"
        )

        print(
            f"  error = "
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        return row

    finally:
        gc.collect()


# ======================================================================
# Final summary
# ======================================================================

def print_final_summary(
    rows,
    csv_path,
):
    print()
    print("=" * 100)

    print(
        "PHASE 2E SCALING SUMMARY"
    )

    print("=" * 100)

    for row in rows:
        status = row.get(
            "status",
            "UNKNOWN",
        )

        m = row.get(
            "m",
            "?",
        )

        n = row.get(
            "n",
            "?",
        )

        t = row.get(
            "t",
            "?",
        )

        k = row.get(
            "k",
            "?",
        )

        if status == "PASS":
            construction = float(
                row[
                    "construction_wall_sec"
                ]
            )

            decryption = float(
                row[
                    "total_decryption_avg_sec"
                ]
            )

            rss = float(
                row[
                    "peak_rss_mib"
                ]
            )

            print(
                f"(m,n,t)=({m},{n},{t}) "
                f"k={k} "
                f"construction={construction:.3f}s "
                f"decrypt={decryption:.6f}s "
                f"peakRSS={rss:.1f}MiB "
                f"PASS"
            )

        else:
            print(
                f"(m,n,t)=({m},{n},{t}) "
                f"k={k} "
                f"FAIL at "
                f"{row.get('error_stage', '?')}: "
                f"{row.get('error_message', '')}"
            )

    print()
    print(
        "CSV results:"
    )

    print(
        f"  {Path(csv_path).resolve()}"
    )


# ======================================================================
# Command-line interface
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 2E detailed McEliece scaling profiler."
        )
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=(
            "Run only one parameter-set index "
            "(0,1,2,3). Default: run all."
        ),
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV,
        help=(
            "CSV output path."
        ),
    )

    parser.add_argument(
        "--reset-csv",
        action="store_true",
        help=(
            "Delete the existing CSV before running."
        ),
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use one ciphertext trial for every "
            "parameter set."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    csv_path = Path(
        args.csv
    )

    if (
        args.reset_csv
        and
        csv_path.exists()
    ):
        csv_path.unlink()

    if args.index is not None:

        matching = [
            params
            for params
            in PARAMETER_SETS
            if params[0] == args.index
        ]

        if not matching:
            raise ValueError(
                "index must be one of "
                "0, 1, 2, 3."
            )

        selected_sets = matching

    else:
        selected_sets = (
            PARAMETER_SETS
        )

    rows = []

    for (
        index,
        m,
        n,
        t,
        configured_trials,
    ) in selected_sets:

        if args.quick:
            trials = 1
        else:
            trials = (
                configured_trials
            )

        row = profile_parameter_set(
            index=index,
            m=m,
            n=n,
            t=t,
            trials=trials,
            seed=(
                BASE_SEED
                +
                index
            ),
        )

        append_csv_row(
            csv_path,
            row,
        )

        rows.append(
            row
        )

        print()
        print(
            "Result written to CSV."
        )

        # Release parameter-specific objects before continuing.
        gc.collect()

    print_final_summary(
        rows,
        csv_path,
    )


if __name__ == "__main__":
    main()
"""
Phase 2E.1 detailed scaling and Goppa-polynomial profiler.

This experiment extends Phase 2E by decomposing

    _generate_goppa_polynomial()

into its internal timing components.

No cryptographic construction is changed.

Measured Goppa-polynomial components
------------------------------------

1. Candidate generation:
       irreducible_poly_ext_candidate(...)

2. Cheap candidate rejection:
       candidate(0) == 0
       candidate(1) == 0

3. Full extension-field root search:
       first_alpha_power_root(...)

4. Alpha-power representation conversion:
       reduce_to_alpha_power(...)

5. Final support-wide verification:
       g(L_j) != 0 for every support element

6. Candidate counters:
       candidate attempts
       cheap rejections
       root-search calls
       root-search rejections
       accepted attempt

The original Phase 2E measurements are retained:

1.  Goppa field/support/polynomial generation.
2.  H construction.
3.  Nullspace computation for G.
4.  Factorized right inverse J,C.
5.  Dense S generation and inversion.
6.  G_pub construction.
7.  Encryption.
8.  Patterson correction.
9.  Factorized extraction c[J]C.
10. Total decryption.
11. Matrix/key memory and storage.

Results are written incrementally to

    phase2e1_goppa_profile_results.csv.
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

from sympy.abc import alpha

from goppa.goppacodegenerator import (
    GoppaCodeGenerator,
)

from mceliece.mceliececipher import (
    McElieceCipher,
)

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


DEFAULT_CSV = (
    "phase2e1_goppa_profile_results.csv"
)

BASE_SEED = 20260816


# ======================================================================
# CSV columns
# ======================================================================

CSV_FIELDS = [
    # ------------------------------------------------------------
    # Run information
    # ------------------------------------------------------------

    "run_time",
    "parameter_index",
    "status",
    "error_stage",
    "error_message",
    "seed",

    # ------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------

    "m",
    "n",
    "t",
    "k",
    "mt",
    "dimension_lower_bound",
    "rate",

    # ------------------------------------------------------------
    # Stage 1: field/support/Goppa polynomial
    # ------------------------------------------------------------

    "field_generation_sec",
    "support_generation_sec",

    "goppa_polynomial_sec",

    # Phase 2E.1 subprofile
    "goppa_candidate_attempts",
    "goppa_candidate_accepted_attempt",

    "goppa_candidate_generation_sec",

    "goppa_cheap_rejection_sec",
    "goppa_cheap_rejections",

    "goppa_first_root_sec",
    "goppa_first_root_calls",
    "goppa_first_root_rejections",
    "goppa_first_root_avg_sec",
    "goppa_first_root_max_sec",

    "goppa_candidate_loop_sec",

    "goppa_reduce_to_alpha_power_sec",
    "goppa_degree_check_sec",

    "goppa_final_support_verification_sec",
    "goppa_final_support_points",

    "goppa_profile_total_sec",

    # Difference between externally timed method and accounted
    # profiler total/subcomponents.
    "goppa_external_minus_profile_sec",

    # Component percentages of externally measured
    # goppa_polynomial_sec.
    "goppa_candidate_generation_pct",
    "goppa_cheap_rejection_pct",
    "goppa_first_root_pct",
    "goppa_reduce_to_alpha_power_pct",
    "goppa_final_support_verification_pct",

    # Remaining Stage 1 measurements
    "goppa_storage_conversion_sec",
    "goppa_support_total_sec",

    # ------------------------------------------------------------
    # Stage 2: H
    # ------------------------------------------------------------

    "H_extension_sec",
    "H_binary_expansion_sec",
    "H_total_sec",

    # ------------------------------------------------------------
    # Stage 3: G
    # ------------------------------------------------------------

    "nullspace_sec",
    "generator_finalize_sec",
    "generator_total_sec",
    "rank_H",

    # ------------------------------------------------------------
    # Stage 4: J,C
    # ------------------------------------------------------------

    "right_inverse_factorization_sec",

    # ------------------------------------------------------------
    # Stage 5: S
    # ------------------------------------------------------------

    "S_generation_sec",
    "S_inversion_sec",
    "S_total_sec",

    # ------------------------------------------------------------
    # Permutation
    # ------------------------------------------------------------

    "permutation_generation_sec",

    # ------------------------------------------------------------
    # Stage 6: G_pub
    # ------------------------------------------------------------

    "SG_multiplication_sec",
    "Gpub_permutation_sec",
    "Gpub_total_sec",

    # ------------------------------------------------------------
    # Total construction
    # ------------------------------------------------------------

    "construction_wall_sec",

    # ------------------------------------------------------------
    # Stages 7-10
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Matrix dimensions
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Logical bit-packed storage
    # ------------------------------------------------------------

    "G_logical_packed_bytes",
    "H_logical_packed_bytes",
    "S_logical_packed_bytes",
    "S_inv_logical_packed_bytes",
    "Gpub_logical_packed_bytes",
    "C_logical_packed_bytes",

    # ------------------------------------------------------------
    # uint8 serialized storage
    # ------------------------------------------------------------

    "G_uint8_bytes",
    "H_uint8_bytes",
    "S_uint8_bytes",
    "S_inv_uint8_bytes",
    "Gpub_uint8_bytes",
    "C_uint8_bytes",
    "J_int64_bytes",

    # ------------------------------------------------------------
    # NumPy object-array buffers
    # ------------------------------------------------------------

    "G_object_buffer_bytes",
    "H_object_buffer_bytes",
    "S_object_buffer_bytes",
    "S_inv_object_buffer_bytes",
    "Gpub_object_buffer_bytes",
    "C_object_buffer_bytes",

    # ------------------------------------------------------------
    # Approximate Python representation
    # ------------------------------------------------------------

    "G_estimated_python_bytes",
    "H_estimated_python_bytes",
    "S_estimated_python_bytes",
    "S_inv_estimated_python_bytes",
    "Gpub_estimated_python_bytes",
    "C_estimated_python_bytes",

    # ------------------------------------------------------------
    # Aggregate storage
    # ------------------------------------------------------------

    "total_binary_packed_bytes",
    "total_uint8_plus_J_bytes",
    "total_object_buffer_plus_J_bytes",
    "total_estimated_python_plus_J_bytes",

    # ------------------------------------------------------------
    # Actual compressed NPZ sizes
    # ------------------------------------------------------------

    "private_npz_bytes",
    "public_npz_bytes",

    # ------------------------------------------------------------
    # Process memory diagnostics
    # ------------------------------------------------------------

    "rss_after_goppa_mib",
    "rss_after_H_mib",
    "rss_after_G_mib",
    "rss_after_right_inverse_mib",
    "rss_after_S_mib",
    "rss_after_Gpub_mib",
    "rss_before_trials_mib",
    "rss_after_trials_mib",
    "peak_rss_mib",

    # ------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------

    "GHt_zero",
    "right_inverse_identity",
    "scrambling_inverse_identity",
    "public_generator_identity",
    "factorized_clean_extraction",
    "patterson_success",
    "decryption_success",
]


# ======================================================================
# Generic helpers
# ======================================================================

def now_string():
    return time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def seconds_since(
    start,
):
    return (
        time.perf_counter()
        -
        start
    )


def gf2_numpy(
    matrix,
):
    """
    Convert GF2Matrix to uint8 while retaining shape.
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
            for x in (
                matrix
                .arr
                .flat
            )
        ],
        dtype=np.uint8,
    ).reshape(
        matrix.arr.shape
    )


def matrix_equal(
    A,
    B,
):
    if (
        A.arr.shape
        !=
        B.arr.shape
    ):
        return False

    return np.array_equal(
        gf2_numpy(
            A
        ),
        gf2_numpy(
            B
        ),
    )


def matrix_is_zero(
    matrix,
):
    return not np.any(
        gf2_numpy(
            matrix
        )
    )


def gf2_identity(
    size,
):
    return GF2Matrix.from_list(
        np.eye(
            size,
            dtype=np.uint8,
        )
    )


def vector_equal(
    A,
    B,
):
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


def timing_stats(
    values,
):
    if not values:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    return (
        float(
            np.mean(
                values
            )
        ),
        float(
            np.min(
                values
            )
        ),
        float(
            np.max(
                values
            )
        ),
    )


def percentage(
    component,
    total,
):
    if (
        total is None
        or
        total <= 0
    ):
        return float(
            "nan"
        )

    return (
        100.0
        *
        float(
            component
        )
        /
        float(
            total
        )
    )


# ======================================================================
# Process memory
# ======================================================================

def current_rss_mib():
    """
    Current Linux resident-set size.
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
                        kb
                        /
                        1024.0
                    )

    except Exception:
        pass

    return float(
        "nan"
    )


def peak_rss_mib():
    """
    Peak process resident memory.

    Linux reports ru_maxrss in KiB.
    """

    try:
        usage = (
            resource
            .getrusage(
                resource.RUSAGE_SELF
            )
        )

        return (
            float(
                usage.ru_maxrss
            )
            /
            1024.0
        )

    except Exception:
        return float(
            "nan"
        )


# ======================================================================
# Matrix memory measurements
# ======================================================================

def logical_packed_bytes(
    entries,
):
    return (
        int(
            entries
        )
        +
        7
    ) // 8


def matrix_memory_stats(
    matrix,
):
    """
    Measure several representations of a GF2Matrix.

    packed:
        theoretical bit-packed size.

    uint8:
        one byte per bit.

    object_buffer:
        NumPy object-array pointer storage only.

    estimated_python:
        pointer storage plus sys.getsizeof() for each GF2 object.
    """

    if not isinstance(
        matrix,
        GF2Matrix,
    ):
        raise TypeError(
            "matrix_memory_stats expects "
            "GF2Matrix."
        )

    entries = int(
        matrix.arr.size
    )

    pointer_buffer = int(
        matrix.arr.nbytes
    )

    if entries:
        one_object_size = int(
            sys.getsizeof(
                matrix
                .arr
                .flat[0]
            )
        )

    else:
        one_object_size = 0

    estimated_python = (
        pointer_buffer
        +
        entries
        *
        one_object_size
    )

    return {
        "entries": entries,
        "packed": (
            logical_packed_bytes(
                entries
            )
        ),
        "uint8": entries,
        "object_buffer": (
            pointer_buffer
        ),
        "estimated_python": (
            estimated_python
        ),
    }


def record_matrix_stats(
    row,
    prefix,
    matrix,
):
    stats = (
        matrix_memory_stats(
            matrix
        )
    )

    shape = (
        matrix
        .arr
        .shape
    )

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
# CSV
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
        csv_path.stat().st_size
        >
        0
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
            for field
            in CSV_FIELDS
        }

        writer.writerow(
            complete_row
        )

        file.flush()

        os.fsync(
            file.fileno()
        )


# ======================================================================
# Polynomial-storage conversion
# ======================================================================

def convert_polynomials_for_cipher(
    g_poly,
    irr_poly,
):
    g_coeffs = []

    for coefficient in (
        g_poly
        .all_coeffs()[::-1]
    ):
        bits = (
            get_binary_from_alpha(
                coefficient,
                irr_poly,
                2,
            )
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
            for c in (
                irr_poly
                .all_coeffs()[::-1]
            )
        ],
        dtype=np.uint8,
    )

    return (
        g_storage,
        irr_storage,
    )


# ======================================================================
# NPZ-size measurement
# ======================================================================

def measure_npz_sizes(
    mc,
    g_storage,
    irr_storage,
):
    """
    Measure actual compressed Phase-2D key representation.
    """

    with tempfile.TemporaryDirectory(
        prefix="phase2e1_storage_"
    ) as temp_dir:

        temp_dir = Path(
            temp_dir
        )

        private_path = (
            temp_dir
            /
            "private.npz"
        )

        public_path = (
            temp_dir
            /
            "public.npz"
        )

        np.savez_compressed(
            private_path,

            format_version=np.asarray(
                4,
                dtype=np.int64,
            ),

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

            right_inverse_pivots=np.asarray(
                mc.right_inverse_pivots,
                dtype=np.int64,
            ),

            right_inverse_core=gf2_numpy(
                mc.right_inverse_core
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

            format_version=np.asarray(
                4,
                dtype=np.int64,
            ),

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
                private_path
                .stat()
                .st_size
            ),
            int(
                public_path
                .stat()
                .st_size
            ),
        )


# ======================================================================
# Goppa-profile transfer
# ======================================================================

def copy_goppa_profile_to_row(
    row,
    profile,
    external_total,
):
    """
    Copy GoppaCodeGenerator.goppa_profile into the CSV row.
    """

    row[
        "goppa_candidate_attempts"
    ] = int(
        profile.get(
            "candidate_attempts",
            0,
        )
    )

    row[
        "goppa_candidate_accepted_attempt"
    ] = int(
        profile.get(
            "candidate_accepted_attempt",
            0,
        )
    )

    row[
        "goppa_candidate_generation_sec"
    ] = float(
        profile.get(
            "candidate_generation_sec",
            0.0,
        )
    )

    row[
        "goppa_cheap_rejection_sec"
    ] = float(
        profile.get(
            "cheap_rejection_sec",
            0.0,
        )
    )

    row[
        "goppa_cheap_rejections"
    ] = int(
        profile.get(
            "cheap_rejections",
            0,
        )
    )

    row[
        "goppa_first_root_sec"
    ] = float(
        profile.get(
            "first_alpha_power_root_sec",
            0.0,
        )
    )

    row[
        "goppa_first_root_calls"
    ] = int(
        profile.get(
            "first_alpha_power_root_calls",
            0,
        )
    )

    row[
        "goppa_first_root_rejections"
    ] = int(
        profile.get(
            "first_alpha_power_root_rejections",
            0,
        )
    )

    row[
        "goppa_first_root_max_sec"
    ] = float(
        profile.get(
            "first_alpha_power_root_max_sec",
            0.0,
        )
    )

    root_calls = row[
        "goppa_first_root_calls"
    ]

    if root_calls:
        row[
            "goppa_first_root_avg_sec"
        ] = (
            row[
                "goppa_first_root_sec"
            ]
            /
            root_calls
        )

    else:
        row[
            "goppa_first_root_avg_sec"
        ] = 0.0

    row[
        "goppa_candidate_loop_sec"
    ] = float(
        profile.get(
            "candidate_loop_sec",
            0.0,
        )
    )

    row[
        "goppa_reduce_to_alpha_power_sec"
    ] = float(
        profile.get(
            "reduce_to_alpha_power_sec",
            0.0,
        )
    )

    row[
        "goppa_degree_check_sec"
    ] = float(
        profile.get(
            "degree_check_sec",
            0.0,
        )
    )

    row[
        "goppa_final_support_verification_sec"
    ] = float(
        profile.get(
            "final_support_verification_sec",
            0.0,
        )
    )

    row[
        "goppa_final_support_points"
    ] = int(
        profile.get(
            "final_support_points",
            0,
        )
    )

    row[
        "goppa_profile_total_sec"
    ] = float(
        profile.get(
            "total_sec",
            0.0,
        )
    )

    row[
        "goppa_external_minus_profile_sec"
    ] = (
        float(
            external_total
        )
        -
        row[
            "goppa_profile_total_sec"
        ]
    )

    row[
        "goppa_candidate_generation_pct"
    ] = percentage(
        row[
            "goppa_candidate_generation_sec"
        ],
        external_total,
    )

    row[
        "goppa_cheap_rejection_pct"
    ] = percentage(
        row[
            "goppa_cheap_rejection_sec"
        ],
        external_total,
    )

    row[
        "goppa_first_root_pct"
    ] = percentage(
        row[
            "goppa_first_root_sec"
        ],
        external_total,
    )

    row[
        "goppa_reduce_to_alpha_power_pct"
    ] = percentage(
        row[
            "goppa_reduce_to_alpha_power_sec"
        ],
        external_total,
    )

    row[
        "goppa_final_support_verification_pct"
    ] = percentage(
        row[
            "goppa_final_support_verification_sec"
        ],
        external_total,
    )


# ======================================================================
# Profile one parameter set
# ======================================================================

def profile_parameter_set(
    index,
    m,
    n,
    t,
    trials,
    seed,
):
    row = {
        "run_time": (
            now_string()
        ),
        "parameter_index": (
            index
        ),
        "status": (
            "RUNNING"
        ),
        "error_stage": "",
        "error_message": "",
        "seed": seed,

        "m": m,
        "n": n,
        "t": t,
        "mt": (
            m * t
        ),
        "dimension_lower_bound": (
            n
            -
            m * t
        ),
        "trials": trials,
    }

    stage = (
        "initialization"
    )

    print()
    print(
        "=" * 96
    )

    print(
        "PHASE 2E.1 PROFILE "
        f"index={index}: "
        f"m={m}, n={n}, t={t}, "
        f"trials={trials}"
    )

    print(
        "=" * 96
    )

    np.random.seed(
        seed
    )

    construction_start = (
        time.perf_counter()
    )

    try:
        # ==========================================================
        # 1a. Extension field
        # ==========================================================

        stage = (
            "field_generation"
        )

        generator = (
            GoppaCodeGenerator(
                m,
                n,
                t,
            )
        )

        start = (
            time.perf_counter()
        )

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

        # ==========================================================
        # 1b. Support
        # ==========================================================

        stage = (
            "support_generation"
        )

        start = (
            time.perf_counter()
        )

        support_exponents = list(
            range(
                n
            )
        )

        support = [
            alpha ** exponent
            for exponent
            in support_exponents
        ]

        generator.support_exponents = (
            support_exponents
        )

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
            f"  length = "
            f"{len(support)}"
        )

        print(
            "  time   = "
            f"{row['support_generation_sec']:.6f} s"
        )

        # ==========================================================
        # 1c. Goppa polynomial
        # ==========================================================

        stage = (
            "goppa_polynomial"
        )

        start = (
            time.perf_counter()
        )

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

        copy_goppa_profile_to_row(
            row,
            generator.goppa_profile,
            row[
                "goppa_polynomial_sec"
            ],
        )

        print()
        print(
            "[1c] Goppa polynomial"
        )

        print(
            f"  degree = "
            f"{g_poly.degree()}"
        )

        print(
            "  total external time     = "
            f"{row['goppa_polynomial_sec']:.6f} s"
        )

        print()
        print(
            "  Phase 2E.1 subprofile:"
        )

        print(
            "    candidate attempts    = "
            f"{row['goppa_candidate_attempts']}"
        )

        print(
            "    accepted attempt      = "
            f"{row['goppa_candidate_accepted_attempt']}"
        )

        print(
            "    candidate generation  = "
            f"{row['goppa_candidate_generation_sec']:.6f} s "
            f"("
            f"{row['goppa_candidate_generation_pct']:.2f}%"
            f")"
        )

        print(
            "    cheap rejection       = "
            f"{row['goppa_cheap_rejection_sec']:.6f} s "
            f"("
            f"{row['goppa_cheap_rejections']} rejects, "
            f"{row['goppa_cheap_rejection_pct']:.2f}%"
            f")"
        )

        print(
            "    first-root search     = "
            f"{row['goppa_first_root_sec']:.6f} s "
            f"("
            f"{row['goppa_first_root_calls']} calls, "
            f"{row['goppa_first_root_pct']:.2f}%"
            f")"
        )

        print(
            "    root-search average   = "
            f"{row['goppa_first_root_avg_sec']:.6f} s"
        )

        print(
            "    root-search maximum   = "
            f"{row['goppa_first_root_max_sec']:.6f} s"
        )

        print(
            "    root rejections       = "
            f"{row['goppa_first_root_rejections']}"
        )

        print(
            "    candidate loop total  = "
            f"{row['goppa_candidate_loop_sec']:.6f} s"
        )

        print(
            "    alpha-power reduction = "
            f"{row['goppa_reduce_to_alpha_power_sec']:.6f} s "
            f"("
            f"{row['goppa_reduce_to_alpha_power_pct']:.2f}%"
            f")"
        )

        print(
            "    support verification  = "
            f"{row['goppa_final_support_verification_sec']:.6f} s "
            f"("
            f"{row['goppa_final_support_points']} points, "
            f"{row['goppa_final_support_verification_pct']:.2f}%"
            f")"
        )

        print(
            "    internal profile      = "
            f"{row['goppa_profile_total_sec']:.6f} s"
        )

        print(
            "    external-profile diff = "
            f"{row['goppa_external_minus_profile_sec']:.6f} s"
        )

        # ==========================================================
        # Primitive storage conversion
        # ==========================================================

        stage = (
            "goppa_storage_conversion"
        )

        start = (
            time.perf_counter()
        )

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

        stage = (
            "H_extension"
        )

        print()
        print(
            "[2] Parity-check construction"
        )

        start = (
            time.perf_counter()
        )

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

        stage = (
            "H_binary_expansion"
        )

        start = (
            time.perf_counter()
        )

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
            "  extension H     = "
            f"{row['H_extension_sec']:.6f} s"
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
            f"  H shape          = "
            f"{H.arr.shape}"
        )

        row[
            "rss_after_H_mib"
        ] = current_rss_mib()

        del H_ext

        gc.collect()

        # ==========================================================
        # 3. Nullspace -> G
        # ==========================================================

        stage = (
            "nullspace"
        )

        print()
        print(
            "[3] Nullspace / generator"
        )

        start = (
            time.perf_counter()
        )

        (
            H_nullspace,
            nullity,
        ) = H.nullspace()

        row[
            "nullspace_sec"
        ] = seconds_since(
            start
        )

        stage = (
            "generator_finalize"
        )

        start = (
            time.perf_counter()
        )

        G = GF2Matrix(
            H_nullspace
            .T()[
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
            "  nullspace           = "
            f"{row['nullspace_sec']:.6f} s"
        )

        print(
            "  generator finalize  = "
            f"{row['generator_finalize_sec']:.6f} s"
        )

        print(
            f"  G shape             = "
            f"{G.arr.shape}"
        )

        GHt = (
            G
            *
            H.T()
        )

        GHt_zero = (
            matrix_is_zero(
                GHt
            )
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
        # Assemble McEliece object from profiled components
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
        # 4. Factorized right inverse
        # ==========================================================

        stage = (
            "right_inverse_factorization"
        )

        print()
        print(
            "[4] Factorized right inverse"
        )

        start = (
            time.perf_counter()
        )

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

        right_inverse_ok = (
            matrix_equal(
                B * C,
                gf2_identity(
                    k
                ),
            )
        )

        row[
            "right_inverse_identity"
        ] = right_inverse_ok

        if not right_inverse_ok:
            raise RuntimeError(
                "G[:,J] C != I_k."
            )

        print(
            f"  J shape = "
            f"{J.shape}"
        )

        print(
            f"  C shape = "
            f"{C.arr.shape}"
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
        # 5. Dense S
        # ==========================================================

        stage = (
            "S_generation"
        )

        print()
        print(
            "[5] Dense scrambling matrix S"
        )

        start = (
            time.perf_counter()
        )

        S_numpy = (
            random_inv_matrix(
                k
            )
        )

        mc.S = (
            GF2Matrix.from_list(
                S_numpy
            )
        )

        row[
            "S_generation_sec"
        ] = seconds_since(
            start
        )

        del S_numpy

        gc.collect()

        stage = (
            "S_inversion"
        )

        start = (
            time.perf_counter()
        )

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

        scrambling_ok = (
            matrix_equal(
                mc.S
                *
                mc.S_inv,

                gf2_identity(
                    k
                ),
            )
        )

        row[
            "scrambling_inverse_identity"
        ] = scrambling_ok

        if not scrambling_ok:
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

        stage = (
            "permutation_generation"
        )

        start = (
            time.perf_counter()
        )

        mc.P = (
            np.random
            .permutation(
                n
            )
            .astype(
                np.int64
            )
        )

        mc.P_inv = (
            np.argsort(
                mc.P
            )
            .astype(
                np.int64
            )
        )

        row[
            "permutation_generation_sec"
        ] = seconds_since(
            start
        )

        identity_positions = (
            np.arange(
                n,
                dtype=np.int64,
            )
        )

        if not np.array_equal(
            mc.P[
                mc.P_inv
            ],
            identity_positions,
        ):
            raise RuntimeError(
                "Permutation inverse failed."
            )

        # ==========================================================
        # 6. Public generator
        # ==========================================================

        stage = (
            "SG_multiplication"
        )

        print()
        print(
            "[6] Public generator"
        )

        start = (
            time.perf_counter()
        )

        SG = (
            mc.S
            *
            mc.G
        )

        row[
            "SG_multiplication_sec"
        ] = seconds_since(
            start
        )

        stage = (
            "Gpub_permutation"
        )

        start = (
            time.perf_counter()
        )

        mc.Gp = (
            mc
            ._permute_columns(
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

        public_generator_ok = (
            matrix_equal(
                expected_Gp,
                mc.Gp,
            )
        )

        row[
            "public_generator_identity"
        ] = public_generator_ok

        if not public_generator_ok:
            raise RuntimeError(
                "G_pub != (SG)[:,P]."
            )

        print(
            "  S G         = "
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
        # 11. Memory/storage
        # ==========================================================

        stage = (
            "memory_measurement"
        )

        print()
        print(
            "[11] Matrix memory/storage"
        )

        G_stats = (
            record_matrix_stats(
                row,
                "G",
                mc.G,
            )
        )

        H_stats = (
            record_matrix_stats(
                row,
                "H",
                mc.H,
            )
        )

        S_stats = (
            record_matrix_stats(
                row,
                "S",
                mc.S,
            )
        )

        S_inv_stats = (
            record_matrix_stats(
                row,
                "S_inv",
                mc.S_inv,
            )
        )

        Gpub_stats = (
            record_matrix_stats(
                row,
                "Gpub",
                mc.Gp,
            )
        )

        C_stats = (
            record_matrix_stats(
                row,
                "C",
                mc.right_inverse_core,
            )
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
            item[
                "packed"
            ]
            for item
            in binary_stats
        )

        row[
            "total_uint8_plus_J_bytes"
        ] = (
            sum(
                item[
                    "uint8"
                ]
                for item
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
                item[
                    "object_buffer"
                ]
                for item
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
                item[
                    "estimated_python"
                ]
                for item
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
        ) = (
            measure_npz_sizes(
                mc,
                g_storage,
                irr_storage,
            )
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
        # 7-10. Ciphertext operations
        # ==========================================================

        stage = (
            "ciphertext_trials"
        )

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
                f"  Trial "
                f"{trial}/{trials}"
            )

            message = (
                np.random
                .randint(
                    0,
                    2,
                    size=k,
                    dtype=np.uint8,
                )
            )

            # ------------------------------------------------------
            # 7. Encryption
            # ------------------------------------------------------

            start = (
                time.perf_counter()
            )

            ciphertext = (
                mc.encrypt(
                    message
                )
            )

            encryption_time = (
                seconds_since(
                    start
                )
            )

            encryption_times.append(
                encryption_time
            )

            # ------------------------------------------------------
            # Secret-coordinate ciphertext
            # ------------------------------------------------------

            secret_ct = (
                mc
                ._permute_vector(
                    ciphertext,
                    mc.P_inv,
                )
            )

            # ------------------------------------------------------
            # Syndrome
            # ------------------------------------------------------

            start = (
                time.perf_counter()
            )

            syndrome = (
                secret_ct
                *
                mc.H.T()
            )

            syndrome_time = (
                seconds_since(
                    start
                )
            )

            syndrome_times.append(
                syndrome_time
            )

            # ------------------------------------------------------
            # 8. Patterson
            # ------------------------------------------------------

            mc.g_poly = (
                g_storage.copy()
            )

            mc.irr_poly = (
                irr_storage.copy()
            )

            working_word = GF2Matrix(
                secret_ct
                .arr
                .copy()
            )

            start = (
                time.perf_counter()
            )

            corrected = (
                mc.repair_errors(
                    working_word,
                    syndrome,
                )
            )

            patterson_time = (
                seconds_since(
                    start
                )
            )

            patterson_times.append(
                patterson_time
            )

            repaired_syndrome = (
                corrected
                *
                mc.H.T()
            )

            patterson_ok = (
                matrix_is_zero(
                    repaired_syndrome
                )
            )

            all_patterson_ok = (
                all_patterson_ok
                and
                patterson_ok
            )

            if not patterson_ok:
                raise RuntimeError(
                    "Patterson correction failed."
                )

            # ------------------------------------------------------
            # 9. Factorized extraction
            # ------------------------------------------------------

            start = (
                time.perf_counter()
            )

            extracted = (
                mc
                ._extract_information_vector(
                    corrected
                )
            )

            extraction_time = (
                seconds_since(
                    start
                )
            )

            extraction_times.append(
                extraction_time
            )

            expected_scrambled = (
                GF2Matrix.from_list(
                    message
                )
                *
                mc.S
            )

            extraction_ok = (
                matrix_equal(
                    extracted,
                    expected_scrambled,
                )
            )

            all_extraction_ok = (
                all_extraction_ok
                and
                extraction_ok
            )

            if not extraction_ok:
                raise RuntimeError(
                    "Factorized extraction failed."
                )

            # ------------------------------------------------------
            # S^{-1}
            # ------------------------------------------------------

            start = (
                time.perf_counter()
            )

            manual_plaintext = (
                extracted
                *
                mc.S_inv
            )

            unscramble_time = (
                seconds_since(
                    start
                )
            )

            unscramble_times.append(
                unscramble_time
            )

            manual_ok = (
                vector_equal(
                    manual_plaintext
                    .to_numpy()
                    .astype(
                        np.uint8
                    ),
                    message,
                )
            )

            if not manual_ok:
                raise RuntimeError(
                    "S^{-1} recovery failed."
                )

            # ------------------------------------------------------
            # 10. Full decrypt
            # ------------------------------------------------------

            mc.g_poly = (
                g_storage.copy()
            )

            mc.irr_poly = (
                irr_storage.copy()
            )

            start = (
                time.perf_counter()
            )

            decoded = (
                mc.decrypt(
                    ciphertext
                )
            )

            total_decryption_time = (
                seconds_since(
                    start
                )
            )

            total_decryption_times.append(
                total_decryption_time
            )

            decryption_ok = (
                vector_equal(
                    decoded,
                    message,
                )
            )

            all_decryption_ok = (
                all_decryption_ok
                and
                decryption_ok
            )

            if decryption_ok:
                successful_trials += 1

            print(
                "    encryption         = "
                f"{encryption_time:.6f} s"
            )

            print(
                "    syndrome           = "
                f"{syndrome_time:.6f} s"
            )

            print(
                "    Patterson          = "
                f"{patterson_time:.6f} s "
                f"(valid={patterson_ok})"
            )

            print(
                "    factor extraction  = "
                f"{extraction_time:.6f} s "
                f"(valid={extraction_ok})"
            )

            print(
                "    S^-1 multiply      = "
                f"{unscramble_time:.6f} s"
            )

            print(
                "    total decryption   = "
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

        # ==========================================================
        # Summary
        # ==========================================================

        print()
        print(
            "-" * 96
        )

        print(
            "PARAMETER-SET SUMMARY"
        )

        print(
            "-" * 96
        )

        print(
            f"  (m,n,t,k)            = "
            f"({m},{n},{t},{k})"
        )

        print(
            f"  rate                 = "
            f"{row['rate']:.6f}"
        )

        print(
            "  Goppa polynomial     = "
            f"{row['goppa_polynomial_sec']:.6f} s"
        )

        print(
            "  first-root fraction  = "
            f"{row['goppa_first_root_pct']:.2f}%"
        )

        print(
            "  support-check frac.  = "
            f"{row['goppa_final_support_verification_pct']:.2f}%"
        )

        print(
            "  construction wall    = "
            f"{row['construction_wall_sec']:.6f} s"
        )

        print(
            "  encryption avg       = "
            f"{row['encryption_avg_sec']:.6f} s"
        )

        print(
            "  Patterson avg        = "
            f"{row['patterson_avg_sec']:.6f} s"
        )

        print(
            "  extraction avg       = "
            f"{row['factorized_extraction_avg_sec']:.6f} s"
        )

        print(
            "  total decrypt avg    = "
            f"{row['total_decryption_avg_sec']:.6f} s"
        )

        print(
            "  peak RSS             = "
            f"{row['peak_rss_mib']:.2f} MiB"
        )

        print(
            f"  correctness          = "
            f"{successful_trials}/"
            f"{trials} PASS"
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
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        row[
            "peak_rss_mib"
        ] = peak_rss_mib()

        # If failure occurred inside Goppa generation, preserve
        # whatever partial profile was accumulated.
        if (
            "generator"
            in locals()
            and
            hasattr(
                generator,
                "goppa_profile",
            )
        ):
            try:
                external_total = row.get(
                    "goppa_polynomial_sec",
                    generator
                    .goppa_profile
                    .get(
                        "total_sec",
                        0.0,
                    ),
                )

                copy_goppa_profile_to_row(
                    row,
                    generator.goppa_profile,
                    external_total,
                )

            except Exception:
                pass

        print()
        print(
            "!" * 96
        )

        print(
            "PARAMETER SET FAILED"
        )

        print(
            "!" * 96
        )

        print(
            f"  stage = {stage}"
        )

        print(
            "  error = "
            f"{type(exc).__name__}: "
            f"{exc}"
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
    print(
        "=" * 110
    )

    print(
        "PHASE 2E.1 GOPPA-POLYNOMIAL "
        "PROFILING SUMMARY"
    )

    print(
        "=" * 110
    )

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
            print(
                f"(m,n,t)="
                f"({m},{n},{t}) "
                f"k={k} "
                f"g={float(row['goppa_polynomial_sec']):.3f}s "
                f"root="
                f"{float(row['goppa_first_root_pct']):.1f}% "
                f"verify="
                f"{float(row['goppa_final_support_verification_pct']):.1f}% "
                f"attempts="
                f"{row['goppa_candidate_attempts']} "
                f"PASS"
            )

        else:
            print(
                f"(m,n,t)="
                f"({m},{n},{t}) "
                f"FAIL at "
                f"{row.get('error_stage', '?')}: "
                f"{row.get('error_message', '')}"
            )

    print()
    print(
        "CSV results:"
    )

    print(
        f"  "
        f"{Path(csv_path).resolve()}"
    )


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2E.1 detailed "
            "Goppa-polynomial profiler."
        )
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=(
            "Run one parameter-set index "
            "(0,1,2,3). "
            "Default: run all."
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
            "Delete existing output CSV "
            "before running."
        ),
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use one ciphertext trial."
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

    if args.index is None:
        selected_sets = (
            PARAMETER_SETS
        )

    else:
        selected_sets = [
            params
            for params
            in PARAMETER_SETS
            if params[0]
            ==
            args.index
        ]

        if not selected_sets:
            raise ValueError(
                "--index must be "
                "0, 1, 2, or 3."
            )

    rows = []

    for (
        index,
        m,
        n,
        t,
        configured_trials,
    ) in selected_sets:

        trials = (
            1
            if args.quick
            else configured_trials
        )

        row = (
            profile_parameter_set(
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

        gc.collect()

    print_final_summary(
        rows,
        csv_path,
    )


if __name__ == "__main__":
    main()
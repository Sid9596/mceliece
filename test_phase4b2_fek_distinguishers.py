#!/usr/bin/env python3
"""
Phase 4B2 / 4B2.1
=================

Structural distinguishers and explicit public syndrome-table denoising
experiments for the compressed fused evaluation key.

The public fused rows have the form

    Y_{u,v} = K_{u,v} + z_{u,v},

where

    K_{u,v}
        =
    (Lambda_u odot Lambda_v) G_pub

and

    wt(z_{u,v}) = tau_F.

The script performs, across independently generated McEliece / HE keys:

    1. rank measurements of sampled fused rows;
    2. sampled Schur-product rank measurements;
    3. public-syndrome measurements;
    4. syndrome duplicate / collision measurements;
    5. augmented-code measurements for <G_pub, Y>;
    6. explicit public bounded-weight syndrome-table denoising.

The public denoising attack uses

    V(n,tau_F)
        =
    sum_{w=0}^{tau_F} binom(n,w)

candidate error vectors.

If this bounded-weight search space exceeds --max-enumeration, the
denoising experiment is skipped for that key.  Structural measurements
still run and the CSV row is still written.

Skipped attack measurements are stored as NaN, not zero.  Therefore a
skipped experiment cannot be confused with an attempted attack having
zero recovery rate.

This is a finite experimental structural-security diagnostic.  It does
not by itself establish asymptotic secret-key recovery.
"""

import argparse
import csv
import math
import os
import time
from collections import Counter
from itertools import combinations

import numpy as np

from mceliece.mceliececipher import McElieceCipher

from mceliece.fused_he import (
    FusedMcElieceHE,
    fused_pair_count,
    gf2_matmul,
    gf2_uint8,
    truncated_product,
)

from mceliece.mathutils import GF2Matrix


# ============================================================================
# Formatting helpers
# ============================================================================

LINE = "=" * 78
SUBLINE = "-" * 78


def format_seconds(value):
    value = float(value)

    if value < 60.0:
        return f"{value:.6f} s"

    minutes = value / 60.0

    if minutes < 60.0:
        return f"{minutes:.3f} min"

    hours = minutes / 60.0

    if hours < 24.0:
        return f"{hours:.3f} h"

    return f"{hours / 24.0:.3f} days"


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def finite_values(values):
    result = []

    for value in values:
        try:
            value = float(value)
        except Exception:
            continue

        if math.isfinite(value):
            result.append(value)

    return result


def mean_finite(values):
    values = finite_values(values)

    if not values:
        return float("nan")

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def std_finite(values):
    values = finite_values(values)

    if len(values) < 2:
        return 0.0

    return float(
        np.std(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            ddof=1,
        )
    )


def cohen_d_absolute(
    real_values,
    random_values,
):
    """
    Absolute two-sample Cohen d.

    Used only as a finite experimental effect-size summary.
    """

    x = np.asarray(
        finite_values(
            real_values
        ),
        dtype=np.float64,
    )

    y = np.asarray(
        finite_values(
            random_values
        ),
        dtype=np.float64,
    )

    if (
        x.size == 0
        or
        y.size == 0
    ):
        return float("nan")

    mean_difference = abs(
        float(
            np.mean(x)
            -
            np.mean(y)
        )
    )

    if (
        x.size < 2
        or
        y.size < 2
    ):
        return (
            0.0
            if mean_difference == 0.0
            else float("inf")
        )

    var_x = float(
        np.var(
            x,
            ddof=1,
        )
    )

    var_y = float(
        np.var(
            y,
            ddof=1,
        )
    )

    denominator = (
        x.size
        +
        y.size
        -
        2
    )

    if denominator <= 0:
        return float("nan")

    pooled_variance = (
        (
            (x.size - 1)
            *
            var_x
        )
        +
        (
            (y.size - 1)
            *
            var_y
        )
    ) / denominator

    if pooled_variance <= 0.0:
        return (
            0.0
            if mean_difference == 0.0
            else float("inf")
        )

    return float(
        mean_difference
        /
        math.sqrt(
            pooled_variance
        )
    )


# ============================================================================
# GF(2) helpers
# ============================================================================

def gf2_rank(matrix):
    """
    Exact rank over GF(2), using the existing FLINT-backed GF2Matrix class.
    """

    arr = np.asarray(
        matrix,
        dtype=np.uint8,
    )

    if arr.ndim != 2:
        raise ValueError(
            "gf2_rank expects a two-dimensional matrix."
        )

    if (
        arr.shape[0] == 0
        or
        arr.shape[1] == 0
    ):
        return 0

    arr = (
        arr
        &
        np.uint8(1)
    )

    gf2_matrix = (
        GF2Matrix.from_list(
            arr
        )
    )

    _, rank = (
        gf2_matrix.rref()
    )

    return int(rank)


def independent_row_basis(
    matrix,
):
    """
    Return a row-basis array over GF(2).

    This helper performs simple binary Gaussian elimination and is used
    only to reduce redundant rows before some Schur measurements.
    """

    A = np.asarray(
        matrix,
        dtype=np.uint8,
    ).copy()

    A &= np.uint8(1)

    if A.ndim != 2:
        raise ValueError(
            "independent_row_basis expects a matrix."
        )

    rows, cols = (
        A.shape
    )

    pivot_row = 0
    pivot_rows = []

    for col in range(
        cols
    ):
        candidates = (
            np.flatnonzero(
                A[
                    pivot_row:,
                    col,
                ]
            )
        )

        if candidates.size == 0:
            continue

        selected = (
            pivot_row
            +
            int(
                candidates[0]
            )
        )

        if (
            selected
            !=
            pivot_row
        ):
            A[
                [pivot_row, selected]
            ] = A[
                [selected, pivot_row]
            ]

        for row in range(
            rows
        ):
            if (
                row
                !=
                pivot_row
                and
                A[
                    row,
                    col,
                ]
            ):
                A[
                    row
                ] ^= A[
                    pivot_row
                ]

        pivot_rows.append(
            pivot_row
        )

        pivot_row += 1

        if pivot_row >= rows:
            break

    if not pivot_rows:
        return np.zeros(
            (
                0,
                cols,
            ),
            dtype=np.uint8,
        )

    return (
        A[
            pivot_rows
        ]
        .copy()
    )


def sampled_schur_rank(
    matrix,
    pair_limit,
    rng,
):
    """
    Sample coordinate-wise products of rows and return their GF(2) rank.

    The experiment is intentionally bounded by pair_limit so that the
    structural test remains practical for larger sampled objects.
    """

    matrix = np.asarray(
        matrix,
        dtype=np.uint8,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "sampled_schur_rank expects a matrix."
        )

    row_count = int(
        matrix.shape[0]
    )

    column_count = int(
        matrix.shape[1]
    )

    if (
        row_count == 0
        or
        column_count == 0
        or
        pair_limit <= 0
    ):
        return 0

    # Keep the original sampled-row distribution.
    total_pairs = (
        row_count
        *
        (
            row_count
            +
            1
        )
        //
        2
    )

    target = min(
        int(pair_limit),
        int(total_pairs),
    )

    products = np.empty(
        (
            target,
            column_count,
        ),
        dtype=np.uint8,
    )

    if target == total_pairs:

        write_index = 0

        for i in range(
            row_count
        ):
            for j in range(
                i,
                row_count,
            ):
                products[
                    write_index
                ] = (
                    matrix[i]
                    &
                    matrix[j]
                )

                write_index += 1

    else:

        seen = set()

        write_index = 0

        while (
            write_index
            <
            target
        ):
            i = int(
                rng.integers(
                    0,
                    row_count,
                )
            )

            j = int(
                rng.integers(
                    0,
                    row_count,
                )
            )

            if i > j:
                i, j = j, i

            key = (
                i,
                j,
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            products[
                write_index
            ] = (
                matrix[i]
                &
                matrix[j]
            )

            write_index += 1

    return gf2_rank(
        products
    )


def row_key(
    row,
):
    """
    Compact immutable representation of a binary vector.
    """

    row = np.asarray(
        row,
        dtype=np.uint8,
    ).reshape(-1)

    return (
        np.packbits(
            row,
            bitorder="little",
        )
        .tobytes()
    )


def duplicate_statistics(
    matrix,
):
    """
    Return:

        duplicate_rows
            =
        sum multiplicity-1

    and

        collision_pairs
            =
        sum binom(multiplicity,2).
    """

    counts = Counter(
        row_key(row)
        for row
        in np.asarray(
            matrix,
            dtype=np.uint8,
        )
    )

    duplicate_rows = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    collision_pairs = sum(
        (
            count
            *
            (
                count - 1
            )
            //
            2
        )
        for count in counts.values()
        if count > 1
    )

    return (
        int(
            duplicate_rows
        ),
        int(
            collision_pairs
        ),
    )


# ============================================================================
# Pair indexing
# ============================================================================

def symmetric_pair_from_index(
    index,
    n,
):
    """
    Invert the compressed symmetric pair ordering

        (0,0),...,(0,n-1),
        (1,1),...,(1,n-1),
        ...

    without allocating the full pair list.
    """

    index = int(index)
    n = int(n)

    pair_count = fused_pair_count(
        n
    )

    if not (
        0
        <=
        index
        <
        pair_count
    ):
        raise ValueError(
            "Invalid fused pair index."
        )

    low = 0
    high = n - 1

    while low <= high:

        mid = (
            low
            +
            high
        ) // 2

        prefix = (
            mid
            *
            n
            -
            (
                mid
                *
                (
                    mid - 1
                )
                //
                2
            )
        )

        next_prefix = (
            (
                mid + 1
            )
            *
            n
            -
            (
                (
                    mid + 1
                )
                *
                mid
                //
                2
            )
        )

        if (
            prefix
            <=
            index
            <
            next_prefix
        ):
            u = mid
            v = (
                u
                +
                (
                    index
                    -
                    prefix
                )
            )

            return (
                int(u),
                int(v),
            )

        if index < prefix:
            high = (
                mid - 1
            )

        else:
            low = (
                mid + 1
            )

    raise RuntimeError(
        "Unable to invert fused pair index."
    )


def sample_symmetric_pairs(
    n,
    row_count,
    rng,
):
    pair_count = fused_pair_count(
        n
    )

    row_count = min(
        int(row_count),
        int(pair_count),
    )

    indices = (
        rng.choice(
            pair_count,
            size=row_count,
            replace=False,
        )
    )

    return [
        symmetric_pair_from_index(
            int(index),
            n,
        )
        for index in indices
    ]


# ============================================================================
# Reference row generation
# ============================================================================

def random_exact_weight_vector(
    n,
    weight,
    rng,
):
    weight = int(weight)

    if not (
        0
        <=
        weight
        <=
        n
    ):
        raise ValueError(
            "Invalid Hamming weight."
        )

    result = np.zeros(
        n,
        dtype=np.uint8,
    )

    if weight == 0:
        return result

    support = (
        rng.choice(
            n,
            size=weight,
            replace=False,
        )
    )

    result[
        support
    ] = 1

    return result


def bernoulli_reference(
    rows,
    n,
    rng,
):
    return rng.integers(
        0,
        2,
        size=(
            int(rows),
            int(n),
        ),
        dtype=np.uint8,
    )


def weight_matched_reference(
    source,
    rng,
):
    source = np.asarray(
        source,
        dtype=np.uint8,
    )

    result = np.zeros_like(
        source,
        dtype=np.uint8,
    )

    for index in range(
        source.shape[0]
    ):

        weight = int(
            np.count_nonzero(
                source[
                    index
                ]
            )
        )

        if weight == 0:
            continue

        support = (
            rng.choice(
                source.shape[1],
                size=weight,
                replace=False,
            )
        )

        result[
            index,
            support,
        ] = 1

    return result


# ============================================================================
# Public parity check
# ============================================================================

def public_parity_check(
    mc,
):
    """
    Construct a parity-check matrix for the public permuted code.

    Current repository convention:

        G_pub = (S G)[:, P].

    Therefore

        H_pub = H[:, P]

    satisfies

        G_pub H_pub^T = 0.
    """

    H = gf2_uint8(
        mc.H
    )

    P = np.asarray(
        mc.P,
        dtype=np.int64,
    ).reshape(-1)

    H_pub = (
        H[
            :,
            P,
        ]
        .copy()
    )

    G_pub = gf2_uint8(
        mc.Gp
    )

    check = gf2_matmul(
        G_pub,
        H_pub.T,
    )

    if np.any(
        check
    ):
        raise RuntimeError(
            "Public parity-check construction failed: "
            "G_pub H_pub^T != 0."
        )

    return H_pub


# ============================================================================
# Sample public fused rows
# ============================================================================

def sample_fused_rows(
    he,
    row_count,
    tau_F,
    rng,
):
    """
    Generate a controlled sample

        Y = K + Z

    without materializing the complete FEK.
    """

    pairs = sample_symmetric_pairs(
        he.n,
        row_count,
        rng,
    )

    sample_count = len(
        pairs
    )

    K = np.empty(
        (
            sample_count,
            he.n,
        ),
        dtype=np.uint8,
    )

    Z = np.zeros(
        (
            sample_count,
            he.n,
        ),
        dtype=np.uint8,
    )

    G_pub = gf2_uint8(
        he.mc.Gp
    )

    start = time.perf_counter()

    for row_index, (
        u,
        v,
    ) in enumerate(
        pairs
    ):

        product = (
            truncated_product(
                he.Lambda[u],
                he.Lambda[v],
                output_length=he.k,
            )
        )

        K[
            row_index
        ] = gf2_matmul(
            product,
            G_pub,
        )

        if tau_F > 0:

            support = (
                rng.choice(
                    he.n,
                    size=tau_F,
                    replace=False,
                )
            )

            Z[
                row_index,
                support,
            ] = 1

    Y = (
        K
        ^
        Z
    )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    return (
        pairs,
        K,
        Z,
        Y,
        elapsed,
    )


# ============================================================================
# Syndrome statistics
# ============================================================================

def syndrome_matrix(
    rows,
    H_pub,
):
    return gf2_matmul(
        np.asarray(
            rows,
            dtype=np.uint8,
        ),
        np.asarray(
            H_pub,
            dtype=np.uint8,
        ).T,
    )


def syndrome_statistics(
    rows,
    H_pub,
    schur_pairs,
    rng,
):
    syndromes = syndrome_matrix(
        rows,
        H_pub,
    )

    weights = np.sum(
        syndromes,
        axis=1,
        dtype=np.int64,
    )

    duplicate_rows, collision_pairs = (
        duplicate_statistics(
            syndromes
        )
    )

    return {
        "matrix": (
            syndromes
        ),
        "rank": gf2_rank(
            syndromes
        ),
        "mean_weight": float(
            np.mean(
                weights
            )
        ),
        "duplicate_rows": int(
            duplicate_rows
        ),
        "collision_pairs": int(
            collision_pairs
        ),
        "schur_rank": sampled_schur_rank(
            syndromes,
            schur_pairs,
            rng,
        ),
    }


# ============================================================================
# Bounded-weight enumeration
# ============================================================================

def bounded_weight_search_space(
    n,
    tau_F,
):
    """
    V(n,tau_F) = sum_{w=0}^{tau_F} binom(n,w).
    """

    n = int(n)
    tau_F = int(tau_F)

    if tau_F < 0:
        raise ValueError(
            "tau_F must be nonnegative."
        )

    if tau_F > n:
        raise ValueError(
            "tau_F cannot exceed n."
        )

    return int(
        sum(
            math.comb(
                n,
                weight,
            )
            for weight
            in range(
                tau_F + 1
            )
        )
    )


def syndrome_key_from_support(
    H_pub,
    support,
):
    """
    Syndrome of the binary error whose support is `support`.

    H_pub has shape (n-k,n), so the syndrome is the XOR of the
    corresponding columns.
    """

    parity_rows = int(
        H_pub.shape[0]
    )

    syndrome = np.zeros(
        parity_rows,
        dtype=np.uint8,
    )

    for position in support:
        syndrome ^= (
            H_pub[
                :,
                int(position),
            ]
        )

    return (
        np.packbits(
            syndrome,
            bitorder="little",
        )
        .tobytes()
    )


def build_bounded_weight_syndrome_table(
    H_pub,
    tau_F,
    max_enumeration,
):
    """
    Enumerate every binary error e satisfying

        wt(e) <= tau_F.

    Return

        table,
        ambiguous_syndromes,
        enumerated_words,
        elapsed_seconds.

    table[syndrome] is:

        tuple(support)
            if a unique bounded-weight representative is known;

        None
            if two distinct bounded-weight representatives have the
            same syndrome.

    The function checks the enumeration guard itself as a defensive
    measure, although callers normally check it first.
    """

    H_pub = np.asarray(
        H_pub,
        dtype=np.uint8,
    )

    tau_F = int(
        tau_F
    )

    n = int(
        H_pub.shape[1]
    )

    search_space = (
        bounded_weight_search_space(
            n,
            tau_F,
        )
    )

    if (
        search_space
        >
        int(
            max_enumeration
        )
    ):
        raise RuntimeError(
            "Bounded-weight search requires "
            f"{search_space:,} words; "
            f"guard={int(max_enumeration):,}."
        )

    table = {}

    ambiguous_keys = set()

    enumerated_words = 0

    start = time.perf_counter()

    for weight in range(
        tau_F + 1
    ):

        for support in combinations(
            range(n),
            weight,
        ):

            enumerated_words += 1

            key = (
                syndrome_key_from_support(
                    H_pub,
                    support,
                )
            )

            if key not in table:

                table[
                    key
                ] = tuple(
                    int(position)
                    for position
                    in support
                )

                continue

            previous = (
                table[
                    key
                ]
            )

            if previous is None:
                continue

            if (
                previous
                !=
                tuple(
                    support
                )
            ):

                table[
                    key
                ] = None

                ambiguous_keys.add(
                    key
                )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    return (
        table,
        int(
            len(
                ambiguous_keys
            )
        ),
        int(
            enumerated_words
        ),
        float(
            elapsed
        ),
    )


def support_to_vector(
    n,
    support,
):
    result = np.zeros(
        int(n),
        dtype=np.uint8,
    )

    if support:
        result[
            np.asarray(
                support,
                dtype=np.int64,
            )
        ] = 1

    return result


# ============================================================================
# Explicit public denoising attack
# ============================================================================

def empty_denoising_result(
    n,
    tau_F,
):
    search_space = (
        bounded_weight_search_space(
            n,
            tau_F,
        )
    )

    return {
        "attempted": False,
        "skipped": False,
        "skip_reason": "",
        "search_space": int(
            search_space
        ),
        "log2_search_space": (
            float(
                math.log2(
                    search_space
                )
            )
            if search_space > 0
            else 0.0
        ),
        "table_size": float("nan"),
        "ambiguous_syndromes": float("nan"),
        "enumerated_words": float("nan"),
        "sample_count": float("nan"),
        "recovered": float("nan"),
        "exact_noise": float("nan"),
        "exact_K": float("nan"),
        "ambiguous": float("nan"),
        "missing": float("nan"),
        "false_recoveries": float("nan"),
        "recovery_fraction": float("nan"),
        "exact_noise_fraction": float("nan"),
        "exact_K_fraction": float("nan"),
        "ambiguity_fraction": float("nan"),
        "missing_fraction": float("nan"),
        "false_recovery_fraction": float("nan"),
        "table_build_seconds": float("nan"),
        "lookup_seconds": float("nan"),
    }


def run_public_denoising_attack(
    Y_real,
    K_real,
    Z_real,
    H_pub,
    tau_F,
    max_enumeration,
):
    """
    Explicitly decode each public fused-row syndrome using all errors with

        wt(e) <= tau_F.

    If V(n,tau_F) exceeds max_enumeration, return a structurally complete
    result dictionary with NaN attack measurements.
    """

    Y_real = np.asarray(
        Y_real,
        dtype=np.uint8,
    )

    K_real = np.asarray(
        K_real,
        dtype=np.uint8,
    )

    Z_real = np.asarray(
        Z_real,
        dtype=np.uint8,
    )

    H_pub = np.asarray(
        H_pub,
        dtype=np.uint8,
    )

    n = int(
        Y_real.shape[1]
    )

    result = (
        empty_denoising_result(
            n,
            tau_F,
        )
    )

    search_space = int(
        result[
            "search_space"
        ]
    )

    log2_search_space = float(
        result[
            "log2_search_space"
        ]
    )

    print()
    print(
        "Public syndrome-table denoising attack"
    )
    print(
        SUBLINE
    )
    print(
        "Bounded search: "
        f"V({n},{tau_F}) = "
        f"{search_space:,} "
        f"(2^{log2_search_space:.3f})"
    )

    if (
        search_space
        >
        int(
            max_enumeration
        )
    ):

        result[
            "skipped"
        ] = True

        result[
            "skip_reason"
        ] = (
            "bounded-weight search space exceeds "
            "the configured enumeration guard"
        )

        print(
            "Enumeration  : SKIPPED"
        )

        print(
            "Guard        : "
            f"{int(max_enumeration):,}"
        )

        print(
            "Reason       : bounded-weight search space "
            "exceeds enumeration guard"
        )

        return result

    result[
        "attempted"
    ] = True

    (
        syndrome_table,
        ambiguous_syndromes,
        enumerated_words,
        table_build_seconds,
    ) = (
        build_bounded_weight_syndrome_table(
            H_pub=H_pub,
            tau_F=tau_F,
            max_enumeration=max_enumeration,
        )
    )

    recovered = 0
    exact_noise = 0
    exact_K = 0
    ambiguous = 0
    missing = 0
    false_recoveries = 0

    lookup_start = (
        time.perf_counter()
    )

    for row_index in range(
        Y_real.shape[0]
    ):

        Y_row = (
            Y_real[
                row_index
            ]
        )

        syndrome = (
            gf2_matmul(
                Y_row,
                H_pub.T,
            )
        )

        key = row_key(
            syndrome
        )

        if key not in syndrome_table:

            missing += 1
            continue

        candidate_support = (
            syndrome_table[
                key
            ]
        )

        if candidate_support is None:

            ambiguous += 1
            continue

        candidate = (
            support_to_vector(
                n,
                candidate_support,
            )
        )

        recovered += 1

        true_noise = (
            Z_real[
                row_index
            ]
        )

        true_K = (
            K_real[
                row_index
            ]
        )

        recovered_K = (
            Y_row
            ^
            candidate
        )

        if np.array_equal(
            candidate,
            true_noise,
        ):

            exact_noise += 1

        else:

            false_recoveries += 1

        if np.array_equal(
            recovered_K,
            true_K,
        ):

            exact_K += 1

    lookup_seconds = (
        time.perf_counter()
        -
        lookup_start
    )

    sample_count = int(
        Y_real.shape[0]
    )

    result.update(
        {
            "table_size": int(
                len(
                    syndrome_table
                )
            ),
            "ambiguous_syndromes": int(
                ambiguous_syndromes
            ),
            "enumerated_words": int(
                enumerated_words
            ),
            "sample_count": int(
                sample_count
            ),
            "recovered": int(
                recovered
            ),
            "exact_noise": int(
                exact_noise
            ),
            "exact_K": int(
                exact_K
            ),
            "ambiguous": int(
                ambiguous
            ),
            "missing": int(
                missing
            ),
            "false_recoveries": int(
                false_recoveries
            ),
            "recovery_fraction": (
                recovered
                /
                float(
                    sample_count
                )
            ),
            "exact_noise_fraction": (
                exact_noise
                /
                float(
                    sample_count
                )
            ),
            "exact_K_fraction": (
                exact_K
                /
                float(
                    sample_count
                )
            ),
            "ambiguity_fraction": (
                ambiguous
                /
                float(
                    sample_count
                )
            ),
            "missing_fraction": (
                missing
                /
                float(
                    sample_count
                )
            ),
            "false_recovery_fraction": (
                false_recoveries
                /
                float(
                    sample_count
                )
            ),
            "table_build_seconds": float(
                table_build_seconds
            ),
            "lookup_seconds": float(
                lookup_seconds
            ),
        }
    )

    print(
        "Table        : "
        f"{len(syndrome_table):,} syndrome entries, "
        f"{ambiguous_syndromes:,} ambiguous"
    )

    print(
        "Enumerated   : "
        f"{enumerated_words:,}"
    )

    print(
        "Recovered    : "
        f"{recovered}/{sample_count} "
        f"({100.0 * recovered / sample_count:.2f}%)"
    )

    print(
        "Exact noise  : "
        f"{exact_noise}/{sample_count} "
        f"({100.0 * exact_noise / sample_count:.2f}%)"
    )

    print(
        "Exact K      : "
        f"{exact_K}/{sample_count} "
        f"({100.0 * exact_K / sample_count:.2f}%)"
    )

    print(
        f"Ambiguous    : {ambiguous}"
    )

    print(
        f"Missing      : {missing}"
    )

    print(
        "False recover: "
        f"{false_recoveries}"
    )

    print(
        "Table build  : "
        f"{table_build_seconds:.6f} s"
    )

    print(
        "Lookups      : "
        f"{lookup_seconds:.6f} s"
    )

    return result


# ============================================================================
# One independent-key experiment
# ============================================================================

def run_instance(
    instance_index,
    args,
):
    seed = int(
        args.seed
        +
        1000003
        *
        instance_index
    )

    rng = np.random.default_rng(
        seed
        +
        700001
    )

    np.random.seed(
        seed
    )

    # ------------------------------------------------------------------
    # McEliece key generation
    # ------------------------------------------------------------------

    keygen_start = (
        time.perf_counter()
    )

    mc = McElieceCipher(
        args.m,
        args.n,
        args.t,
    )

    mc.generate_random_keys()

    keygen_seconds = (
        time.perf_counter()
        -
        keygen_start
    )

    # ------------------------------------------------------------------
    # HE transport
    # ------------------------------------------------------------------

    he = FusedMcElieceHE(
        mc,
        ell=args.ell,
        tau_enc=0,
        tau_F=args.tau_f,
        seed=(
            seed
            +
            10000
        ),
    )

    transport_start = (
        time.perf_counter()
    )

    he.construct_transport(
        randomization_trials=64,
        store_right_inverse=False,
    )

    transport_seconds = (
        time.perf_counter()
        -
        transport_start
    )

    # ------------------------------------------------------------------
    # Public code objects
    # ------------------------------------------------------------------

    G_pub = gf2_uint8(
        mc.Gp
    )

    H_pub = public_parity_check(
        mc
    )

    # ------------------------------------------------------------------
    # Sample real fused rows
    # ------------------------------------------------------------------

    (
        pairs,
        K_real,
        Z_real,
        Y_real,
        sample_fek_seconds,
    ) = sample_fused_rows(
        he=he,
        row_count=args.rows,
        tau_F=args.tau_f,
        rng=rng,
    )

    sampled_rows = int(
        Y_real.shape[0]
    )

    total_fek_rows = fused_pair_count(
        args.n
    )

    # ------------------------------------------------------------------
    # Reference distributions
    # ------------------------------------------------------------------

    Y_bernoulli = (
        bernoulli_reference(
            sampled_rows,
            args.n,
            rng,
        )
    )

    Y_weight = (
        weight_matched_reference(
            Y_real,
            rng,
        )
    )

    # ------------------------------------------------------------------
    # Direct Y statistics
    # ------------------------------------------------------------------

    Y_rank_real = gf2_rank(
        Y_real
    )

    Y_rank_bernoulli = gf2_rank(
        Y_bernoulli
    )

    Y_rank_weight = gf2_rank(
        Y_weight
    )

    Y_schur_real = (
        sampled_schur_rank(
            Y_real,
            args.schur_pairs,
            rng,
        )
    )

    Y_schur_bernoulli = (
        sampled_schur_rank(
            Y_bernoulli,
            args.schur_pairs,
            rng,
        )
    )

    Y_schur_weight = (
        sampled_schur_rank(
            Y_weight,
            args.schur_pairs,
            rng,
        )
    )

    # ------------------------------------------------------------------
    # Public syndrome statistics
    # ------------------------------------------------------------------

    syndrome_real = syndrome_statistics(
        Y_real,
        H_pub,
        args.schur_pairs,
        rng,
    )

    syndrome_bernoulli = (
        syndrome_statistics(
            Y_bernoulli,
            H_pub,
            args.schur_pairs,
            rng,
        )
    )

    syndrome_weight = (
        syndrome_statistics(
            Y_weight,
            H_pub,
            args.schur_pairs,
            rng,
        )
    )

    # ------------------------------------------------------------------
    # Augmented code <G_pub, Y>
    # ------------------------------------------------------------------

    augmented_real = np.vstack(
        (
            G_pub,
            Y_real,
        )
    )

    augmented_bernoulli = np.vstack(
        (
            G_pub,
            Y_bernoulli,
        )
    )

    augmented_weight = np.vstack(
        (
            G_pub,
            Y_weight,
        )
    )

    augmented_rank_real = (
        gf2_rank(
            augmented_real
        )
    )

    augmented_rank_bernoulli = (
        gf2_rank(
            augmented_bernoulli
        )
    )

    augmented_rank_weight = (
        gf2_rank(
            augmented_weight
        )
    )

    augmented_schur_real = (
        sampled_schur_rank(
            augmented_real,
            args.schur_pairs,
            rng,
        )
    )

    augmented_schur_bernoulli = (
        sampled_schur_rank(
            augmented_bernoulli,
            args.schur_pairs,
            rng,
        )
    )

    augmented_schur_weight = (
        sampled_schur_rank(
            augmented_weight,
            args.schur_pairs,
            rng,
        )
    )

    # ------------------------------------------------------------------
    # Print structural report
    # ------------------------------------------------------------------

    print()
    print(
        SUBLINE
    )

    print(
        f"Instance {instance_index}"
    )

    print(
        "Parameters : "
        f"(m,n,t,k,ell)="
        f"({args.m},{args.n},{args.t},{mc.k},{args.ell})"
    )

    print(
        f"tau_F      : {args.tau_f}"
    )

    print(
        "FEK rows   : "
        f"{sampled_rows:,} / "
        f"{total_fek_rows:,}"
    )

    print()

    print(
        "Y rank     : "
        f"real={Y_rank_real}, "
        f"Bernoulli={Y_rank_bernoulli}, "
        f"weight-matched={Y_rank_weight}"
    )

    print(
        "Y Schur    : "
        f"real={Y_schur_real}, "
        f"Bernoulli={Y_schur_bernoulli}, "
        f"weight-matched={Y_schur_weight}"
    )

    print()

    print(
        "Synd rank  : "
        f"real={syndrome_real['rank']}, "
        f"Bernoulli={syndrome_bernoulli['rank']}, "
        f"weight-matched={syndrome_weight['rank']}"
    )

    print(
        "Synd wt    : "
        f"real={syndrome_real['mean_weight']:.3f}, "
        f"Bernoulli={syndrome_bernoulli['mean_weight']:.3f}"
    )

    print(
        "Synd dup   : "
        f"real={syndrome_real['duplicate_rows']}, "
        f"Bernoulli={syndrome_bernoulli['duplicate_rows']}, "
        f"weight-matched={syndrome_weight['duplicate_rows']}"
    )

    print(
        "Synd coll  : "
        f"real={syndrome_real['collision_pairs']}, "
        f"Bernoulli={syndrome_bernoulli['collision_pairs']}, "
        f"weight-matched={syndrome_weight['collision_pairs']}"
    )

    print()

    print(
        "C+ rank    : "
        f"real={augmented_rank_real}, "
        f"Bernoulli={augmented_rank_bernoulli}, "
        f"weight-matched={augmented_rank_weight}"
    )

    print(
        "C+ Schur   : "
        f"real={augmented_schur_real}, "
        f"Bernoulli={augmented_schur_bernoulli}, "
        f"weight-matched={augmented_schur_weight}"
    )

    # ------------------------------------------------------------------
    # Explicit public denoising
    # ------------------------------------------------------------------

    denoising = (
        run_public_denoising_attack(
            Y_real=Y_real,
            K_real=K_real,
            Z_real=Z_real,
            H_pub=H_pub,
            tau_F=args.tau_f,
            max_enumeration=args.max_enumeration,
        )
    )

    print()

    print(
        "Timings     : "
        f"keygen={keygen_seconds:.6f} s, "
        f"transport={transport_seconds:.6f} s, "
        f"sample FEK={sample_fek_seconds:.6f} s"
    )

    # ------------------------------------------------------------------
    # Flat CSV row
    # ------------------------------------------------------------------

    row = {
        "instance": int(
            instance_index
        ),
        "seed": int(
            seed
        ),
        "m": int(
            args.m
        ),
        "n": int(
            args.n
        ),
        "t": int(
            args.t
        ),
        "k": int(
            mc.k
        ),
        "ell": int(
            args.ell
        ),
        "tau_F": int(
            args.tau_f
        ),
        "sampled_fek_rows": int(
            sampled_rows
        ),
        "total_fek_rows": int(
            total_fek_rows
        ),

        # Y statistics
        "Y_rank_real": int(
            Y_rank_real
        ),
        "Y_rank_bernoulli": int(
            Y_rank_bernoulli
        ),
        "Y_rank_weight_matched": int(
            Y_rank_weight
        ),
        "Y_rank_deficiency_real": int(
            args.n
            -
            Y_rank_real
        ),
        "Y_rank_deficiency_bernoulli": int(
            args.n
            -
            Y_rank_bernoulli
        ),
        "Y_rank_deficiency_weight_matched": int(
            args.n
            -
            Y_rank_weight
        ),
        "Y_schur_rank_real": int(
            Y_schur_real
        ),
        "Y_schur_rank_bernoulli": int(
            Y_schur_bernoulli
        ),
        "Y_schur_rank_weight_matched": int(
            Y_schur_weight
        ),

        # Syndrome statistics
        "syndrome_rank_real": int(
            syndrome_real[
                "rank"
            ]
        ),
        "syndrome_rank_bernoulli": int(
            syndrome_bernoulli[
                "rank"
            ]
        ),
        "syndrome_rank_weight_matched": int(
            syndrome_weight[
                "rank"
            ]
        ),
        "syndrome_mean_weight_real": float(
            syndrome_real[
                "mean_weight"
            ]
        ),
        "syndrome_mean_weight_bernoulli": float(
            syndrome_bernoulli[
                "mean_weight"
            ]
        ),
        "syndrome_mean_weight_weight_matched": float(
            syndrome_weight[
                "mean_weight"
            ]
        ),
        "syndrome_duplicate_rows_real": int(
            syndrome_real[
                "duplicate_rows"
            ]
        ),
        "syndrome_duplicate_rows_bernoulli": int(
            syndrome_bernoulli[
                "duplicate_rows"
            ]
        ),
        "syndrome_duplicate_rows_weight_matched": int(
            syndrome_weight[
                "duplicate_rows"
            ]
        ),
        "syndrome_collision_pairs_real": int(
            syndrome_real[
                "collision_pairs"
            ]
        ),
        "syndrome_collision_pairs_bernoulli": int(
            syndrome_bernoulli[
                "collision_pairs"
            ]
        ),
        "syndrome_collision_pairs_weight_matched": int(
            syndrome_weight[
                "collision_pairs"
            ]
        ),
        "syndrome_schur_rank_real": int(
            syndrome_real[
                "schur_rank"
            ]
        ),
        "syndrome_schur_rank_bernoulli": int(
            syndrome_bernoulli[
                "schur_rank"
            ]
        ),
        "syndrome_schur_rank_weight_matched": int(
            syndrome_weight[
                "schur_rank"
            ]
        ),

        # Augmented code
        "augmented_rank_real": int(
            augmented_rank_real
        ),
        "augmented_rank_bernoulli": int(
            augmented_rank_bernoulli
        ),
        "augmented_rank_weight_matched": int(
            augmented_rank_weight
        ),
        "augmented_schur_rank_real": int(
            augmented_schur_real
        ),
        "augmented_schur_rank_bernoulli": int(
            augmented_schur_bernoulli
        ),
        "augmented_schur_rank_weight_matched": int(
            augmented_schur_weight
        ),

        # Denoising status
        "denoise_attempted": int(
            denoising[
                "attempted"
            ]
        ),
        "denoise_skipped": int(
            denoising[
                "skipped"
            ]
        ),
        "denoise_skip_reason": (
            denoising[
                "skip_reason"
            ]
        ),
        "denoise_search_space": int(
            denoising[
                "search_space"
            ]
        ),
        "denoise_log2_search_space": float(
            denoising[
                "log2_search_space"
            ]
        ),
        "denoise_table_size": (
            denoising[
                "table_size"
            ]
        ),
        "denoise_ambiguous_syndromes": (
            denoising[
                "ambiguous_syndromes"
            ]
        ),
        "denoise_enumerated_words": (
            denoising[
                "enumerated_words"
            ]
        ),
        "denoise_sample_count": (
            denoising[
                "sample_count"
            ]
        ),
        "denoise_recovered": (
            denoising[
                "recovered"
            ]
        ),
        "denoise_exact_noise": (
            denoising[
                "exact_noise"
            ]
        ),
        "denoise_exact_K": (
            denoising[
                "exact_K"
            ]
        ),
        "denoise_ambiguous": (
            denoising[
                "ambiguous"
            ]
        ),
        "denoise_missing": (
            denoising[
                "missing"
            ]
        ),
        "denoise_false_recoveries": (
            denoising[
                "false_recoveries"
            ]
        ),
        "denoise_recovery_fraction": (
            denoising[
                "recovery_fraction"
            ]
        ),
        "denoise_exact_noise_fraction": (
            denoising[
                "exact_noise_fraction"
            ]
        ),
        "denoise_exact_K_fraction": (
            denoising[
                "exact_K_fraction"
            ]
        ),
        "denoise_ambiguity_fraction": (
            denoising[
                "ambiguity_fraction"
            ]
        ),
        "denoise_missing_fraction": (
            denoising[
                "missing_fraction"
            ]
        ),
        "denoise_false_recovery_fraction": (
            denoising[
                "false_recovery_fraction"
            ]
        ),
        "denoise_table_build_seconds": (
            denoising[
                "table_build_seconds"
            ]
        ),
        "denoise_lookup_seconds": (
            denoising[
                "lookup_seconds"
            ]
        ),

        # Timing
        "keygen_seconds": float(
            keygen_seconds
        ),
        "transport_seconds": float(
            transport_seconds
        ),
        "sample_fek_seconds": float(
            sample_fek_seconds
        ),
    }

    return row


# ============================================================================
# Robust CSV handling
# ============================================================================

def append_csv_row(
    filename,
    row,
):
    """
    Append one result row.

    If an older CSV exists with fewer columns, preserve its old rows and
    rewrite it once using the union of old and new field names.

    This prevents the new denoise_skipped / skip_reason fields from
    breaking continuation of an older Phase-4B2 CSV.
    """

    filename = os.fspath(
        filename
    )

    if not os.path.exists(
        filename
    ):

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    row.keys()
                ),
            )

            writer.writeheader()

            writer.writerow(
                row
            )

        return

    # Read existing header and rows.
    with open(
        filename,
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        old_fieldnames = (
            list(
                reader.fieldnames
                or
                []
            )
        )

        old_rows = list(
            reader
        )

    new_fieldnames = list(
        old_fieldnames
    )

    for field in row.keys():

        if field not in new_fieldnames:

            new_fieldnames.append(
                field
            )

    header_changed = (
        new_fieldnames
        !=
        old_fieldnames
    )

    if header_changed:

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    new_fieldnames
                ),
            )

            writer.writeheader()

            for old_row in old_rows:

                writer.writerow(
                    old_row
                )

            writer.writerow(
                row
            )

        return

    with open(
        filename,
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=(
                new_fieldnames
            ),
        )

        writer.writerow(
            row
        )


# ============================================================================
# Aggregate reporting
# ============================================================================

def print_comparison(
    label,
    rows,
    real_field,
    random_field,
):
    real_values = [
        row[
            real_field
        ]
        for row in rows
    ]

    random_values = [
        row[
            random_field
        ]
        for row in rows
    ]

    real_mean = mean_finite(
        real_values
    )

    random_mean = mean_finite(
        random_values
    )

    effect = cohen_d_absolute(
        real_values,
        random_values,
    )

    print(
        f"{label:<32} "
        f"real={real_mean:10.4f} "
        f"random={random_mean:10.4f} "
        f"|d|={effect:8.4f}"
    )


def print_aggregate_summary(
    rows,
    args,
    total_seconds,
):
    print()
    print(
        LINE
    )

    print(
        "PHASE 4B2 AGGREGATE SUMMARY"
    )

    print(
        LINE
    )

    print(
        "Independent keys             : "
        f"{len(rows)}"
    )

    print(
        "Parameters                   : "
        f"m={args.m}, "
        f"n={args.n}, "
        f"t={args.t}, "
        f"ell={args.ell}, "
        f"tau_F={args.tau_f}"
    )

    print(
        "Fused rows sampled/key       : "
        f"{args.rows}"
    )

    print()

    print_comparison(
        "Y rank deficiency",
        rows,
        "Y_rank_deficiency_real",
        "Y_rank_deficiency_bernoulli",
    )

    print_comparison(
        "Y Schur rank",
        rows,
        "Y_schur_rank_real",
        "Y_schur_rank_bernoulli",
    )

    print_comparison(
        "syndrome rank",
        rows,
        "syndrome_rank_real",
        "syndrome_rank_bernoulli",
    )

    print_comparison(
        "syndrome mean weight",
        rows,
        "syndrome_mean_weight_real",
        "syndrome_mean_weight_bernoulli",
    )

    print_comparison(
        "syndrome duplicate rows",
        rows,
        "syndrome_duplicate_rows_real",
        "syndrome_duplicate_rows_bernoulli",
    )

    print_comparison(
        "syndrome collision pairs",
        rows,
        "syndrome_collision_pairs_real",
        "syndrome_collision_pairs_bernoulli",
    )

    print_comparison(
        "syndrome Schur rank",
        rows,
        "syndrome_schur_rank_real",
        "syndrome_schur_rank_bernoulli",
    )

    print_comparison(
        "augmented-code rank",
        rows,
        "augmented_rank_real",
        "augmented_rank_bernoulli",
    )

    print_comparison(
        "augmented-code Schur rank",
        rows,
        "augmented_schur_rank_real",
        "augmented_schur_rank_bernoulli",
    )

    # ------------------------------------------------------------------
    # Denoising aggregate
    # ------------------------------------------------------------------

    attempted = [
        row
        for row in rows
        if int(
            row[
                "denoise_attempted"
            ]
        ) == 1
    ]

    skipped = [
        row
        for row in rows
        if int(
            row[
                "denoise_skipped"
            ]
        ) == 1
    ]

    print()
    print(
        "Public bounded-weight denoising"
    )

    print(
        SUBLINE
    )

    print(
        "Bounded search space          : "
        f"V({args.n},{args.tau_f}) = "
        f"{bounded_weight_search_space(args.n, args.tau_f):,}"
    )

    print(
        "Enumeration guard             : "
        f"{args.max_enumeration:,}"
    )

    print(
        "Keys attempted                : "
        f"{len(attempted)}"
    )

    print(
        "Keys skipped                  : "
        f"{len(skipped)}"
    )

    if attempted:

        print(
            "Mean row recovery fraction   : "
            f"{mean_finite([row['denoise_recovery_fraction'] for row in attempted]):.6f}"
        )

        print(
            "Mean exact-noise fraction    : "
            f"{mean_finite([row['denoise_exact_noise_fraction'] for row in attempted]):.6f}"
        )

        print(
            "Mean exact-K fraction        : "
            f"{mean_finite([row['denoise_exact_K_fraction'] for row in attempted]):.6f}"
        )

        print(
            "Mean ambiguity fraction      : "
            f"{mean_finite([row['denoise_ambiguity_fraction'] for row in attempted]):.6f}"
        )

        print(
            "Mean missing fraction        : "
            f"{mean_finite([row['denoise_missing_fraction'] for row in attempted]):.6f}"
        )

        print(
            "Mean table-build time        : "
            f"{mean_finite([row['denoise_table_build_seconds'] for row in attempted]):.6f} s"
        )

    elif skipped:

        print(
            "Attack measurements          : "
            "NaN because enumeration was skipped."
        )

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    print()
    print(
        "Timing"
    )

    print(
        SUBLINE
    )

    print(
        "Mean McEliece key generation : "
        f"{mean_finite([row['keygen_seconds'] for row in rows]):.6f} s"
    )

    print(
        "Mean transport construction   : "
        f"{mean_finite([row['transport_seconds'] for row in rows]):.6f} s"
    )

    print(
        "Mean sampled FEK generation   : "
        f"{mean_finite([row['sample_fek_seconds'] for row in rows]):.6f} s"
    )

    print()

    print(
        "Interpretation: finite differences in these statistics "
        "establish structural distinguishability only.  They do not, "
        "by themselves, establish secret-key recovery or an asymptotic "
        "attack."
    )

    print()

    print(
        "Total experiment time        : "
        f"{total_seconds:.6f} s"
    )

    print(
        "CSV written                  : "
        f"{args.csv}"
    )

    print()

    print(
        "PHASE 4B2 COMPLETE."
    )


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4B2 / 4B2.1 fused-evaluation-key structural "
            "and public-denoising experiments."
        )
    )

    parser.add_argument(
        "--m",
        type=int,
        default=6,
        help="Binary extension degree.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=63,
        help="Code length.",
    )

    parser.add_argument(
        "--t",
        type=int,
        default=3,
        help="Goppa polynomial degree / decoding radius.",
    )

    parser.add_argument(
        "--ell",
        type=int,
        default=2,
        help="Message dimension.",
    )

    parser.add_argument(
        "--tau-f",
        dest="tau_f",
        type=int,
        default=2,
        help="Exact fused-row perturbation weight.",
    )

    parser.add_argument(
        "--keys",
        type=int,
        default=20,
        help="Number of independent keys.",
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=256,
        help="Number of fused rows sampled per key.",
    )

    parser.add_argument(
        "--schur-pairs",
        dest="schur_pairs",
        type=int,
        default=4096,
        help="Maximum coordinate-wise row products used per Schur-rank test.",
    )

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help=(
            "Retained for compatibility with earlier Phase-4B2 command "
            "lines. Controlled row generation is currently rowwise."
        ),
    )

    parser.add_argument(
        "--max-enumeration",
        dest="max_enumeration",
        type=int,
        default=3000000,
        help=(
            "Maximum V(n,tau_F) for explicit public bounded-weight "
            "syndrome-table construction."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Master random seed.",
    )

    parser.add_argument(
        "--csv",
        default="phase4b2_fek_distinguishers.csv",
        help="Per-key CSV output.",
    )

    parser.add_argument(
        "--reset-csv",
        action="store_true",
        help="Delete the CSV before beginning the experiment.",
    )

    return parser


def validate_args(
    args,
):
    if args.m <= 0:
        raise ValueError(
            "m must be positive."
        )

    if args.n <= 0:
        raise ValueError(
            "n must be positive."
        )

    if args.t <= 0:
        raise ValueError(
            "t must be positive."
        )

    if args.ell <= 0:
        raise ValueError(
            "ell must be positive."
        )

    if not (
        0
        <=
        args.tau_f
        <=
        args.n
    ):
        raise ValueError(
            "tau_F must satisfy 0 <= tau_F <= n."
        )

    if args.keys <= 0:
        raise ValueError(
            "--keys must be positive."
        )

    if args.rows <= 0:
        raise ValueError(
            "--rows must be positive."
        )

    if args.schur_pairs <= 0:
        raise ValueError(
            "--schur-pairs must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.max_enumeration <= 0:
        raise ValueError(
            "--max-enumeration must be positive."
        )

    if (
        args.n
        >
        (
            2 ** args.m
            -
            1
        )
    ):
        raise ValueError(
            "Current nonzero-support Goppa implementation requires "
            "n <= 2^m - 1."
        )


def main():
    parser = build_parser()

    args = parser.parse_args()

    validate_args(
        args
    )

    search_space = (
        bounded_weight_search_space(
            args.n,
            args.tau_f,
        )
    )

    log2_search_space = (
        math.log2(
            search_space
        )
        if search_space > 0
        else 0.0
    )

    if (
        args.reset_csv
        and
        os.path.exists(
            args.csv
        )
    ):
        os.remove(
            args.csv
        )

    print(
        LINE
    )

    print(
        "PHASE 4B2 / 4B2.1 — FEK STRUCTURAL AND DENOISING ATTACKS"
    )

    print(
        LINE
    )

    print(
        "Parameters requested : "
        f"m={args.m}, "
        f"n={args.n}, "
        f"t={args.t}, "
        f"ell={args.ell}, "
        f"tau_F={args.tau_f}"
    )

    print(
        "Independent keys      : "
        f"{args.keys}"
    )

    print(
        "Sampled FEK rows/key  : "
        f"{args.rows}"
    )

    print(
        "Schur pair limit      : "
        f"{args.schur_pairs}"
    )

    print(
        "Bounded-weight search : "
        f"V({args.n},{args.tau_f}) = "
        f"{search_space:,}"
    )

    print(
        "log2 search space     : "
        f"{log2_search_space:.4f}"
    )

    print(
        "Enumeration guard     : "
        f"{args.max_enumeration:,}"
    )

    if (
        search_space
        >
        args.max_enumeration
    ):
        print(
            "Denoising status     : "
            "will be SKIPPED; structural tests will continue"
        )

    print(
        "CSV                   : "
        f"{args.csv}"
    )

    experiment_start = (
        time.perf_counter()
    )

    rows = []

    for instance_index in range(
        args.keys
    ):

        print()

        print(
            f"[{instance_index + 1}/{args.keys}] "
            "Generating independent instance..."
        )

        row = run_instance(
            instance_index=instance_index,
            args=args,
        )

        rows.append(
            row
        )

        append_csv_row(
            args.csv,
            row,
        )

    total_seconds = (
        time.perf_counter()
        -
        experiment_start
    )

    print_aggregate_summary(
        rows=rows,
        args=args,
        total_seconds=total_seconds,
    )


if __name__ == "__main__":
    main()
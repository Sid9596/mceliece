#!/usr/bin/env python3
"""
Phase 3F: controlled fused-evaluation-key baseline benchmark.

Purpose
=======

The current Phase-3 fused evaluation key contains

    N_FEK = n(n+1)/2

compressed rows

    Y_{u,v} = K_{u,v} + z_{u,v},

where

    K_{u,v}
        =
    (Lambda_u odot Lambda_v) G_pub.

At n = 1023,

    N_FEK = 523,776.

Generating the complete table before profiling can therefore be
expensive.

This script benchmarks only a controlled PREFIX of the clean fused
rows.  It deliberately does not construct the complete evaluation key.

For every benchmarked pair (u,v), it performs the same main clean-row
computations used by the Phase-3 implementation:

    product = Lambda_u odot Lambda_v
    K_uv    = product G_pub
    packed  = packbits(K_uv)

The benchmark separately measures:

    1. McEliece key-generation time,
    2. HE transport-construction time,
    3. truncated polynomial-product time,
    4. GF(2) multiplication by G_pub,
    5. bit-packing/storage time,
    6. Python/loop overhead,
    7. total fused-row time,

and extrapolates the runtime of the complete FEK.

This is a profiling tool.  It does not modify the cryptographic
construction.
"""

import argparse
import time

import numpy as np

from mceliece.mceliececipher import McElieceCipher

from mceliece.fused_he import (
    FusedMcElieceHE,
    fused_pair_count,
    gf2_matmul,
    gf2_uint8,
    truncated_product,
)


# ============================================================================
# Large experimental parameter set
# ============================================================================

DEFAULT_M = 10
DEFAULT_N = 1023
DEFAULT_T = 32
DEFAULT_ELL = 1
DEFAULT_SEED = 3

DEFAULT_BATCH_SIZE = 64


# ============================================================================
# Formatting
# ============================================================================

def format_seconds(seconds):
    """
    Human-readable representation of a duration.
    """

    seconds = float(seconds)

    if seconds < 60.0:
        return f"{seconds:.6f} s"

    minutes = seconds / 60.0

    if minutes < 60.0:
        return f"{minutes:.3f} min"

    hours = minutes / 60.0

    if hours < 24.0:
        return f"{hours:.3f} h"

    days = hours / 24.0

    return f"{days:.3f} days"


def format_bytes(byte_count):
    """
    Human-readable memory size.
    """

    byte_count = int(byte_count)

    if byte_count < 1024:
        return f"{byte_count} B"

    if byte_count < 2**20:
        return f"{byte_count / 2**10:.3f} KiB"

    if byte_count < 2**30:
        return f"{byte_count / 2**20:.3f} MiB"

    return f"{byte_count / 2**30:.3f} GiB"


# ============================================================================
# Symmetric-pair iterator
# ============================================================================

def iter_symmetric_pairs(n):
    """
    Yield pairs in the same compressed ordering used by the fused key:

        (0,0), (0,1), ..., (0,n-1),
        (1,1), (1,2), ..., (1,n-1),
        ...
        (n-1,n-1).
    """

    n = int(n)

    for u in range(n):

        for v in range(
            u,
            n,
        ):
            yield (
                u,
                v,
            )


# ============================================================================
# Prefix benchmark
# ============================================================================

def benchmark_prefix(
    he,
    row_limit,
    batch_size,
):
    """
    Benchmark exactly row_limit clean fused rows.

    The benchmark follows the same main computation as

        generate_evaluation_key()

    but stores only the requested prefix and uses tau_F = 0.

    This allows performance profiling without allocating the entire
    n(n+1)/2-row public evaluation key.
    """

    row_limit = int(
        row_limit
    )

    batch_size = int(
        batch_size
    )

    total_pairs = fused_pair_count(
        he.n
    )

    if not (
        1
        <=
        row_limit
        <=
        total_pairs
    ):
        raise ValueError(
            "row_limit must satisfy "
            f"1 <= row_limit <= {total_pairs}."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    # ------------------------------------------------------------------
    # Public generator
    # ------------------------------------------------------------------

    G_pub = gf2_uint8(
        he.mc.Gp
    )

    # ------------------------------------------------------------------
    # Packed-row size
    # ------------------------------------------------------------------

    row_bytes = (
        he.n
        +
        7
    ) // 8

    # Only allocate memory for the benchmarked prefix.
    packed_prefix = np.empty(
        (
            row_limit,
            row_bytes,
        ),
        dtype=np.uint8,
    )

    # ------------------------------------------------------------------
    # Timing counters
    # ------------------------------------------------------------------

    product_seconds = 0.0
    matmul_seconds = 0.0
    packing_seconds = 0.0

    product_batch = []

    write_index = 0

    # ------------------------------------------------------------------
    # Flush one product batch
    # ------------------------------------------------------------------

    def flush_batch():

        nonlocal product_batch
        nonlocal write_index
        nonlocal matmul_seconds
        nonlocal packing_seconds

        if not product_batch:
            return

        products = np.asarray(
            product_batch,
            dtype=np.uint8,
        )

        # --------------------------------------------------------------
        # (Lambda_u odot Lambda_v) G_pub
        # --------------------------------------------------------------

        start = time.perf_counter()

        clean_rows = gf2_matmul(
            products,
            G_pub,
        )

        matmul_seconds += (
            time.perf_counter()
            -
            start
        )

        # --------------------------------------------------------------
        # Pack the resulting n-bit rows.
        # --------------------------------------------------------------

        start = time.perf_counter()

        packed = np.packbits(
            clean_rows,
            axis=1,
            bitorder="little",
        )

        batch_length = int(
            packed.shape[0]
        )

        packed_prefix[
            write_index:
            write_index
            +
            batch_length
        ] = packed

        write_index += (
            batch_length
        )

        packing_seconds += (
            time.perf_counter()
            -
            start
        )

        product_batch = []

    # ------------------------------------------------------------------
    # Generate controlled prefix
    # ------------------------------------------------------------------

    pair_iterator = (
        iter_symmetric_pairs(
            he.n
        )
    )

    start_total = (
        time.perf_counter()
    )

    for _ in range(
        row_limit
    ):

        u, v = next(
            pair_iterator
        )

        # --------------------------------------------------------------
        # Lambda_u odot Lambda_v
        # --------------------------------------------------------------

        start = (
            time.perf_counter()
        )

        product = (
            truncated_product(
                he.Lambda[
                    u
                ],
                he.Lambda[
                    v
                ],
                output_length=he.k,
            )
        )

        product_seconds += (
            time.perf_counter()
            -
            start
        )

        product_batch.append(
            product
        )

        if (
            len(
                product_batch
            )
            >=
            batch_size
        ):
            flush_batch()

    flush_batch()

    total_seconds = (
        time.perf_counter()
        -
        start_total
    )

    # ------------------------------------------------------------------
    # Correctness of benchmark machinery
    # ------------------------------------------------------------------

    if (
        write_index
        !=
        row_limit
    ):
        raise RuntimeError(
            "Incorrect number of generated benchmark rows: "
            f"{write_index} != {row_limit}."
        )

    # Force the generated array to be consumed.
    checksum = int(
        np.bitwise_xor.reduce(
            packed_prefix.reshape(
                -1
            )
        )
    )

    # ------------------------------------------------------------------
    # Derived performance values
    # ------------------------------------------------------------------

    accounted_seconds = (
        product_seconds
        +
        matmul_seconds
        +
        packing_seconds
    )

    overhead_seconds = max(
        0.0,
        total_seconds
        -
        accounted_seconds,
    )

    seconds_per_row = (
        total_seconds
        /
        float(
            row_limit
        )
    )

    projected_full_seconds = (
        seconds_per_row
        *
        float(
            total_pairs
        )
    )

    return {
        "rows": (
            row_limit
        ),

        "total_pairs": (
            total_pairs
        ),

        "row_bytes": (
            row_bytes
        ),

        "prefix_bytes": int(
            packed_prefix.nbytes
        ),

        "product_seconds": (
            product_seconds
        ),

        "matmul_seconds": (
            matmul_seconds
        ),

        "packing_seconds": (
            packing_seconds
        ),

        "overhead_seconds": (
            overhead_seconds
        ),

        "total_seconds": (
            total_seconds
        ),

        "seconds_per_row": (
            seconds_per_row
        ),

        "projected_full_seconds": (
            projected_full_seconds
        ),

        "checksum": (
            checksum
        ),
    }


# ============================================================================
# Report
# ============================================================================

def print_result(
    result,
):
    """
    Print one benchmark report.
    """

    total = float(
        result[
            "total_seconds"
        ]
    )

    def percentage(
        value,
    ):
        if total <= 0.0:
            return 0.0

        return (
            100.0
            *
            float(
                value
            )
            /
            total
        )

    print()
    print("=" * 88)

    print(
        "BENCHMARK PREFIX: "
        f"{result['rows']:,} ROWS"
    )

    print("=" * 88)

    print(
        "Full FEK pair count          : "
        f"{result['total_pairs']:,}"
    )

    print(
        "Packed bytes per row         : "
        f"{result['row_bytes']}"
    )

    print(
        "Stored benchmark prefix      : "
        f"{format_bytes(result['prefix_bytes'])}"
    )

    print()

    print(
        "Truncated products           : "
        f"{result['product_seconds']:.6f} s "
        f"("
        f"{percentage(result['product_seconds']):.2f}%"
        f")"
    )

    print(
        "GF(2) products * G_pub       : "
        f"{result['matmul_seconds']:.6f} s "
        f"("
        f"{percentage(result['matmul_seconds']):.2f}%"
        f")"
    )

    print(
        "Pack/store                   : "
        f"{result['packing_seconds']:.6f} s "
        f"("
        f"{percentage(result['packing_seconds']):.2f}%"
        f")"
    )

    print(
        "Loop/Python overhead         : "
        f"{result['overhead_seconds']:.6f} s "
        f"("
        f"{percentage(result['overhead_seconds']):.2f}%"
        f")"
    )

    print()

    print(
        "Total benchmark time         : "
        f"{format_seconds(result['total_seconds'])}"
    )

    print(
        "Mean time per fused row      : "
        f"{result['seconds_per_row']:.9f} s"
    )

    print(
        "Projected complete FEK time  : "
        f"{format_seconds(result['projected_full_seconds'])}"
    )

    print(
        "Checksum                     : "
        f"{result['checksum']}"
    )


# ============================================================================
# Main
# ============================================================================

def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Controlled Phase-3F fused-evaluation-key "
                "baseline benchmark."
            )
        )
    )

    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        default=[
            100,
            1000,
            10000,
        ],
        help=(
            "Numbers of clean fused rows to benchmark. "
            "Example: --rows 100"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of truncated products multiplied "
            "by G_pub in one batch."
        ),
    )

    parser.add_argument(
        "--m",
        type=int,
        default=DEFAULT_M,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
    )

    parser.add_argument(
        "--t",
        type=int,
        default=DEFAULT_T,
    )

    parser.add_argument(
        "--ell",
        type=int,
        default=DEFAULT_ELL,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = (
        parser.parse_args()
    )

    m = int(
        args.m
    )

    n = int(
        args.n
    )

    t = int(
        args.t
    )

    ell = int(
        args.ell
    )

    seed = int(
        args.seed
    )

    batch_size = int(
        args.batch_size
    )

    if (
        m <= 0
        or
        n <= 0
        or
        t <= 0
        or
        ell <= 0
    ):
        raise ValueError(
            "m, n, t and ell must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch size must be positive."
        )

    print()
    print("#" * 88)

    print(
        "PHASE 3F FUSED-EVALUATION-KEY "
        "BASELINE BENCHMARK"
    )

    print("#" * 88)

    print(
        f"(m,n,t,ell) = "
        f"({m},{n},{t},{ell})"
    )

    print(
        f"seed = {seed}"
    )

    print(
        f"batch size = {batch_size}"
    )

    print(
        "requested prefixes = "
        f"{args.rows}"
    )

    print()

    # Existing McEliece key generation uses NumPy's legacy RNG.
    np.random.seed(
        seed
    )

    # ==================================================================
    # McEliece key generation
    # ==================================================================

    print(
        "Generating McEliece key..."
    )

    start = (
        time.perf_counter()
    )

    mc = (
        McElieceCipher(
            m,
            n,
            t,
        )
    )

    mc.generate_random_keys()

    keygen_seconds = (
        time.perf_counter()
        -
        start
    )

    print()

    print(
        f"Derived k                    : "
        f"{mc.k}"
    )

    print(
        "McEliece key generation     : "
        f"{format_seconds(keygen_seconds)}"
    )

    # ==================================================================
    # HE transport
    # ==================================================================

    print()
    print(
        "Constructing randomized HE transport..."
    )

    he = (
        FusedMcElieceHE(
            mc,
            ell=ell,
            tau_enc=0,
            tau_F=0,
            seed=seed + 10000,
        )
    )

    start = (
        time.perf_counter()
    )

    he.construct_transport(
        randomization_trials=64,
        store_right_inverse=False,
    )

    transport_seconds = (
        time.perf_counter()
        -
        start
    )

    print()

    print(
        "G_pub Lambda = I_k           : "
        f"{he.transport_diagnostics['GpLambda_identity']}"
    )

    print(
        "Lambda zero rows             : "
        f"{he.transport_diagnostics['lambda_zero_rows']}"
    )

    print(
        "Lambda row-weight range      : "
        f"["
        f"{he.transport_diagnostics['lambda_row_weight_min']}, "
        f"{he.transport_diagnostics['lambda_row_weight_max']}"
        f"]"
    )

    print(
        "Lambda mean row weight       : "
        f"{he.transport_diagnostics['lambda_row_weight_mean']:.3f}"
    )

    print(
        "Transport construction       : "
        f"{format_seconds(transport_seconds)}"
    )

    # ==================================================================
    # FEK dimensions
    # ==================================================================

    total_pairs = (
        fused_pair_count(
            n
        )
    )

    row_bytes = (
        n
        +
        7
    ) // 8

    full_packed_bytes = (
        total_pairs
        *
        row_bytes
    )

    print()
    print(
        "Full compressed FEK rows     : "
        f"{total_pairs:,}"
    )

    print(
        "Packed bytes per row         : "
        f"{row_bytes}"
    )

    print(
        "Packed complete FEK size     : "
        f"{format_bytes(full_packed_bytes)}"
    )

    # ==================================================================
    # Controlled prefix benchmarks
    # ==================================================================

    previous_limit = 0

    for row_limit in (
        args.rows
    ):

        row_limit = int(
            row_limit
        )

        if row_limit <= 0:
            raise ValueError(
                "Every --rows value must be positive."
            )

        if row_limit > total_pairs:
            raise ValueError(
                f"Requested {row_limit:,} rows, "
                f"but only {total_pairs:,} symmetric "
                "pairs exist."
            )

        if row_limit < previous_limit:
            print()
            print(
                "Warning: benchmark prefix sizes "
                "are not increasing."
            )

        result = (
            benchmark_prefix(
                he=he,
                row_limit=row_limit,
                batch_size=batch_size,
            )
        )

        print_result(
            result
        )

        previous_limit = (
            row_limit
        )

    print()
    print("=" * 88)

    print(
        "PHASE 3F BASELINE BENCHMARK COMPLETE"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
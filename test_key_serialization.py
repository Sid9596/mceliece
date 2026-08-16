"""
Phase 2C key-serialization regression.

This test verifies the complete lifecycle

    KeyGen
        ->
    serialize private/public keys
        ->
    reload keys with allow_pickle=False
        ->
    validate G R = I_k
        ->
    encrypt with the serialized public key
        ->
    decrypt with the serialized private key.

Phase 2C extends the private-key format with

    R                      : shape (n,k)
    right_inverse_pivots   : shape (k,)

where

    G R = I_k.

The test also verifies that disk-loaded decryption does not reconstruct
R. The stored right inverse must be used directly.
"""

import importlib.util
from pathlib import Path
import tempfile

import numpy as np

from mceliece.mathutils import GF2Matrix


# ================================================================
# Configuration
# ================================================================

M = 6
N = 63
T = 3

TRIALS = 10

RANDOM_SEED = 20260816

EXPECTED_KEY_FORMAT_VERSION = 3


# ================================================================
# Load top-level mceliece.py as a module
# ================================================================

PROJECT_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parent
)

CLI_FILE = (
    PROJECT_ROOT
    / "mceliece.py"
)


def load_cli_module():
    """
    Load the top-level mceliece.py without confusing it with the
    mceliece/ package.
    """

    spec = (
        importlib.util.spec_from_file_location(
            "mceliece_cli",
            CLI_FILE,
        )
    )

    if spec is None:
        raise RuntimeError(
            "Unable to create module specification "
            "for mceliece.py."
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    if spec.loader is None:
        raise RuntimeError(
            "mceliece.py module loader is unavailable."
        )

    spec.loader.exec_module(
        module
    )

    return module


# ================================================================
# Helpers
# ================================================================

def gf2_numpy(matrix):
    """
    Convert GF2Matrix to uint8 while preserving shape.
    """

    if not isinstance(
        matrix,
        GF2Matrix,
    ):
        raise TypeError(
            "Expected GF2Matrix."
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


def matrix_equal(
    A,
    B,
):
    """
    Compare two GF2Matrix objects.
    """

    return np.array_equal(
        gf2_numpy(
            A
        ),
        gf2_numpy(
            B
        ),
    )


def identity_gf2(size):
    """
    Identity matrix over GF(2).
    """

    return GF2Matrix.from_list(
        np.eye(
            size,
            dtype=np.uint8,
        )
    )


def is_valid_permutation(
    P,
    n,
):
    """
    Verify that P is a permutation of 0,...,n-1.
    """

    P = np.asarray(
        P,
        dtype=np.int64,
    ).reshape(-1)

    return (
        P.shape == (
            n,
        )
        and
        np.array_equal(
            np.sort(
                P
            ),
            np.arange(
                n,
                dtype=np.int64,
            ),
        )
    )


def validate_serialized_pivots(
    pivots,
    n,
    k,
):
    """
    Verify serialized right-inverse pivot indices.
    """

    pivots = np.asarray(
        pivots,
        dtype=np.int64,
    ).reshape(-1)

    return (
        pivots.shape == (
            k,
        )
        and
        len(
            np.unique(
                pivots
            )
        ) == k
        and
        np.all(
            pivots >= 0
        )
        and
        np.all(
            pivots < n
        )
    )


def check_right_inverse(
    G,
    R,
):
    """
    Check

        G R = I_k.
    """

    k = G.arr.shape[0]

    GR = (
        G
        * R
    )

    return matrix_equal(
        GR,
        identity_gf2(
            k
        ),
    )


# ================================================================
# Main regression
# ================================================================

def main():

    np.random.seed(
        RANDOM_SEED
    )

    cli = load_cli_module()

    print()
    print("=" * 78)

    print(
        "PHASE 2C KEY-SERIALIZATION REGRESSION"
    )

    print("=" * 78)

    print(
        f"Parameters: "
        f"m={M}, "
        f"n={N}, "
        f"t={T}"
    )

    # ------------------------------------------------------------
    # Temporary key directory
    # ------------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="mceliece_phase2c_"
    ) as temp_dir:

        temp_dir = Path(
            temp_dir
        )

        private_key_file = (
            temp_dir
            / "private_key.npz"
        )

        public_key_file = (
            temp_dir
            / "public_key.npz"
        )

        # ========================================================
        # 1. Generate and serialize key pair
        # ========================================================

        print()
        print(
            "[1] Generating and serializing keys..."
        )

        generated = cli.generate(
            M,
            N,
            T,
            private_key_file,
            public_key_file,
        )

        k = int(
            generated.k
        )

        print(
            f"  generated k = {k}"
        )

        print(
            "  generated R shape =",
            generated.R.arr.shape,
        )

        print(
            "  generated pivot count =",
            len(
                generated.right_inverse_pivots
            ),
        )

        print(
            "  private key exists :",
            private_key_file.exists(),
        )

        print(
            "  public key exists  :",
            public_key_file.exists(),
        )

        assert private_key_file.exists()

        assert public_key_file.exists()

        assert generated.R.arr.shape == (
            N,
            k,
        )

        assert (
            generated.right_inverse_pivots
            .shape
            ==
            (
                k,
            )
        )

        generated_GR_ok = (
            check_right_inverse(
                generated.G,
                generated.R,
            )
        )

        print(
            "  generated G R = I_k :",
            generated_GR_ok,
        )

        assert generated_GR_ok

        # ========================================================
        # 2. Inspect serialized private key
        # ========================================================

        print()
        print(
            "[2] Inspecting serialized private key..."
        )

        with np.load(
            private_key_file,
            allow_pickle=False,
        ) as private_data:

            fields = sorted(
                private_data.files
            )

            version = int(
                np.asarray(
                    private_data[
                        "format_version"
                    ]
                ).item()
            )

            serialized_G = np.asarray(
                private_data["G"],
                dtype=np.uint8,
            )

            serialized_H = np.asarray(
                private_data["H"],
                dtype=np.uint8,
            )

            serialized_R = np.asarray(
                private_data["R"],
                dtype=np.uint8,
            )

            serialized_pivots = np.asarray(
                private_data[
                    "right_inverse_pivots"
                ],
                dtype=np.int64,
            ).reshape(-1)

            serialized_S = np.asarray(
                private_data["S"],
                dtype=np.uint8,
            )

            serialized_S_inv = np.asarray(
                private_data["S_inv"],
                dtype=np.uint8,
            )

            serialized_P = np.asarray(
                private_data["P"],
                dtype=np.int64,
            ).reshape(-1)

            serialized_P_inv = np.asarray(
                private_data["P_inv"],
                dtype=np.int64,
            ).reshape(-1)

            serialized_g_poly = np.asarray(
                private_data[
                    "g_poly"
                ],
                dtype=np.uint8,
            )

            serialized_irr_poly = np.asarray(
                private_data[
                    "irr_poly"
                ],
                dtype=np.uint8,
            )

        print(
            "  fields =",
            fields,
        )

        print(
            "  format version =",
            version,
        )

        print(
            "  G shape                     =",
            serialized_G.shape,
        )

        print(
            "  H shape                     =",
            serialized_H.shape,
        )

        print(
            "  R shape                     =",
            serialized_R.shape,
        )

        print(
            "  right_inverse_pivots shape  =",
            serialized_pivots.shape,
        )

        print(
            "  S shape                     =",
            serialized_S.shape,
        )

        print(
            "  S_inv shape                 =",
            serialized_S_inv.shape,
        )

        print(
            "  P shape                     =",
            serialized_P.shape,
        )

        print(
            "  P_inv shape                 =",
            serialized_P_inv.shape,
        )

        print(
            "  g_poly shape                =",
            serialized_g_poly.shape,
        )

        print(
            "  irr_poly shape              =",
            serialized_irr_poly.shape,
        )

        required_private_fields = {
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

        assert set(
            fields
        ) == required_private_fields

        assert version == (
            EXPECTED_KEY_FORMAT_VERSION
        )

        assert serialized_G.shape == (
            k,
            N,
        )

        assert serialized_H.shape[1] == N

        assert serialized_R.shape == (
            N,
            k,
        )

        assert serialized_pivots.shape == (
            k,
        )

        assert serialized_S.shape == (
            k,
            k,
        )

        assert serialized_S_inv.shape == (
            k,
            k,
        )

        assert serialized_P.shape == (
            N,
        )

        assert serialized_P_inv.shape == (
            N,
        )

        # --------------------------------------------------------
        # Validate serialized permutation.
        # --------------------------------------------------------

        assert is_valid_permutation(
            serialized_P,
            N,
        )

        assert is_valid_permutation(
            serialized_P_inv,
            N,
        )

        identity_positions = np.arange(
            N,
            dtype=np.int64,
        )

        assert np.array_equal(
            serialized_P[
                serialized_P_inv
            ],
            identity_positions,
        )

        assert np.array_equal(
            serialized_P_inv[
                serialized_P
            ],
            identity_positions,
        )

        print(
            "  permutation serialization    : PASS"
        )

        # --------------------------------------------------------
        # Validate serialized pivot vector.
        # --------------------------------------------------------

        pivots_ok = (
            validate_serialized_pivots(
                serialized_pivots,
                N,
                k,
            )
        )

        print(
            "  right-inverse pivots         :",
            pivots_ok,
        )

        assert pivots_ok

        # --------------------------------------------------------
        # Validate serialized G R = I_k directly.
        # --------------------------------------------------------

        serialized_G_gf2 = (
            GF2Matrix.from_list(
                serialized_G
            )
        )

        serialized_R_gf2 = (
            GF2Matrix.from_list(
                serialized_R
            )
        )

        serialized_GR_ok = (
            check_right_inverse(
                serialized_G_gf2,
                serialized_R_gf2,
            )
        )

        print(
            "  serialized G R = I_k         :",
            serialized_GR_ok,
        )

        assert serialized_GR_ok

        # --------------------------------------------------------
        # Verify sparse-selector structure of R.
        # --------------------------------------------------------

        nonpivot_mask = np.ones(
            N,
            dtype=bool,
        )

        nonpivot_mask[
            serialized_pivots
        ] = False

        nonpivot_rows_zero = (
            not np.any(
                serialized_R[
                    nonpivot_mask,
                    :
                ]
            )
        )

        print(
            "  nonpivot rows of R are zero  :",
            nonpivot_rows_zero,
        )

        assert nonpivot_rows_zero

        # ========================================================
        # 3. Inspect serialized public key
        # ========================================================

        print()
        print(
            "[3] Inspecting serialized public key..."
        )

        with np.load(
            public_key_file,
            allow_pickle=False,
        ) as public_data:

            public_version = int(
                np.asarray(
                    public_data[
                        "format_version"
                    ]
                ).item()
            )

            serialized_Gp = np.asarray(
                public_data["Gp"],
                dtype=np.uint8,
            )

            serialized_k = int(
                np.asarray(
                    public_data["k"]
                ).item()
            )

        print(
            "  format version =",
            public_version,
        )

        print(
            "  G_pub shape    =",
            serialized_Gp.shape,
        )

        assert public_version == (
            EXPECTED_KEY_FORMAT_VERSION
        )

        assert serialized_k == k

        assert serialized_Gp.shape == (
            k,
            N,
        )

        # ========================================================
        # 4. Reload both keys
        # ========================================================

        print()
        print(
            "[4] Reloading keys with allow_pickle=False..."
        )

        public_mc = (
            cli.load_public_key(
                public_key_file
            )
        )

        private_mc = (
            cli.load_private_key(
                private_key_file
            )
        )

        print(
            "  public G_pub shape          =",
            public_mc.Gp.arr.shape,
        )

        print(
            "  private G shape             =",
            private_mc.G.arr.shape,
        )

        print(
            "  private R shape             =",
            private_mc.R.arr.shape,
        )

        print(
            "  private pivots shape        =",
            private_mc
            .right_inverse_pivots
            .shape,
        )

        print(
            "  private P shape             =",
            private_mc.P.shape,
        )

        print(
            "  private P_inv shape         =",
            private_mc.P_inv.shape,
        )

        assert public_mc.Gp.arr.shape == (
            k,
            N,
        )

        assert private_mc.G.arr.shape == (
            k,
            N,
        )

        assert private_mc.R is not None

        assert private_mc.R.arr.shape == (
            N,
            k,
        )

        assert (
            private_mc
            .right_inverse_pivots
            .shape
            ==
            (
                k,
            )
        )

        assert private_mc.P.shape == (
            N,
        )

        assert private_mc.P_inv.shape == (
            N,
        )

        # --------------------------------------------------------
        # Loaded right-inverse invariant.
        # --------------------------------------------------------

        loaded_GR_ok = (
            check_right_inverse(
                private_mc.G,
                private_mc.R,
            )
        )

        print(
            "  loaded G R = I_k            :",
            loaded_GR_ok,
        )

        assert loaded_GR_ok

        # ========================================================
        # 5. Compare generated and reloaded key material
        # ========================================================

        print()
        print(
            "[5] Checking key-material round trip..."
        )

        assert matrix_equal(
            generated.G,
            private_mc.G,
        )

        assert matrix_equal(
            generated.H,
            private_mc.H,
        )

        assert matrix_equal(
            generated.R,
            private_mc.R,
        )

        assert matrix_equal(
            generated.S,
            private_mc.S,
        )

        assert matrix_equal(
            generated.S_inv,
            private_mc.S_inv,
        )

        assert matrix_equal(
            generated.Gp,
            public_mc.Gp,
        )

        assert np.array_equal(
            generated.P,
            private_mc.P,
        )

        assert np.array_equal(
            generated.P_inv,
            private_mc.P_inv,
        )

        assert np.array_equal(
            generated.right_inverse_pivots,
            private_mc.right_inverse_pivots,
        )

        print(
            "  G round trip                 : PASS"
        )

        print(
            "  H round trip                 : PASS"
        )

        print(
            "  R round trip                 : PASS"
        )

        print(
            "  right_inverse_pivots         : PASS"
        )

        print(
            "  S/S_inv round trip           : PASS"
        )

        print(
            "  P/P_inv round trip           : PASS"
        )

        print(
            "  G_pub round trip             : PASS"
        )

        # ========================================================
        # 6. Public-generator reconstruction after reload
        # ========================================================

        print()
        print(
            "[6] Checking reloaded "
            "G_pub = (S G)[:,P]..."
        )

        SG = (
            private_mc.S
            * private_mc.G
        )

        reconstructed_Gp = (
            GF2Matrix(
                SG.arr[
                    :,
                    private_mc.P
                ]
            )
        )

        public_generator_ok = (
            matrix_equal(
                reconstructed_Gp,
                public_mc.Gp,
            )
        )

        print(
            "  public-generator consistency :",
            public_generator_ok,
        )

        assert public_generator_ok

        # ========================================================
        # 7. Clean extraction through loaded R
        #
        #     (a G) R = a.
        # ========================================================

        print()
        print(
            "[7] Checking clean extraction "
            "with disk-loaded R..."
        )

        CLEAN_EXTRACTION_TRIALS = 10

        for trial in range(
            1,
            CLEAN_EXTRACTION_TRIALS + 1,
        ):

            a_numpy = np.random.randint(
                0,
                2,
                size=k,
                dtype=np.uint8,
            )

            a = GF2Matrix.from_list(
                a_numpy
            )

            clean_codeword = (
                a
                * private_mc.G
            )

            recovered = (
                clean_codeword
                * private_mc.R
            )

            recovered_numpy = (
                recovered
                .to_numpy()
                .astype(
                    np.uint8
                )
            )

            extraction_ok = (
                np.array_equal(
                    a_numpy,
                    recovered_numpy,
                )
            )

            print(
                f"  clean trial {trial:02d}: "
                f"{extraction_ok}"
            )

            assert extraction_ok

        # ========================================================
        # 8. Ensure disk-loaded decryption does not reconstruct R
        # ========================================================

        print()
        print(
            "[8] Disabling right-inverse reconstruction..."
        )

        original_construct_right_inverse = (
            cli.McElieceCipher
            ._construct_right_inverse
        )

        def forbidden_right_inverse_reconstruction(
            self,
        ):
            raise AssertionError(
                "Disk-loaded Phase 2C decryption "
                "attempted to reconstruct R."
            )

        cli.McElieceCipher._construct_right_inverse = (
            forbidden_right_inverse_reconstruction
        )

        print(
            "  _construct_right_inverse disabled"
        )

        # ========================================================
        # 9. Encrypt/decrypt through serialized key files
        # ========================================================

        print()
        print(
            f"[9] Running {TRIALS} disk-key "
            "encryption/decryption trials..."
        )

        successes = 0

        try:

            for trial in range(
                1,
                TRIALS + 1,
            ):

                message = np.random.randint(
                    0,
                    2,
                    size=k,
                    dtype=np.uint8,
                )

                # ------------------------------------------------
                # Encrypt by independently loading PUBLIC key.
                # ------------------------------------------------

                ciphertext = cli.encrypt(
                    public_key_file,
                    message,
                    block=False,
                )

                # ------------------------------------------------
                # Decrypt by independently loading PRIVATE key.
                #
                # If R were not loaded from disk, decode() would
                # attempt _construct_right_inverse(), which has
                # deliberately been disabled above.
                # ------------------------------------------------

                decoded = cli.decrypt(
                    private_key_file,
                    ciphertext,
                    block=False,
                )

                success = np.array_equal(
                    message,
                    decoded,
                )

                if success:
                    successes += 1

                print()
                print(
                    f"  Trial {trial:02d}:"
                )

                print(
                    "    message length    =",
                    len(
                        message
                    ),
                )

                print(
                    "    ciphertext length =",
                    len(
                        ciphertext
                    ),
                )

                print(
                    "    decoded length    =",
                    len(
                        decoded
                    ),
                )

                print(
                    "    success           =",
                    success,
                )

                assert len(
                    ciphertext
                ) == N

                assert len(
                    decoded
                ) == k

                if not success:

                    print(
                        "    message =",
                        message,
                    )

                    print(
                        "    decoded =",
                        decoded,
                    )

                    raise AssertionError(
                        "Serialized-key "
                        "encryption/decryption failed."
                    )

        finally:

            # Restore the class method even if a regression fails.
            cli.McElieceCipher._construct_right_inverse = (
                original_construct_right_inverse
            )

        # ========================================================
        # 10. Final result
        # ========================================================

        print()
        print("=" * 78)

        print(
            "PHASE 2C SERIALIZATION RESULT"
        )

        print("=" * 78)

        print(
            "Primitive-array key format       : PASS"
        )

        print(
            "Key format version 3             : PASS"
        )

        print(
            "allow_pickle=False loading       : PASS"
        )

        print(
            "Permutation-vector persistence   : PASS"
        )

        print(
            "Right-inverse persistence        : PASS"
        )

        print(
            "Right-inverse pivot persistence  : PASS"
        )

        print(
            "Serialized G R = I_k             : PASS"
        )

        print(
            "Loaded G R = I_k                 : PASS"
        )

        print(
            "(a G) R = a after reload         : PASS"
        )

        print(
            "No R reconstruction on decrypt   : PASS"
        )

        print(
            "Private-key round trip           : PASS"
        )

        print(
            "Public-key round trip            : PASS"
        )

        print(
            "Public-generator reconstruction  : PASS"
        )

        print(
            "Patterson decoding after reload  : PASS"
        )

        print(
            "Disk-key encryption/decryption   : "
            f"{successes}/{TRIALS} PASS"
        )

        print()

        print(
            "PHASE 2C KEY SERIALIZATION "
            "WORKS CORRECTLY."
        )


if __name__ == "__main__":
    main()
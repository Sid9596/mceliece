"""
Phase 2B key-serialization regression.

This test verifies the complete lifecycle

    KeyGen
        ->
    serialize private/public keys
        ->
    close original in-memory key objects
        ->
    reload keys from disk with allow_pickle=False
        ->
    encrypt with reloaded public key
        ->
    decrypt with reloaded private key.

It also verifies that the Phase 2B secret permutation is stored as

    P     : shape (n,)
    P_inv : shape (n,)

instead of dense n x n matrices.
"""

import importlib.util
import os
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


# ================================================================
# Load the top-level mceliece.py as a module
# ================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

CLI_FILE = (
    PROJECT_ROOT
    / "mceliece.py"
)


def load_cli_module():
    """
    Load the top-level mceliece.py file without confusing it with
    the mceliece/ Python package.
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
    Convert GF2Matrix to uint8 while retaining its shape.
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
    return np.array_equal(
        gf2_numpy(A),
        gf2_numpy(B),
    )


def is_valid_permutation(
    P,
    n,
):
    P = np.asarray(
        P,
        dtype=np.int64,
    ).reshape(-1)

    return (
        P.shape == (n,)
        and
        np.array_equal(
            np.sort(P),
            np.arange(
                n,
                dtype=np.int64,
            ),
        )
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
        "PHASE 2B KEY-SERIALIZATION REGRESSION"
    )
    print("=" * 78)

    print(
        f"Parameters: "
        f"m={M}, n={N}, t={T}"
    )

    # ------------------------------------------------------------
    # Temporary key directory
    # ------------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="mceliece_phase2b_"
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

        # --------------------------------------------------------
        # 1. Generate and serialize key pair
        # --------------------------------------------------------

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

        k = generated.k

        print(
            f"  generated k = {k}"
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

        # --------------------------------------------------------
        # 2. Inspect serialized private-key structure
        # --------------------------------------------------------

        print()
        print(
            "[2] Inspecting serialized private key..."
        )

        with np.load(
            private_key_file,
            allow_pickle=False,
        ) as private_data:

            print(
                "  fields =",
                sorted(
                    private_data.files
                ),
            )

            serialized_P = np.asarray(
                private_data["P"]
            )

            serialized_P_inv = np.asarray(
                private_data["P_inv"]
            )

            serialized_G = np.asarray(
                private_data["G"]
            )

            serialized_H = np.asarray(
                private_data["H"]
            )

            serialized_S = np.asarray(
                private_data["S"]
            )

            serialized_S_inv = np.asarray(
                private_data["S_inv"]
            )

            serialized_g_poly = np.asarray(
                private_data["g_poly"]
            )

            serialized_irr_poly = np.asarray(
                private_data["irr_poly"]
            )

            version = int(
                private_data[
                    "format_version"
                ]
            )

        print(
            "  format version =",
            version,
        )

        print(
            "  G shape     =",
            serialized_G.shape,
        )

        print(
            "  H shape     =",
            serialized_H.shape,
        )

        print(
            "  S shape     =",
            serialized_S.shape,
        )

        print(
            "  S_inv shape =",
            serialized_S_inv.shape,
        )

        print(
            "  P shape     =",
            serialized_P.shape,
        )

        print(
            "  P_inv shape =",
            serialized_P_inv.shape,
        )

        print(
            "  g_poly shape   =",
            serialized_g_poly.shape,
        )

        print(
            "  irr_poly shape =",
            serialized_irr_poly.shape,
        )

        assert version == 2

        assert serialized_G.shape == (
            k,
            N,
        )

        assert serialized_H.shape[1] == N

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

        assert is_valid_permutation(
            serialized_P,
            N,
        )

        assert is_valid_permutation(
            serialized_P_inv,
            N,
        )

        identity = np.arange(
            N,
            dtype=np.int64,
        )

        assert np.array_equal(
            serialized_P[
                serialized_P_inv
            ],
            identity,
        )

        assert np.array_equal(
            serialized_P_inv[
                serialized_P
            ],
            identity,
        )

        print(
            "  permutation serialization : PASS"
        )

        # --------------------------------------------------------
        # 3. Inspect public key
        # --------------------------------------------------------

        print()
        print(
            "[3] Inspecting serialized public key..."
        )

        with np.load(
            public_key_file,
            allow_pickle=False,
        ) as public_data:

            public_version = int(
                public_data[
                    "format_version"
                ]
            )

            serialized_Gp = np.asarray(
                public_data["Gp"]
            )

            serialized_k = int(
                public_data["k"]
            )

        print(
            "  format version =",
            public_version,
        )

        print(
            "  G_pub shape    =",
            serialized_Gp.shape,
        )

        assert public_version == 2

        assert serialized_k == k

        assert serialized_Gp.shape == (
            k,
            N,
        )

        # --------------------------------------------------------
        # 4. Reload both keys
        # --------------------------------------------------------

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
            "  public G_pub shape =",
            public_mc.Gp.arr.shape,
        )

        print(
            "  private G shape    =",
            private_mc.G.arr.shape,
        )

        print(
            "  private P shape    =",
            private_mc.P.shape,
        )

        print(
            "  private P_inv shape=",
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

        assert private_mc.P.shape == (
            N,
        )

        assert private_mc.P_inv.shape == (
            N,
        )

        # --------------------------------------------------------
        # 5. Compare generated and reloaded key material
        # --------------------------------------------------------

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

        print(
            "  key-material round trip : PASS"
        )

        # --------------------------------------------------------
        # 6. Public-generator reconstruction after reload
        # --------------------------------------------------------

        print()
        print(
            "[6] Checking reloaded G_pub = (S G)[:,P]..."
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

        # --------------------------------------------------------
        # 7. Encrypt/decrypt through serialized key files
        # --------------------------------------------------------

        print()
        print(
            f"[7] Running {TRIALS} disk-key "
            "encryption/decryption trials..."
        )

        successes = 0

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

            # ----------------------------------------------------
            # Encrypt by reopening PUBLIC key file.
            # ----------------------------------------------------

            ciphertext = cli.encrypt(
                public_key_file,
                message,
                block=False,
            )

            # ----------------------------------------------------
            # Decrypt by independently reopening PRIVATE key file.
            # ----------------------------------------------------

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
                len(message),
            )

            print(
                "    ciphertext length =",
                len(ciphertext),
            )

            print(
                "    decoded length    =",
                len(decoded),
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

        # --------------------------------------------------------
        # Final result
        # --------------------------------------------------------

        print()
        print("=" * 78)
        print(
            "PHASE 2B SERIALIZATION RESULT"
        )
        print("=" * 78)

        print(
            "Primitive-array key format       : PASS"
        )

        print(
            "allow_pickle=False loading       : PASS"
        )

        print(
            "Permutation-vector persistence   : PASS"
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
            "PHASE 2B KEY SERIALIZATION "
            "WORKS CORRECTLY."
        )


if __name__ == "__main__":
    main()
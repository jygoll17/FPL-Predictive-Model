"""Check that XGBoost/LightGBM can load (OpenMP/libomp on macOS)."""

import sys


def check_ml_backends() -> None:
    """
    Ensure XGBoost can be loaded (requires OpenMP on macOS).
    Exit with a clear message if libomp is missing.
    """
    try:
        import xgboost  # noqa: F401
    except (OSError, Exception) as e:
        err = str(e).lower()
        if "libomp" in err or "openmp" in err or "could not be loaded" in err:
            print("Error: XGBoost/LightGBM need the OpenMP library on macOS.", file=sys.stderr)
            print("", file=sys.stderr)
            print("Install with:", file=sys.stderr)
            print("  brew install libomp", file=sys.stderr)
            print(
                "  export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH",
                file=sys.stderr,
            )
            print("", file=sys.stderr)
            print("If brew says directories are not writable, fix ownership first:", file=sys.stderr)
            print("  sudo chown -R $(whoami) /opt/homebrew", file=sys.stderr)
            sys.exit(1)
        raise

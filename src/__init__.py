"""Human Action & Behavior Monitor — core package."""
__version__ = "1.0.0"

# Loaded before any submodule on purpose: config sets process-wide
# environment (MPLBACKEND, TORCH_HOME) that third-party imports below read at
# import time — ultralytics and mediapipe both pull in matplotlib.
import config  # noqa: E402,F401

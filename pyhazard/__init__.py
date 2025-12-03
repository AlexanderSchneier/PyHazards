from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("PyHazard")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback
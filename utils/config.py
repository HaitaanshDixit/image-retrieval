"""Single entry point for loading configs/config.yaml from anywhere in the repo."""
import os
import logging
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "configs", "config.yaml")


def load_config(path: str = _DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Resolve every relative path in `paths:` against the repo root so
    # scripts work regardless of the current working directory.
    for key, rel_path in cfg["paths"].items():
        cfg["paths"][key] = os.path.join(_REPO_ROOT, rel_path)
    return cfg


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def repo_root() -> str:
    return _REPO_ROOT

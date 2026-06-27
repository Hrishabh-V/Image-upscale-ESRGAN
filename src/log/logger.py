from __future__ import annotations

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Optional, Union

ROOT_LOGGER_NAME = "upscale_esrgan"

_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# timestamp | level | service name | logger hierarchy | message
_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "service=%(service)s | %(name)s | %(message)s"
)

_configured = False
_config_lock = Lock()


class _AppFormatter(logging.Formatter):
    """
    Custom formatter with:
    - millisecond timestamps
    - safe default `service` field
    """

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "service"):
            record.service = "-"
        return super().format(record)

    def formatTime(
        self,
        record: logging.LogRecord,
        datefmt: Optional[str] = None,
    ) -> str:
        ct = self.converter(record.created)
        fmt = datefmt or self.datefmt or _DEFAULT_DATEFMT
        base = time.strftime(fmt, ct)
        return f"{base}.{int(record.msecs):03d}"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger under the application hierarchy.

    Example
    -------
    from log.logger import get_logger

    log = get_logger(__name__)
    """

    if name is None:
        return logging.getLogger(ROOT_LOGGER_NAME)

    if (
        name == ROOT_LOGGER_NAME
        or name.startswith(f"{ROOT_LOGGER_NAME}.")
    ):
        return logging.getLogger(name)

    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def get_service_logger(
    service_name: str,
    *,
    component: Optional[str] = None,
) -> logging.LoggerAdapter:
    """
    Return a logger adapter with service metadata.

    Example
    -------
    from log.logger import get_service_logger

    log = get_service_logger(
        "TensorInterpolationService",
        component="worker",
    )
    """

    if not isinstance(service_name, str):
        raise TypeError(
            f"service_name must be str, got {type(service_name).__name__}"
        )

    if not service_name.strip():
        raise ValueError(
            "service_name cannot be empty"
        )

    if component is not None and not isinstance(component, str):
        raise TypeError(
            f"component must be str or None, got {type(component).__name__}"
        )

    suffix = f".{component}" if component else ""

    logger = logging.getLogger(
        f"{ROOT_LOGGER_NAME}.service.{service_name}{suffix}"
    )

    return logging.LoggerAdapter(
        logger,
        {"service": service_name},
    )


def configure_logging(
    level: Optional[int] = None,
    *,
    log_file: Optional[Union[str, Path]] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    format_str: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
    force: bool = False,
    propagate: bool = False,
) -> logging.Logger:
    """
    Configure application logging.

    Should be called ONCE at application startup.

    Log level resolution:
    1. `level` argument
    2. UPSCALE_LOG_LEVEL environment variable
    3. INFO default
    """

    if log_file is not None and not isinstance(
        log_file,
        (str, Path),
    ):
        raise TypeError(
            "log_file must be str, Path, or None"
        )

    if max_bytes <= 0:
        raise ValueError(
            "max_bytes must be > 0"
        )

    if backup_count < 0:
        raise ValueError(
            "backup_count cannot be negative"
        )

    if not isinstance(format_str, str):
        raise TypeError(
            "format_str must be a string"
        )

    if not isinstance(datefmt, str):
        raise TypeError(
            "datefmt must be a string"
        )

    global _configured

    with _config_lock:

        root = logging.getLogger(ROOT_LOGGER_NAME)
        root.setLevel(_resolve_level(level))

        if force:
            for handler in list(root.handlers):
                root.removeHandler(handler)

                try:
                    handler.close()
                except Exception:
                    pass

            _configured = False

        if not _configured:

            formatter = _AppFormatter(
                format_str,
                datefmt,
            )

            # Console handler
            has_stream_handler = any(
                isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
                for h in root.handlers
            )

            if not has_stream_handler:
                stream_handler = logging.StreamHandler(sys.stderr)
                stream_handler.setLevel(logging.DEBUG)
                stream_handler.setFormatter(formatter)

                root.addHandler(stream_handler)

            # Rotating file handler
            if log_file is not None:

                log_path = Path(log_file)

                log_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                resolved_log_path = str(log_path.resolve())

                has_same_file_handler = any(
                    isinstance(h, RotatingFileHandler)
                    and getattr(h, "baseFilename", None)
                    == resolved_log_path
                    for h in root.handlers
                )

                if not has_same_file_handler:

                    try:
                        file_handler = RotatingFileHandler(
                            resolved_log_path,
                            maxBytes=max_bytes,
                            backupCount=backup_count,
                            encoding="utf-8",
                        )

                        file_handler.setLevel(logging.DEBUG)
                        file_handler.setFormatter(formatter)

                        root.addHandler(file_handler)

                    except OSError as e:
                        raise RuntimeError(
                            f"Failed to initialize log file handler: {str(e)}"
                        ) from e

            root.propagate = propagate

            _configured = True

    return root


def _resolve_level(level: Optional[int]) -> int:
    """
    Resolve effective log level.
    """

    if level is not None:
        return level

    env_level = (
        os.environ.get("UPSCALE_LOG_LEVEL", "")
        .strip()
        .upper()
    )

    if env_level:
        mapped = getattr(logging, env_level, None)

        if isinstance(mapped, int):
            return mapped

    return logging.INFO


def _ensure_null_handler() -> None:
    """
    Avoid 'No handler found' warnings if logging
    is used before configure_logging().
    """

    root = logging.getLogger(ROOT_LOGGER_NAME)

    if not root.handlers:
        root.addHandler(logging.NullHandler())


_ensure_null_handler()
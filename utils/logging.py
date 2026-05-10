"""Logging configuration, progress bars, and colored terminal output."""

import logging
import sys
from typing import Optional
from tqdm import tqdm


# ============================================================================
# Logger setup
# ============================================================================

def setup_logger(name: str,
                level: str = 'INFO',
                log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return a named logger with console (and optional file) output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# Progress bar
# ============================================================================

class ProgressBar:
    """Simple context-manager wrapper around tqdm."""

    def __init__(self,
                 total: int = None,
                 desc: str = None,
                 unit: str = 'it',
                 disable: bool = False):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.pbar = None

    def __enter__(self):
        self.pbar = tqdm(
            total=self.total,
            desc=self.desc,
            unit=self.unit,
            disable=self.disable
        )
        return self.pbar

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar is not None:
            self.pbar.close()


def create_progress_bar(iterable,
                       desc: str = None,
                       total: int = None,
                       disable: bool = False):
    """Wrap an iterable with a tqdm progress bar."""
    return tqdm(iterable, desc=desc, total=total, disable=disable)


# ============================================================================
# Colored output
# ============================================================================

class ColoredOutput:
    """Colored terminal output using ANSI escape sequences."""

    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def success(cls, text: str) -> str:
        return f"{cls.GREEN}{text}{cls.RESET}"

    @classmethod
    def error(cls, text: str) -> str:
        return f"{cls.RED}{text}{cls.RESET}"

    @classmethod
    def warning(cls, text: str) -> str:
        return f"{cls.YELLOW}{text}{cls.RESET}"

    @classmethod
    def info(cls, text: str) -> str:
        return f"{cls.BLUE}{text}{cls.RESET}"

    @classmethod
    def header(cls, text: str) -> str:
        return f"{cls.BOLD}{text}{cls.RESET}"


def print_success(text: str):
    print(ColoredOutput.success(text))


def print_error(text: str):
    print(ColoredOutput.error(text))


def print_warning(text: str):
    print(ColoredOutput.warning(text))


def print_info(text: str):
    print(ColoredOutput.info(text))

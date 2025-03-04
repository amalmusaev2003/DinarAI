# logging_config.py
import logging
import sys

def setup_logging(level=logging.INFO):
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=[console_handler]
    )

    return logging.getLogger("DinarAI")

logger = setup_logging()
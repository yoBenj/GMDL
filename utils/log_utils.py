# utils/log_utils.py

import logging
import sys

def set_logger(log_file):
    """Sets up the logger.

    Args:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler(sys.stdout)
                        ])

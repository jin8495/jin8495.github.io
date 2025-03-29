import argparse
import argcomplete

import os
import sys
import logging
from pathlib import Path

from .utils import *

from .Property import Property
from .Content import Content
from .ObsidianPage import ObsidianPage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("md2post.log")
    ]
)
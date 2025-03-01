import logging
import os
from enum import Enum
from dotenv import load_dotenv
from dataclasses import dataclass
from src.config.formatter import ColorFormatter

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Logger(logging.Logger):
    def __init__(self, level=logging.NOTSET):
        super().__init__("", level)
        self._setup_handlers()

    def _setup_handlers(self):
        fmt = '%(asctime)s | %(levelname)8s| %(message)s'
        color_formatter = ColorFormatter(fmt)
        stdout_handler = logging.StreamHandler()
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(color_formatter)
        self.addHandler(stdout_handler)
            
            
@dataclass
class DocSettings:
    TITLE: str
    DESCRIPTION: str
    VERSION: str
    SOURCE_DIR: str
    BUILD_DIR: str
    CACHE_DIR: str
    PLANTUML_JAR: str

@dataclass
class Settings:
    def __post_init__(self):
        load_dotenv()
        self.DOCS = DocSettings(
            TITLE=os.getenv("DOCS__TITLE"),
            DESCRIPTION=os.getenv("DOCS__DESCRIPTION"),
            VERSION=os.getenv("DOCS__VERSION"),
            SOURCE_DIR=os.getenv("DOCS__SOURCE_DIR"),
            BUILD_DIR=os.getenv("DOCS__BUILD_DIR"),
            CACHE_DIR=os.getenv("DOCS__CACHE_DIR"),
            PLANTUML_JAR=os.getenv("DOCS__PLANTUML_JAR"),
        )
        # set Log level using string value


        self.LOG_LEVEL = LogLevel(os.getenv("LOG_LEVEL", "INFO").upper())

    @property
    def logger(self):
        return Logger(level=self.LOG_LEVEL.value)
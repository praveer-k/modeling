from .settings import Settings

# Initialize settings
# -----------------------------------------------------------------------
settings = Settings()
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
logger = settings.logger
logger.setLevel(settings.LOG_LEVEL.value)
logger.info("Initialized Settings ...")
# Visible only when log level is set to DEBUG manually in logger.py file
logger.debug(settings.__dict__)
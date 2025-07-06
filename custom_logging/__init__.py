import logging
import logging.config


def init_logging(filename: str | None = None) -> None:
    if filename is not None:
        # Set 'disable_existing_loggers' to 'False' so
        # module-level loggers aren't disabled
        logging.config.fileConfig(filename, disable_existing_loggers=False)
    else:
        logging.warning(
            "LOGGING_CONFIG environment variable is not set, "
            "falling back to basic logging"
        )

        # Clear the handlers otherwise basicConfig will not work
        logging.getLogger().handlers = []

        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=logging.DEBUG,
        )

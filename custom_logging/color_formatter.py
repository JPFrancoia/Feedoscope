import json
import logging
from typing import Any, Mapping

# Taken from:
# https://github.com/MyColorfulDays/jsonformatter/blob/f7908f1b2bc9e556aea29f26307643e732ac8b5e/src/jsonformatter/jsonformatter.py#L89
_LogRecordDefaultAttributes = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
    "otelSpanID",
    "otelTraceID",
    "otelTraceSampled",
    "otelServiceName",
    "taskName",
}


class ColorFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"
    bold_green = "\x1b[32;1m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt: str, *args: Any, **kwargs: Mapping[str, Any]) -> None:
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.cyan + self.fmt + self.reset,
            logging.INFO: self.bold_green + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)

        extras = get_records_extra_attrs(record)

        if extras := get_records_extra_attrs(record):
            record.msg = f"{record.msg}. Extras: {json.dumps(extras)}"

        return formatter.format(record)


def get_records_extra_attrs(record: logging.LogRecord) -> Mapping[str, Any]:
    """Extract extra attributes from a log record.

    Largely inspired from:
    https://github.com/MyColorfulDays/jsonformatter/blob/master/src/jsonformatter/jsonformatter.py#L344

    Args:
        record: extract extras from this record

    Returns:
        extra dict passed to the logger, as a full dict.

    """
    extras = {
        k: record.__dict__[k]
        for k in record.__dict__
        if k not in _LogRecordDefaultAttributes
    }
    return extras

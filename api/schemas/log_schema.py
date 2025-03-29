import logging
import json
import sys
from pythonjsonlogger.json import JsonFormatter
from api.core import config
from rich.logging import RichHandler
from api.security.filters import RequestContextFilter

context_filter = RequestContextFilter()

class CustomJSONFormatter(JsonFormatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record=record, datefmt="%d/%m/%Y, %H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add request_id if available
        for attr in ["request_id", "ip", "user_agent", "path"]:
            value = getattr(record, attr, None)
            if value:
                log_entry[attr] = value

        if isinstance(record.exc_info, tuple):
            exception_str = self.formatException(record.exc_info)
            log_entry["exception"] = "\n".join(exception_str) if isinstance(exception_str, list) else exception_str

        if config.DEBUG:
            log_entry.update(
                {
                    "function": getattr(record, "funcName", "N/A"),
                    "filename": getattr(record, "filename", "N/A"),
                    "line": getattr(record, "lineno", "N/A"),
                }
            )

        return json.dumps(obj=log_entry, indent=1)

# Initialize Logger
logger = logging.getLogger(name="app_logs")
logger.setLevel(level=logging.DEBUG if config.DEBUG else logging.INFO)

# JSON Console Handler
json_handler = logging.StreamHandler(stream=sys.stdout)
json_handler.setFormatter(fmt=CustomJSONFormatter())

# File Handler (all logs)
file_handler = logging.FileHandler("api/logs/app.log", mode="a", encoding="utf-8")
file_handler.setFormatter(CustomJSONFormatter())

# error log file
error_handler = logging.FileHandler("api/logs/error.log", mode="a", encoding="utf-8")
error_handler.setLevel(logging.ERROR)  # Only log errors
error_handler.setFormatter(CustomJSONFormatter())

# rich handler
rich_handler = RichHandler(rich_tracebacks=True, markup=True) 
rich_handler.setFormatter(fmt=CustomJSONFormatter())

# add handlers
logger.addHandler(rich_handler)      # For dev preview
logger.addHandler(file_handler)      # All logs to file
logger.addHandler(error_handler)     # Error logs only

# add filters
json_handler.addFilter(context_filter)
file_handler.addFilter(context_filter)
error_handler.addFilter(context_filter)
rich_handler.addFilter(context_filter)

logger.propagate = False
logging.getLogger("uvicorn.access").propagate = True
from dotenv import load_dotenv
import os
from urllib.parse import urlparse

ENVIRONMENT = os.getenv(key="ENV", default="development")

BASE_DIR = os.path.dirname(p=os.path.dirname(p=os.path.dirname(p=os.path.abspath(path=__file__))))
if ENVIRONMENT == "production":
    load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env.production"))
else:
    load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env.development"))

def env_variables(key) -> str:
    value: str | None = os.getenv(key=key)
    if value and value.startswith("http"):
        return urlparse(url=value).path
    return str(object=value)

LENGTH = int(env_variables(key="LENGTH"))
DEBUG = env_variables(key="DEBUG").strip().lower() in ("true", "1", "yes")
APP_NAME = env_variables(key="APP_NAME")
APP_VERSION = env_variables(key="APP_VERSION")
MODEL_PATH = env_variables(key="MODEL_PATH_DETECT_LANG")
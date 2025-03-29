from api.handlers.error_handlers import create_error_handlers
from api.middlewares.custom_middleware import add_x_request_id, setup_cors_middleware
from api.logging.logger import init_logger

import uvicorn
from fastapi import FastAPI

init_logger(message="Starting FastAPI Application")

app = FastAPI()

setup_cors_middleware(app=app)

app.middleware("http")(add_x_request_id)

create_error_handlers(app=app)

@app.get(path="/")
async def root() -> dict:
    return {"status": "Success"}


def initializeAPP():
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True, log_level="debug")

if __name__ == "__main__":
    initializeAPP()
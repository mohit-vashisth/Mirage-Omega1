from fastapi import FastAPI
from api.core.config import DEBUG
from api.services.translate_service import translate_route

def include_routers(app: FastAPI):
    routes: list[tuple] = [
        (translate_route, ["services"])
    ]

    for route, tag in routes:
        if DEBUG:
            app.include_router(router=route, tags=tag)
        app.include_router(router=route, tags=tag)

    app.include_router(translate_route)
            

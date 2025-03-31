from httpx import AsyncClient
from hypothesis import given, strategies as st
from api.main import app
from api.core.config import TRANSLATE_EP

import pytest

@pytest.mark.asyncio
@given()

async def test_translate(text, dest, status_code):
    async with AsyncClient(base_url="http://localhost:8000") as ac:
        request_data = {"text": text, "dest": dest}
        response = await ac.post(TRANSLATE_EP, json=request_data)
        assert response.status_code == status_code, f"Unexpected status: {response.status_code}, body: {response.text}"

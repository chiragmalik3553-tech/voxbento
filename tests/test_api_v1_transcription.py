"""
Tests for the /api/v1 transcription control endpoints.

Covers:
- Booth ID is built from event slug, room and language
- Worker receives the booth's configured provider, model and key
- Booth missing from the in-memory registry
- Transcription disabled on the booth
- Stop targets the room-scoped booth
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from portal.globals import booths
from portal.routers import api_v1

EVENT_SLUG = "pycon2026"
ROOM_ID = 14


def make_event(**overrides):
    """Build a stand-in Event row, with any field overridden by keyword."""
    event = SimpleNamespace(id=1, slug=EVENT_SLUG, transcription_api_enabled=False)
    event.__dict__.update(overrides)
    return event


def make_booth(**overrides):
    """Build a stand-in DBBooth row, with any field overridden by keyword."""
    booth = SimpleNamespace(transcription_enabled=True, transcription_provider="local", transcription_model="tiny")
    booth.__dict__.update(overrides)
    return booth


class FakeSession:
    """Stands in for AsyncSession: hands back queued rows in query order."""

    def __init__(self, *rows):
        """Queue the rows to hand back, one per query."""
        self._rows = list(rows)

    async def execute(self, _stmt):
        """Return the next queued row wrapped the way scalars().first() expects."""
        row = self._rows.pop(0) if self._rows else None
        return SimpleNamespace(scalars=lambda: SimpleNamespace(first=lambda: row))

    async def scalar(self, _stmt):
        """Return the next queued row directly."""
        return self._rows.pop(0) if self._rows else None


@pytest.fixture(autouse=True)
def skip_rbac(monkeypatch):
    """The OAuth/RBAC gate is covered elsewhere; these tests exercise the body."""

    async def allow(*_args, **_kwargs):
        """Let every caller through."""
        return None

    monkeypatch.setattr(api_v1, "_verify_token_rbac", allow)


@pytest.fixture
async def live_booth():
    """Register the booth in the in-memory registry for the duration of a test."""
    await booths.create_booth(EVENT_SLUG, "en", "English", ROOM_ID)
    yield
    await booths.remove_booth(EVENT_SLUG, ROOM_ID, "en")


@pytest.fixture
def token():
    """Return an OAuth token scoped to the seeded event."""
    return SimpleNamespace(event_id=1, user_id=1)


@pytest.mark.anyio
async def test_start_passes_room_scoped_booth_and_booth_settings(monkeypatch, live_booth, token):
    """Start builds a room-scoped booth ID and forwards the booth's provider and model."""
    started = {}

    async def fake_worker(*args, **kwargs):
        """Record the arguments the endpoint passes to the worker."""
        started["args"] = args
        started["kwargs"] = kwargs

    monkeypatch.setattr(api_v1, "start_transcription_worker", fake_worker)

    result = await api_v1.start_transcription(
        event_slug=EVENT_SLUG,
        room_id=ROOM_ID,
        language_code="en",
        db=FakeSession(make_event(), make_booth()),
        token=token,
    )

    assert result == {"status": "started", "booth_id": "pycon2026-14-en"}
    assert started["args"][:3] == (EVENT_SLUG, "en", "pycon2026-14-en")
    assert started["args"][4:6] == ("local", "tiny")
    assert started["kwargs"] == {"room_id": ROOM_ID}


@pytest.mark.anyio
async def test_start_rejects_booth_missing_from_registry(token):
    """Start returns 400 when the booth is not live in the registry."""
    with pytest.raises(HTTPException) as exc:
        await api_v1.start_transcription(
            event_slug=EVENT_SLUG,
            room_id=99,
            language_code="de",
            db=FakeSession(make_event()),
            token=token,
        )

    assert exc.value.status_code == 400
    assert "not active" in exc.value.detail


@pytest.mark.anyio
async def test_start_rejects_booth_with_transcription_disabled(live_booth, token):
    """Start returns 400 when the booth has transcription switched off."""
    with pytest.raises(HTTPException) as exc:
        await api_v1.start_transcription(
            event_slug=EVENT_SLUG,
            room_id=ROOM_ID,
            language_code="en",
            db=FakeSession(make_event(), make_booth(transcription_enabled=False)),
            token=token,
        )

    assert exc.value.status_code == 400


@pytest.mark.anyio
async def test_stop_targets_room_scoped_booth(monkeypatch, token):
    """Stop hands the worker the room-scoped booth ID."""
    stopped = []

    async def fake_stop(booth_id):
        """Record the booth ID the endpoint asks to stop."""
        stopped.append(booth_id)

    monkeypatch.setattr(api_v1, "stop_transcription_worker", fake_stop)

    result = await api_v1.stop_transcription(
        event_slug=EVENT_SLUG,
        room_id=ROOM_ID,
        language_code="en",
        db=FakeSession(make_event()),
        token=token,
    )

    assert stopped == ["pycon2026-14-en"]
    assert result == {"status": "stopped", "booth_id": "pycon2026-14-en"}

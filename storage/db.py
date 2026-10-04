from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from storage.models import Base


def make_engine(database_url: str) -> Engine:
    return create_engine(database_url, future=True)


# libpq parameters that carry a secret when a URL passes them in its query.
_SECRET_QUERY_KEYS = ("password", "sslpassword")


def redacted_url(database_url: str) -> str:
    """The URL for a log line: its password masked, whether in the userinfo
    or the query (render_as_string only masks the userinfo)."""
    url = make_url(database_url)
    masked = {key: "redacted" for key in _SECRET_QUERY_KEYS if key in url.query}
    return url.update_query_dict(masked).render_as_string(hide_password=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)

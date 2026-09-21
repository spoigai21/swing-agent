"""Alembic environment. URL comes from settings, never from alembic.ini."""
from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from swing.common.settings import get_settings
from swing.store.models import Base

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database_url)
target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """schema.sql owns indexes and CHECK constraints; models own tables/columns.

    Without this, autogenerate sees every index and CHECK that schema.sql
    created but the ORM does not declare, and emits DROP statements for all of
    them — including the hnsw vector index, which is expensive to rebuild and
    silently degrades retrieval if it goes missing.

    Consequence to know: alembic will NOT manage indexes or CHECK constraints.
    Add those to schema.sql and, if an existing database needs them, write the
    migration by hand.
    """
    if type_ in ("index", "unique_constraint"):
        return False
    return not (reflected and type_ == "table_constraint")


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

"""Importing this package registers every skill (order = catalog order)."""
from skills import basic, tools, motion, dance_skill  # noqa: F401
from skills.registry import REGISTRY, catalog_text, plan_schema, validate_plan  # noqa: F401

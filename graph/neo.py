import os
from typing import Any
from config.config import neomodel_config
from s_types import s_errors
from neomodel import db as neomodel_db
from utils.logging_config import get_logger,setup_logging

setup_logging()
logger = get_logger(__name__)

_DEFAULT_PLACEHOLDER_URL = 'bolt://neo4j:foobarbaz@localhost:7687'

class Neo4jDB(object):

    def __init__(self):
        if getattr(neomodel_config, 'DATABASE_URL', None) == _DEFAULT_PLACEHOLDER_URL:
            raise s_errors.DatabaseNotConfiguredError(
                "Neo4j connection not configured. "
                "Please call Neo4jConfig(...) before instantiating Neo4jDB."
            )
        print(_DEFAULT_PLACEHOLDER_URL)
        print(neomodel_config.DATABASE_URL)
        logger.debug("Neo4jDB initialized successfully.")

    @staticmethod
    def is_connected() -> bool:
        try:
            driver = neomodel_db.driver
            return driver is not None
        except Exception:
            return False

    @staticmethod
    def healthy_check(timeout: float = 5.0) -> bool:

        try:
            result, _ = neomodel_db.cypher_query("RETURN 1 AS alive", {}, resolve_objects=False)
            if result and len(result) == 1 and result[0][0] == 1:
                logger.debug("Neo4j health check passed.")
                return True
            else:
                logger.warning("Neo4j health check returned unexpected result.")
                return False
        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}", exc_info=True)
            return False

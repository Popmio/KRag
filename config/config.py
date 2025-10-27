from neomodel import config as neomodel_config
from s_types import s_errors
from typing import Any, ClassVar, Optional
import threading
import os

__all__ = ["Neo4jConfig"]


class Neo4jConfig(object):
    _instance: ClassVar[Optional['Neo4jConfig']] = None
    _initialized: ClassVar[bool] = False
    _lock: ClassVar = threading.Lock()
    _PROTECTED = frozenset({"DATABASE_URL"})

    def __new__(cls, database_url: str, **kwargs: Any):
        if cls._instance is not None:
            raise s_errors.AlreadyConfiguredError(
                "Neo4jConfig has already been instantiated. "
                "Global Neo4j configuration can only be set once."
            )
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            else:
                raise s_errors.AlreadyConfiguredError(...)
        return cls._instance

    def __init__(self, database_url: str, **kwargs: Any) -> None:
        if self._initialized:
            return
        object.__setattr__(self, "DATABASE_URL", database_url)
        neomodel_config.DATABASE_URL = database_url
        self.update_config(**kwargs)
        self.__class__._initialized = True

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if key in self._PROTECTED:
                raise s_errors.ProtectedError(f"The key {key} is already protected")
            self._set_config(key, value)

    def _set_config(self, key: str, value: Any) -> None:

        if not hasattr(neomodel_config, key):
            raise s_errors.UnexpectedConfigParam(key)
        try:
            setattr(neomodel_config, key, value)
            object.__setattr__(self, key, value)
        except Exception as e:
            raise s_errors.ConfigError(f"Failed to set {key}={value!r}: {e}") from e

    def to_dict(self, hide_protected: bool = True) -> dict:
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if hide_protected and k in self._PROTECTED:
                result[k] = "***"
            else:
                result[k] = v
        return result

    @staticmethod
    def get_available_params():
        params = []
        for attr in dir(neomodel_config):
            if attr.startswith('_'):
                continue
            try:
                val = getattr(neomodel_config, attr)
                if isinstance(val, (str, int, bool, float)) and attr != 'DATABASE_URL':
                    params.append(attr)
            except Exception:
                continue
        return params

    @classmethod
    def from_env(cls, url_env: str = "NEO4J_DATABASE_URL", **extra_kwargs) -> "Neo4jConfig":

        database_url = os.getenv(url_env)
        if not database_url:
            raise s_errors.ConfigError(f"Environment variable {url_env} not set")

        available = cls.get_available_params()
        env_config = {}
        for param in available:
            env_key = f"NEOMODEL_{param.upper()}"
            val = os.getenv(env_key)
            if val is not None:
                env_config[param] = cls._parse_env_value(val)

        env_config.update(extra_kwargs)
        return cls(database_url, **env_config)

    @staticmethod
    def _parse_env_value(val: str) -> Any:
        if val.lower() in ("true", "1"):
            return True
        if val.lower() in ("false", "0"):
            return False
        if val.isdigit():
            return int(val)
        try:
            return float(val)
        except ValueError:
            pass
        return val

    def __setattr__(self, key, value):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self._set_config(key, value)

    def __repr__(self):
        hidden_keys = self._PROTECTED
        attrs = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if k in hidden_keys:
                attrs[k] = "***"
            else:
                attrs[k] = v
        items = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
        return f"{self.__class__.__name__}({items})"
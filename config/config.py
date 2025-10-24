from neomodel import config as neomodel_config
from pydantic import BaseModel
from s_types import s_errors


class Neo4jConfig(object):
    _PROTECTED = {"DATABASE_URL"}
    def __init__(self, database_url,**kwargs):
        object.__setattr__(self, "DATABASE_URL", database_url)
        neomodel_config.DATABASE_URL = database_url

        self.update_config(**kwargs)

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if key in self._PROTECTED:
                raise s_errors.ProtectedError(f"The key {key} is already protected")
            self._set_config(key, value)

    def _set_config(self, key: str, value):

        if not hasattr(neomodel_config, key):
            raise s_errors.UnexpectedConfigParam(key)
        try:
            setattr(neomodel_config, key, value)
            object.__setattr__(self, key, value)
        except ValueError as e:
            raise e
        except Exception as e:
            raise e
    @staticmethod
    def get_available_params():
        return dir(neomodel_config)

    def __setattr__(self, key, value):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self._set_config(key, value)

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"{self.__class__.__name__}(DATABASE_URL='***', {', '.join(f'{k}={v!r}' for k, v in attrs.items())})"
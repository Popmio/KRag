





class ProtectedError(Exception):
    def __init__(self, message = "The Value is Protected"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return repr(self.message)

class UnexpectedConfigParam(Exception):
    def __init__(self, key):
        self.key = key
        super().__init__(self.key)

    def __str__(self):
        return repr(f'{self.key} is not a valid config parameter,please use function get_available_params to confirm params')
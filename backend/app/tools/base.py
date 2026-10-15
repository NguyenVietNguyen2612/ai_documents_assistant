try:
    from langchain_core.tools import tool
except ImportError:
    def tool(fn):
        class _ToolWrapper:
            def __init__(self, func):
                self.func = func
                self.name = func.__name__
                self.__doc__ = func.__doc__

            def __call__(self, *args, **kwargs):
                return self.func(*args, **kwargs)

            def invoke(self, input_data):
                if isinstance(input_data, dict):
                    return self.func(**input_data)
                return self.func(input_data)

        return _ToolWrapper(fn)

class Metric:
    def add(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        result = self.add(*args, **kwargs)

        return result

    def summary(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name()}: {self.summary()}"

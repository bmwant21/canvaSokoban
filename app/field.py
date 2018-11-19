import math


class Position(object):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def forward(self):
        cls = self.__class__
        return cls(x=self.x, y=self.y+1)

    def backward(self):
        cls = self.__class__
        return cls(x=self.x, y=self.y-1)

    def left(self):
        cls = self.__class__
        return cls(x=self.x-1, y=self.y)

    def right(self):
        cls = self.__class__
        return cls(x=self.x+1, y=self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.x, self.y))

    def steps_to(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)


class Field(object):
    def __init__(self):
        self._field = []

    @property
    def lenx(self):
        return len(self._field[0])

    @property
    def leny(self):
        return len(self._field)

    def __getitem__(self, item):
        return self._field.__getitem__(item)


import math
import random
from enum import IntEnum


class Cell(IntEnum):
    WALL = -1


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

    def __add__(self, other):
        cls = self.__class__
        return cls(self.x+other.x, self.y+other.y)

    def steps_to(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)


DIRECTIONS = {
    'left':  Position(x=-1, y= 0),
    'right': Position(x= 1, y= 0),
    'up':    Position(x= 0, y= 1),
    'down':  Position(x= 0, y=-1),
}


class _Col(object):
    def __init__(self, col):
        self.col = col

    def __getitem__(self, row_index):
        return self.col[row_index]


class Field(object):
    def __init__(self, xsize, ysize, max_value):
        self._field = [
            [0 for _ in range(xsize)]
            for _ in range(ysize)
        ]
        self.max_value = max_value

    def fill(self):
        for y_index in range(self.leny):
            for x_index in range(self.lenx):
                if self._is_border(x_index, y_index):
                    value = Cell.WALL.value
                else:
                    value = random.randint(1, self.max_value)
                self._field[y_index][x_index] = value

    def _is_border(self, x, y):
        return (x == 0) or (y == 0) or (x == self.lenx-1) or (y == self.leny-1)

    @property
    def lenx(self):
        return len(self._field[0])

    @property
    def leny(self):
        return len(self._field)

    def __getitem__(self, row_index):
        """
        Swapped
        col = [self._field[row_index][col_index]
               for row_index in range(self.leny)]
        return _Col(col=col)
        """
        return self._field[row_index]

    def __str__(self):
        return '\n'.join([
            ''.join([
                str(self._field[y][x]) for x in range(self.lenx)
            ]) for y in range(self.leny)
        ])


if __name__ == '__main__':
    f = Field(4, 5, max_value=9)
    print(f.lenx, f.leny)
    f.fill()
    print(f._field)

    p1 = Position(x=1, y=2)
    p2 = Position(x=3, y=4)
    p1 += p2
    print(p1)

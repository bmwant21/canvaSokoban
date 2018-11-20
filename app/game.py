import math
import random

from app.field import Position, Field, Cell, DIRECTIONS
from app.utils import logger


class State(object):
    def __init__(
        self,
        score,
        current_position,
        finish_position,
        failed: bool,
        distance_x: int,
        distance_y: int,
        distance: int,
        moves_left: int,
        barrier_up: int,
        barrier_down: int,
        barrier_left: int,
        barrier_right: int,
        value_up: int,
        value_down: int,
        value_left: int,
        value_right: int,
    ):
        self.score = score
        self.current_position = current_position
        self.finish_position = finish_position
        self.failed = failed
        # features
        self.distance_x = distance_x
        self.distance_y = distance_y
        self.distance = distance
        self.moves_left = moves_left
        self.barrier_up = barrier_up
        self.barrier_down = barrier_down
        self.barrier_left = barrier_left
        self.barrier_right = barrier_right
        self.value_up = value_up
        self.value_down = value_down
        self.value_left = value_left
        self.value_right = value_right


class Game(object):
    DEFAULT_SIZE_X = 21
    DEFAULT_SIZE_Y = 21

    def __init__(
        self,
        field: Field,
        start: Position,
        end: Position,
        moves_left: int,
    ):
        self.field = field
        self._start = start
        self._end = end
        self._score = 0
        self._pos = None
        self._moves_left = moves_left

    def start(self):
        self._pos = self._start
        return self.step(self._start)

    def step(self, position: Position):
        if self.can_move(position):
            value = self.field[position.y][position.x]
            self._score += value
            self._pos = position
            self._moves_left -= 1

        distance_x = abs(self._end.x - self._pos.x)
        distance_y = abs(self._end.y - self._pos.y)
        distance = self._pos.steps_to(self._end)
        assert distance_x + distance_y == distance
        state = State(
            score=self._score,
            current_position=self._pos,
            finish_position=self._end,
            failed=self.failed,
            distance_x=distance_x,
            distance_y=distance_y,
            distance=distance,
            moves_left=self._moves_left,
            barrier_up=self._get_barrier_distance('up'),
            barrier_down=self._get_barrier_distance('down'),
            barrier_left=self._get_barrier_distance('left'),
            barrier_right=self._get_barrier_distance('right'),
            value_up=self._get_value('up'),
            value_down=self._get_value('down'),
            value_left=self._get_value('left'),
            value_right=self._get_value('right'),
        )
        return state

    def _get_value(self, direction):
        pos_shift = DIRECTIONS[direction]
        new_pos = self._pos + pos_shift
        if self.field[new_pos.y][new_pos.x] == Cell.WALL.value:
            return 0
        return self.field[new_pos.y][new_pos.x]

    def _get_barrier_distance(self, direction):
        pos_shift = DIRECTIONS[direction]
        steps = 0
        new_pos = self._pos
        while True:
            new_pos += pos_shift
            if not self.can_move(new_pos):
                return steps
            steps += 1

    def can_move(self, position: Position) -> bool:
        return self.field[position.y][position.x] != Cell.WALL.value

    @property
    def failed(self) -> bool:
        return self._pos.steps_to(self._end) > self._moves_left

    @property
    def max_steps(self):
        # Go through all available cells besides border
        return (self.field.lenx-2) * (self.field.leny-2)

    @classmethod
    def create_game(cls):
        xsize = cls.DEFAULT_SIZE_X
        ysize = cls.DEFAULT_SIZE_Y

        field = Field(
            xsize=xsize,
            ysize=ysize,
            max_value=9,
        )
        field.fill()
        # somewhere in III quadrant
        start_x = random.randint(1, xsize // 2)
        start_y = random.randint(ysize // 2, ysize-2)
        start = Position(x=start_x, y=start_y)
        # somewhere in I quadrant
        end_x = random.randint(xsize // 2, xsize-2)
        end_y = random.randint(1, ysize // 2)
        end = Position(x=end_x, y=end_y)
        steps = start.steps_to(end) + 4
        return cls(
            field=field,
            start=start,
            end=end,
            moves_left=steps,
        )

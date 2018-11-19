from app.field import Position, Field


class State(object):
    def __init__(
        self,
        score,
        current_position,
        finish_position,
        failed: bool,
    ):
        self.score = score
        self.current_position = current_position
        self.finish_position = finish_position
        self.failed = failed


class Game(object):
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

        state = State(
            score=self._score,
            current_position=self._pos,
            finish_position=self._end,
            failed=self.failed,
        )
        return state

    def can_move(self, position: Position) -> bool:
        return True

    @property
    def failed(self) -> bool:
        return self._pos.steps_to(self._end) > self._moves_left

    @property
    def max_steps(self):
        # Go through all available cells besides border
        return (self.field.lenx-2) * (self.field.leny-2)

    @classmethod
    def create_game(cls):
        pass

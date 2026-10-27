"""Board dimensions and published heuristic constants."""

COLUMNS = tuple(range(2, 13))
LENGTHS = (3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3)
LENGTH = dict(zip(COLUMNS, LENGTHS))
MARK_VALUE = dict(zip(COLUMNS, (12, 10, 8, 6, 4, 2, 4, 6, 8, 10, 12)))
STEP_VALUE = dict(zip(COLUMNS, (6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6)))
GA_MOVE_WEIGHT = dict(zip(COLUMNS, (7, 0, 2, 0, 4, 3, 4, 0, 2, 0, 7)))
CLASSIC_MOVE_WEIGHT = {c: 6 - abs(c - 7) for c in COLUMNS}

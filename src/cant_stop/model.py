"""Immutable values exposed by the game engine."""
from __future__ import annotations

from dataclasses import dataclass

from .constants import COLUMNS, LENGTH, LENGTHS


@dataclass(frozen=True, order=True)
class Move:
    columns: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.columns) <= 2:
            raise ValueError("A move must advance one or two steps")
        if tuple(sorted(self.columns)) != self.columns:
            raise ValueError("Move columns must be sorted")


@dataclass(frozen=True)
class State:
    """Immutable snapshot, from the perspective of current_player."""
    permanent: tuple[tuple[int, ...], tuple[int, ...]]
    owner: tuple[int, ...]  # -1 unclaimed, otherwise player index 0 or 1
    active: tuple[tuple[int, int], ...]  # (column, absolute runner position)
    current_player: int
    rolls_this_turn: int
    # Columns first opened during this turn.  This is needed by the published
    # Carter encoder, whose Rule-of-28 implementation distinguishes an initial
    # marker from later advances.
    columns_started: tuple[int, ...] = ()

    def active_dict(self) -> dict[int, int]:
        return dict(self.active)

    def completed(self, player: int) -> int:
        return self.owner.count(player)

    def features(self, enhanced: bool = False, domain: bool = False,
                 strategic: bool = False) -> list[float]:
        """58 raw, 60 enhanced, 68 domain, or 80 strategic features.

        Layout: me(11), opponent(11), absolute temp(11), active mask(11),
        relative ownership(11), completed counts(2), successful roll count(1).

        Domain mode includes the two enhanced values followed by aggregates that
        make the immediate bank/risk tradeoff explicit. It is opt-in so old
        58/60-input checkpoints remain loadable. Strategic mode adds per-column
        unbanked progress and an explicit opponent threat count; it implies all
        domain features and remains opt-in for checkpoint compatibility.
        """
        p = self.current_player
        me = self.permanent[p]
        opp = self.permanent[1 - p]
        active = self.active_dict()
        features = [me[i] / LENGTHS[i] for i in range(11)]
        features += [opp[i] / LENGTHS[i] for i in range(11)]
        features += [active.get(c, 0) / LENGTH[c] for c in COLUMNS]
        features += [float(c in active) for c in COLUMNS]
        features += [1.0 if x == p else (-1.0 if x == 1 - p else 0.0)
                     for x in self.owner]
        features += [self.completed(p) / 3.0, self.completed(1-p) / 3.0,
                     min(self.rolls_this_turn / 20.0, 1.0)]
        assert len(features) == 58
        if enhanced or domain or strategic:
            # Local import avoids a model/rules import cycle.
            from .rules import bust_probability, rule28_points
            bust = bust_probability(self)
            at_risk = sum(
                (pos - me[c-2]) / LENGTH[c] for c, pos in active.items()
            )
            features += [bust, at_risk]
            if domain or strategic:
                reached = sum(pos == LENGTH[c] for c, pos in active.items())
                completed_after_stop = self.completed(p) + reached
                my_progress = sum(me[i] / LENGTHS[i] for i in range(11)) / 11.0
                opp_progress = sum(opp[i] / LENGTHS[i] for i in range(11)) / 11.0
                # Eight bounded or naturally small decision-oriented values.
                features += [
                    len(active) / 3.0,
                    reached / 3.0,
                    min(completed_after_stop, 3) / 3.0,
                    float(completed_after_stop >= 3),
                    my_progress - opp_progress,
                    at_risk * (1.0 - bust),
                    min(rule28_points(self) / 40.0, 1.0),
                    float(self.completed(1 - p) >= 2),
                ]
                if strategic:
                    features += [
                        max(0, active.get(c, me[c - 2]) - me[c - 2]) / LENGTH[c]
                        for c in COLUMNS
                    ]
                    opponent_near_top = sum(
                        self.owner[i] == -1 and LENGTHS[i] - opp[i] <= 1
                        for i in range(11)
                    )
                    features.append(opponent_near_top / 3.0)
        expected = 80 if strategic else (68 if domain else (60 if enhanced else 58))
        assert len(features) == expected
        return features

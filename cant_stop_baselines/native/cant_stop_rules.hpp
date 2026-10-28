#pragma once

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace cant_stop_native {

constexpr std::array<int, 11> kLengths{3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3};

struct Move {
    std::array<int, 2> columns{};
    int size = 0;

    bool operator<(const Move& other) const {
        if (columns[0] != other.columns[0]) return columns[0] < other.columns[0];
        if (size != other.size) return size < other.size;
        return size == 2 && columns[1] < other.columns[1];
    }

    bool operator==(const Move& other) const {
        return size == other.size && columns == other.columns;
    }
};

struct State {
    std::array<int, 11> permanent{};
    std::array<int, 11> owner{};
    // -1 means no active runner; otherwise this is the absolute position.
    std::array<int, 11> active{};

    State() { active.fill(-1); }
};

inline bool can_step(int column, const State& state, int active_count) {
    const int index = column - 2;
    if (state.owner[index] != -1) return false;
    if (state.active[index] != -1) return state.active[index] < kLengths[index];
    return active_count < 3 && state.permanent[index] < kLengths[index];
}

inline std::vector<Move> legal_moves(const std::array<int, 4>& dice,
                                     const State& state) {
    const std::array<std::array<int, 2>, 3> partitions{{
        {{dice[0] + dice[1], dice[2] + dice[3]}},
        {{dice[0] + dice[2], dice[1] + dice[3]}},
        {{dice[0] + dice[3], dice[1] + dice[2]}},
    }};
    int active_count = 0;
    for (int position : state.active) active_count += position != -1;

    std::vector<Move> choices;
    const auto add_choice = [&choices](const Move& move) {
        if (std::find(choices.begin(), choices.end(), move) == choices.end()) {
            choices.push_back(move);
        }
    };
    for (const auto& partition : partitions) {
        const int x = partition[0];
        const int y = partition[1];
        if (x == y) {
            if (!can_step(x, state, active_count)) continue;
            State trial = state;
            const int index = x - 2;
            trial.active[index] = (trial.active[index] == -1)
                ? trial.permanent[index] + 1 : trial.active[index] + 1;
            if (can_step(x, trial, active_count)) {
                add_choice(Move{{x, x}, 2});
            } else {
                add_choice(Move{{x, 0}, 1});
            }
            continue;
        }

        const bool can_x = can_step(x, state, active_count);
        const bool can_y = can_step(y, state, active_count);
        if (can_x && can_y) {
            const int new_needed = (state.active[x - 2] == -1)
                + (state.active[y - 2] == -1);
            if (active_count + new_needed <= 3) {
                const int low = std::min(x, y);
                const int high = std::max(x, y);
                add_choice(Move{{low, high}, 2});
            } else {
                add_choice(Move{{x, 0}, 1});
                add_choice(Move{{y, 0}, 1});
            }
        } else if (can_x) {
            add_choice(Move{{x, 0}, 1});
        } else if (can_y) {
            add_choice(Move{{y, 0}, 1});
        }
    }
    return choices;
}

inline State advance_active(const State& state, const Move& move) {
    State next = state;
    for (int i = 0; i < move.size; ++i) {
        const int index = move.columns[i] - 2;
        next.active[index] = (next.active[index] == -1)
            ? next.permanent[index] + 1 : next.active[index] + 1;
    }
    return next;
}

enum class Phase { Roll = 0, Pair = 1, Decision = 2, Done = 3 };

struct Game {
    std::array<std::array<int, 11>, 2> permanent{};
    std::array<int, 11> owner{};
    std::array<int, 11> active{};
    int current_player = 0;
    int rolls_this_turn = 0;
    Phase phase = Phase::Roll;
    int winner = -1;
    std::vector<Move> available;
    std::array<int, 2> turns{};
    std::array<int, 2> rolls{};
    std::array<int, 2> busts{};
    std::array<int, 2> stops{};

    Game() { owner.fill(-1); active.fill(-1); }

    State state() const {
        State snapshot;
        snapshot.permanent = permanent[current_player];
        snapshot.owner = owner;
        snapshot.active = active;
        return snapshot;
    }

    void end_turn() {
        active.fill(-1);
        rolls_this_turn = 0;
        current_player = 1 - current_player;
        phase = Phase::Roll;
        available.clear();
    }

    std::vector<Move> roll(const std::array<int, 4>& dice) {
        if (phase != Phase::Roll || winner != -1) return {};
        const int player = current_player;
        if (rolls_this_turn == 0) ++turns[player];
        std::vector<Move> choices = legal_moves(dice, state());
        ++rolls[player];
        if (choices.empty()) {
            ++busts[player];
            end_turn();
            return {};
        }
        available = choices;
        phase = Phase::Pair;
        return choices;
    }

    bool apply_move(const Move& move) {
        if (phase != Phase::Pair ||
            std::find(available.begin(), available.end(), move) == available.end()) {
            return false;
        }
        State next = state();
        next = advance_active(next, move);
        active = next.active;
        ++rolls_this_turn;
        available.clear();
        phase = Phase::Decision;
        return true;
    }

    bool continue_turn() {
        if (phase != Phase::Decision) return false;
        phase = Phase::Roll;
        return true;
    }

    bool stop() {
        if (phase != Phase::Decision) return false;
        const int player = current_player;
        for (int index = 0; index < 11; ++index) {
            if (active[index] == -1) continue;
            permanent[player][index] = active[index];
            if (active[index] == kLengths[index]) owner[index] = player;
        }
        ++stops[player];
        int completed = 0;
        for (int value : owner) completed += value == player;
        if (completed >= 3) {
            winner = player;
            phase = Phase::Done;
            active.fill(-1);
            available.clear();
        } else {
            end_turn();
        }
        return true;
    }
};

}  // namespace cant_stop_native

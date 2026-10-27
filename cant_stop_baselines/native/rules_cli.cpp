#include "cant_stop_rules.hpp"

#include <iostream>

using cant_stop_native::State;

int main() {
    State state;
    while (true) {
        std::array<int, 4> dice{};
        if (!(std::cin >> dice[0] >> dice[1] >> dice[2] >> dice[3])) return 0;
        for (int& value : state.permanent) std::cin >> value;
        for (int& value : state.owner) std::cin >> value;
        state.active.fill(-1);
        int active_count = 0;
        std::cin >> active_count;
        for (int i = 0; i < active_count; ++i) {
            int column = 0;
            int position = 0;
            std::cin >> column >> position;
            state.active[column - 2] = position;
        }

        const auto moves = cant_stop_native::legal_moves(dice, state);
        std::cout << moves.size();
        for (const auto& move : moves) {
            std::cout << ' ' << move.size;
            for (int i = 0; i < move.size; ++i) std::cout << ' ' << move.columns[i];
        }
        std::cout << '\n';
    }
}

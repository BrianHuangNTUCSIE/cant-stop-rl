#include "cant_stop_rules.hpp"

#include <iostream>

int main() {
    using cant_stop_native::State;
    State state;
    while (true) {
        // The dice are accepted to keep this fixture compatible with the
        // legal-move CLI; transition itself does not inspect them.
        int ignored_dice[4]{};
        if (!(std::cin >> ignored_dice[0] >> ignored_dice[1]
              >> ignored_dice[2] >> ignored_dice[3])) return 0;
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
        int move_size = 0;
        std::cin >> move_size;
        cant_stop_native::Move move;
        move.size = move_size;
        for (int i = 0; i < move_size; ++i) std::cin >> move.columns[i];

        const State next = cant_stop_native::advance_active(state, move);
        int count = 0;
        for (int position : next.active) count += position != -1;
        std::cout << count;
        for (int index = 0; index < 11; ++index) {
            if (next.active[index] != -1) {
                std::cout << ' ' << index + 2 << ' ' << next.active[index];
            }
        }
        std::cout << '\n';
    }
}

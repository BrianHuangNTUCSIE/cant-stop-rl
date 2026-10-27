#include "cant_stop_rules.hpp"

#include <iostream>

int main() {
    using cant_stop_native::Game;
    using cant_stop_native::Move;
    while (true) {
        int steps = 0;
        if (!(std::cin >> steps)) return 0;
        Game game;
        for (int step = 0; step < steps; ++step) {
            std::array<int, 4> dice{};
            std::cin >> dice[0] >> dice[1] >> dice[2] >> dice[3];
            const auto choices = game.roll(dice);
            int move_size = 0;
            std::cin >> move_size;
            Move move;
            move.size = move_size;
            for (int i = 0; i < move_size; ++i) std::cin >> move.columns[i];
            int decision = 0;
            std::cin >> decision;
            if (!choices.empty()) {
                if (!game.apply_move(move)) return 2;
                if (decision == 0) {
                    if (!game.continue_turn()) return 3;
                } else {
                    if (!game.stop()) return 4;
                }
            }
        }
        int active_count = 0;
        for (int position : game.active) active_count += position != -1;
        std::cout << static_cast<int>(game.phase) << ' ' << game.current_player
                  << ' ' << game.winner << ' ' << game.rolls_this_turn;
        for (const auto& row : game.permanent)
            for (int value : row) std::cout << ' ' << value;
        for (int value : game.owner) std::cout << ' ' << value;
        std::cout << ' ' << active_count;
        for (int index = 0; index < 11; ++index) {
            if (game.active[index] != -1)
                std::cout << ' ' << index + 2 << ' ' << game.active[index];
        }
        for (int value : game.turns) std::cout << ' ' << value;
        for (int value : game.rolls) std::cout << ' ' << value;
        for (int value : game.busts) std::cout << ' ' << value;
        for (int value : game.stops) std::cout << ' ' << value;
        std::cout << '\n';
    }
}

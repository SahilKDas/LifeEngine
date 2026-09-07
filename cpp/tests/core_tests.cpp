#include "life/life_core.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>

int main() {
  life::Nnue brain(7);
  const std::vector<int> features{0, 7, 18, 144, 146};
  const auto first = brain.evaluate(features);
  const auto second = brain.evaluate(features);
  assert(first == second);
  std::mt19937 random(9);
  brain.mutate(1.0f, 0.2f, random);
  assert(brain.evaluate(features) != first);
  life::World world(64, 48, 7);
  world.step(100);
  assert(world.width() == 64 && world.height() == 48 && world.cells() != nullptr);
  std::cout << "C++ core tests passed\n";
}

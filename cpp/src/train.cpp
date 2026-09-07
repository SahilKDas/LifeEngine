#include "life/life_core.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

static std::vector<int> sample(int food_square, int killer_square = -1,
                               int mover_square = -1, int mover_direction = 0,
                               int own_direction = 0) {
  std::vector<int> features;
  for (int square = 0; square < 24; ++square) {
    const int category = square == killer_square ? 5 : square == food_square ? 1 :
                         square == mover_square ? 7 + mover_direction : 0;
    features.push_back(square * 11 + category);
  }
  features.push_back(264); features.push_back(266); features.push_back(268 + own_direction);
  return features;
}

static int target_for(int square) {
  int cursor = 0;
  for (int y = -2; y <= 2; ++y) for (int x = -2; x <= 2; ++x) {
    if (!x && !y) continue;
    if (cursor++ != square) continue;
    return std::abs(x) > std::abs(y) ? (x < 0 ? 2 : 3) : (y < 0 ? 0 : 1);
  }
  return 4;
}

static int escape_target_for(int square) {
  const int toward = target_for(square);
  if (toward == 0) return 1;
  if (toward == 1) return 0;
  if (toward == 2) return 3;
  return 2;
}

static bool adjacent_square(int square) {
  int cursor = 0;
  for (int y = -2; y <= 2; ++y) for (int x = -2; x <= 2; ++x) {
    if (!x && !y) continue;
    if (cursor++ == square) return std::max(std::abs(x), std::abs(y)) == 1;
  }
  return false;
}

template <class Values> static void vector_json(std::ostream& out, const Values& values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
  out << ']';
}

int main(int argc, char** argv) {
  const std::string output = argc > 1 ? argv[1] : "trained-brain.json";
  life::Nnue brain(0xC0FFEE);
  std::mt19937 random(42);
  std::uniform_int_distribution<int> food(0, 23);
  for (int epoch = 0; epoch < 500; ++epoch)
    for (int batch = 0; batch < 96; ++batch) {
      const int square = food(random);
      brain.train(sample(square), target_for(square), 0.008f);
      int killer = food(random);
      brain.train(sample(-1, killer), escape_target_for(killer), 0.008f);
      while (killer == square) killer = food(random);
      brain.train(sample(square, killer), escape_target_for(killer), 0.008f);
      const int neighbor = food(random);
      const int heading = int(random() % 4);
      const int own_heading = int(random() % 4);
      const int flock_target = adjacent_square(neighbor) ? heading : target_for(neighbor);
      brain.train(sample(-1, -1, neighbor, heading, own_heading), flock_target, 0.008f);
    }
  int food_correct = 0, escape_correct = 0, mixed_correct = 0, flock_correct = 0;
  for (int square = 0; square < 24; ++square) {
    auto result = brain.evaluate(sample(square));
    food_correct += int(std::max_element(result.begin(), result.end()) - result.begin()) == target_for(square);
    result = brain.evaluate(sample(-1, square));
    escape_correct += int(std::max_element(result.begin(), result.end()) - result.begin()) == escape_target_for(square);
    const int food_square = (square + 12) % 24;
    result = brain.evaluate(sample(food_square, square));
    mixed_correct += int(std::max_element(result.begin(), result.end()) - result.begin()) == escape_target_for(square);
    for (int heading = 0; heading < 4; ++heading) {
      result = brain.evaluate(sample(-1, -1, square, heading, (heading + 1) % 4));
      const int target = adjacent_square(square) ? heading : target_for(square);
      flock_correct += int(std::max_element(result.begin(), result.end()) - result.begin()) == target;
    }
  }
  const std::filesystem::path output_path(output);
  if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
  std::ofstream out(output_path);
  if (!out) {
    std::cerr << "unable to write " << output << '\n';
    return 2;
  }
  out << "{\"inputWeights\":[";
  const auto& input = brain.input_weights();
  for (int feature = 0; feature < life::kInput; ++feature) {
    if (feature) out << ','; out << '[';
    for (int hidden = 0; hidden < life::kHidden; ++hidden) {
      if (hidden) out << ','; out << input[feature * life::kHidden + hidden];
    }
    out << ']';
  }
  out << "],\"hiddenBias\":"; vector_json(out, brain.hidden_bias());
  out << ",\"outputWeights\":[";
  for (int action = 0; action < life::kOutput; ++action) {
    if (action) out << ','; out << '[';
    for (int hidden = 0; hidden < life::kHidden; ++hidden) {
      if (hidden) out << ','; out << brain.output_weights()[action * life::kHidden + hidden];
    }
    out << ']';
  }
  out << "],\"outputBias\":"; vector_json(out, brain.output_bias()); out << "}\n";
  std::cout << "trained policy: food " << food_correct << "/24, killer avoidance "
            << escape_correct << "/24, mixed-scene avoidance " << mixed_correct
            << "/24, flocking " << flock_correct << "/96, wrote " << output << '\n';
  return food_correct >= 20 && escape_correct >= 20 && mixed_correct >= 20 &&
         flock_correct >= 80 ? 0 : 1;
}

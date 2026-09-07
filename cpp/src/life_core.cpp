#include "life/life_core.hpp"
#include <algorithm>
#include <cmath>

namespace life {
Nnue::Nnue(uint32_t seed) : input_(kInput * kHidden) {
  std::mt19937 random(seed);
  std::uniform_real_distribution<float> in(-0.12f, 0.12f), out(-0.18f, 0.18f);
  for (float& value : input_) value = in(random);
  for (float& value : hidden_bias_) value = in(random) * 0.4f;
  for (float& value : output_) value = out(random);
  for (float& value : output_bias_) value = in(random) * 0.4f;
}

void Nnue::rebuild(const std::vector<int>& features) {
  accumulator_ = hidden_bias_;
  for (int feature : features) if (feature >= 0 && feature < kInput)
    for (int h = 0; h < kHidden; ++h) accumulator_[h] += input_[feature * kHidden + h];
}

std::array<float, kOutput> Nnue::evaluate(const std::vector<int>& features) {
  rebuild(features);
  std::array<float, kOutput> result = output_bias_;
  for (int o = 0; o < kOutput; ++o)
    for (int h = 0; h < kHidden; ++h)
      result[o] += std::max(0.0f, accumulator_[h]) * output_[o * kHidden + h];
  return result;
}

void Nnue::train(const std::vector<int>& features, int target, float rate) {
  auto scores = evaluate(features);
  float maximum = *std::max_element(scores.begin(), scores.end()), sum = 0;
  std::array<float, kOutput> probability{};
  for (int o = 0; o < kOutput; ++o) sum += probability[o] = std::exp(scores[o] - maximum);
  for (float& value : probability) value /= sum;
  std::array<float, kHidden> hidden_gradient{};
  for (int o = 0; o < kOutput; ++o) {
    const float gradient = probability[o] - (o == target ? 1.0f : 0.0f);
    for (int h = 0; h < kHidden; ++h) {
      hidden_gradient[h] += gradient * output_[o * kHidden + h];
      output_[o * kHidden + h] -= rate * gradient * std::max(0.0f, accumulator_[h]);
    }
    output_bias_[o] -= rate * gradient;
  }
  for (int h = 0; h < kHidden; ++h) {
    if (accumulator_[h] <= 0) continue;
    const float gradient = hidden_gradient[h];
    hidden_bias_[h] -= rate * gradient;
    for (int feature : features) input_[feature * kHidden + h] -= rate * gradient;
  }
}

void Nnue::mutate(float probability, float magnitude, std::mt19937& random) {
  std::uniform_real_distribution<float> chance(0, 1), delta(-magnitude, magnitude);
  auto apply = [&](auto& values) { for (float& value : values) if (chance(random) < probability) value += delta(random); };
  apply(input_); apply(hidden_bias_); apply(output_); apply(output_bias_);
}

World::World(int width, int height, uint32_t seed)
    : width_(width), height_(height), cells_(width * height), random_(seed) {
  agents_.push_back({width / 2, height / 2, 0, 0, Nnue(seed)});
  cells_[index(width / 2, height / 2)] = 5;
}

std::vector<int> World::features(const Agent& agent) const {
  std::vector<int> result; result.reserve(26); int square = 0;
  for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx) {
    if (!dx && !dy) continue;
    const int x = agent.x + dx, y = agent.y + dy;
    int category = !valid(x, y) ? 6 : cells_[index(x, y)] == 1 ? 1 : cells_[index(x, y)] == 2 ? 2 : cells_[index(x, y)] == 6 ? 5 : cells_[index(x, y)] >= 3 ? 4 : 0;
    result.push_back(square++ * 11 + category);
  }
  result.push_back(264); result.push_back(266); result.push_back(268);
  return result;
}

void World::step(int count) {
  static constexpr int dx[] = {0, 0, -1, 1}, dy[] = {-1, 1, 0, 0};
  for (int tick = 0; tick < count; ++tick) for (Agent& agent : agents_) {
    auto output = agent.brain.evaluate(features(agent));
    int action = int(std::max_element(output.begin(), output.end()) - output.begin());
    if (action >= 4) continue;
    const int nx = agent.x + dx[action], ny = agent.y + dy[action];
    if (!valid(nx, ny) || cells_[index(nx, ny)] >= 2) continue;
    cells_[index(agent.x, agent.y)] = 0;
    agent.x = nx; agent.y = ny; cells_[index(nx, ny)] = 6;
  }
}
}  // namespace life

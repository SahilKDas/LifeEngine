#pragma once
#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace life {
constexpr int kInput = 172, kHidden = 16, kOutput = 5;

class Nnue {
 public:
  explicit Nnue(uint32_t seed = 1);
  std::array<float, kOutput> evaluate(const std::vector<int>& features);
  void train(const std::vector<int>& features, int target, float rate);
  void mutate(float probability, float magnitude, std::mt19937& random);
  const std::vector<float>& input_weights() const { return input_; }
  const std::array<float, kHidden>& hidden_bias() const { return hidden_bias_; }
  const std::array<float, kOutput * kHidden>& output_weights() const { return output_; }
  const std::array<float, kOutput>& output_bias() const { return output_bias_; }
 private:
  void rebuild(const std::vector<int>& features);
  std::vector<float> input_;
  std::array<float, kHidden> hidden_bias_{};
  std::array<float, kOutput * kHidden> output_{};
  std::array<float, kOutput> output_bias_{};
  std::array<float, kHidden> accumulator_{};
};

class World {
 public:
  World(int width, int height, uint32_t seed);
  void step(int count);
  int width() const { return width_; }
  int height() const { return height_; }
  const uint8_t* cells() const { return cells_.data(); }
 private:
  struct Agent { int x, y, food = 0, age = 0; Nnue brain; };
  std::vector<int> features(const Agent& agent) const;
  int index(int x, int y) const { return y * width_ + x; }
  bool valid(int x, int y) const { return x >= 0 && y >= 0 && x < width_ && y < height_; }
  int width_, height_;
  std::vector<uint8_t> cells_;
  std::vector<Agent> agents_;
  std::mt19937 random_;
};
}  // namespace life

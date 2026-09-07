#include "life/life_core.hpp"
extern "C" {
void* life_create(int width, int height, unsigned seed) { return new life::World(width, height, seed); }
void life_destroy(void* world) { delete static_cast<life::World*>(world); }
void life_step(void* world, int count) { static_cast<life::World*>(world)->step(count); }
const unsigned char* life_cells(void* world) { return static_cast<life::World*>(world)->cells(); }
int life_width(void* world) { return static_cast<life::World*>(world)->width(); }
int life_height(void* world) { return static_cast<life::World*>(world)->height(); }
}

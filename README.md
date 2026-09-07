# Life Engine NNUE

A modern rewrite of Life Engine: a browser-based evolutionary ecosystem where organisms inherit body plans and sparse neural-network behavior.

## What changed

- Strict TypeScript replaces the original JavaScript, jQuery, and Webpack application.
- Vite 8 provides the development server and optimized production build.
- Vitest covers the TypeScript NNUE implementation.
- A C++17 core implements fast NNUE evaluation/training and a contiguous-memory world stepper.
- The C++ trainer produces the initial browser model in public/trained-brain.json.
- A C ABI and Emscripten CMake target are included for WebAssembly builds.
- The interface keeps the original blue, black, white, and teal visual identity in a responsive layout.

## Run the application

~~~sh
npm install
npm run dev
~~~

Production verification:

~~~sh
npm test
npm run build
~~~

## Build and train the C++ core

With CMake and a C++17 compiler:

~~~sh
cmake -S cpp -B build/cpp -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp --config Release
ctest --test-dir build/cpp -C Release --output-on-failure
npm run train
~~~

The trainer uses supervised backpropagation in native C++ to teach the seed NNUE to select the direction of nearby food. On this repository's verification set it reaches 24/24 directional decisions. Evolution then clones and mutates those weights independently for each organism.

## WebAssembly

The C++ core exposes a small C ABI in cpp/src/wasm_api.cpp. Configure the same CMake project with Emscripten to produce the browser module:

~~~sh
emcmake cmake -S cpp -B build/wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build/wasm --config Release
~~~

Emscripten is optional for development because the TypeScript engine is a typed-array fallback and consumes the same C++-trained brain format. Native C++ remains the source for model training and performance benchmarks.

## Neural inputs

Each NNUE receives a sparse 5x5 neighborhood encoded as one active feature per square:

- empty space
- food
- walls
- its own body
- other organisms
- map boundaries

Killer cells have their own danger feature rather than being grouped with ordinary organisms. Two additional one-hot pairs encode hunger/reproduction readiness and damage. Outputs represent up, down, left, right, and wait. Only changed features update the hidden accumulator.

NNUEs exist only on organisms containing a mover cell. Static organisms use the original Life Engine rules without allocating or evaluating a neural network. The founding organism is the original three-cell body: one central mouth and two diagonal producers.

## Cell palette

The original cell colors are preserved: green food, orange mouths, white producers, blue movers, red killers, purple armor, gray walls, and a dark-blue world.

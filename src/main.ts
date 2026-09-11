import "./style.css";
import { Simulation } from "./core/simulation";
import { CELL_COLORS, CellType } from "./core/types";
import type { BrainSeed } from "./core/nnue";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing app root");

app.innerHTML = `
  <main class="shell">
    <header>
      <div><span class="eyebrow">EVOLUTION, ACCELERATED</span><h1>Life Engine <em>NNUE</em></h1></div>
      <div class="status"><i></i><span id="run-state">Running</span></div>
    </header>
    <section class="stage">
      <canvas id="world" aria-label="Life Engine simulation"></canvas>
      <div class="legend">
        <span><b class="food"></b>Food</span><span><b class="mouth"></b>Mouth</span>
        <span><b class="producer"></b>Producer</span><span><b class="mover"></b>Mover</span>
        <span><b class="killer"></b>Killer</span><span><b class="armor"></b>Armor</span>
      </div>
    </section>
    <aside>
      <section class="panel hero-panel">
        <p class="label">CORE</p><h2>Sparse speed.<br><span>Neural instinct.</span></h2>
        <p>Sparse NNUE brains inherit, mutate, and learn movement strategies under natural selection.</p>
      </section>
      <section class="metrics">
        <article><span>ORGANISMS</span><strong id="organisms">0</strong></article>
        <article><span>RECORD</span><strong id="record">0</strong></article>
        <article><span>GENERATION</span><strong id="generation">0</strong></article>
        <article><span>LARGEST</span><strong id="largest">0</strong></article>
      </section>
      <section class="panel controls">
        <div class="control-title"><h3>Simulation</h3><button id="toggle">Pause</button></div>
        <label>Ticks per frame <output id="speed-value">2</output><input id="speed" type="range" min="1" max="40" value="2"></label>
        <label>Food production <output id="food-value">4%</output><input id="food-rate" type="range" min="0" max="20" value="4"></label>
        <div class="button-grid"><button id="reset">Reset world</button><button id="seed">Seed food</button></div>
      </section>
      <section class="panel tools">
        <h3>World tools</h3>
        <div class="tool-row">
          <button class="tool active" data-tool="1">Food</button>
          <button class="tool" data-tool="2">Wall</button>
          <button class="tool" data-tool="0">Erase</button>
        </div>
        <p>Paint directly on the ecosystem. Painting over an organism removes it.</p>
      </section>
      <footer><span id="core-state">TypeScript NNUE</span><span id="ticks">0 ticks</span></footer>
    </aside>
  </main>`;

const canvas = document.querySelector<HTMLCanvasElement>("#world")!;
const context = canvas.getContext("2d", { alpha: false })!;
const cellSize = 5;
let seed: BrainSeed | undefined;
try {
  const response = await fetch("/trained-brain.json");
  if (response.ok) seed = await response.json() as BrainSeed;
} catch { /* random initialization remains available */ }

let simulation: Simulation;
let running = true;
let ticksPerFrame = 2;
let tool = CellType.Food;

function resize(): void {
  const bounds = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.floor(bounds.width * scale);
  canvas.height = Math.floor(bounds.height * scale);
  context.setTransform(scale, 0, 0, scale, 0, 0);
  simulation = new Simulation({
    width: Math.max(40, Math.floor(bounds.width / cellSize)),
    height: Math.max(40, Math.floor(bounds.height / cellSize)),
    foodChance: 0.04,
    lifespan: 120,
  }, seed);
}

function render(): void {
  const width = simulation.width;
  context.fillStyle = CELL_COLORS[CellType.Empty];
  context.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  for (let index = 0; index < simulation.cells.length; index++) {
    const type = simulation.cells[index] as CellType;
    if (type === CellType.Empty) continue;
    context.fillStyle = CELL_COLORS[type];
    context.fillRect((index % width) * cellSize, Math.floor(index / width) * cellSize, cellSize, cellSize);
  }
}

function frame(): void {
  if (running) simulation.step(ticksPerFrame);
  render();
  const metrics = simulation.metrics();
  for (const key of ["organisms", "record", "generation", "largest"] as const) {
    document.querySelector<HTMLElement>(`#${key}`)!.textContent = String(metrics[key]);
  }
  document.querySelector("#ticks")!.textContent = `${metrics.ticks.toLocaleString()} ticks`;
  requestAnimationFrame(frame);
}

document.querySelector("#toggle")!.addEventListener("click", (event) => {
  running = !running;
  (event.currentTarget as HTMLButtonElement).textContent = running ? "Pause" : "Resume";
  document.querySelector("#run-state")!.textContent = running ? "Running" : "Paused";
  document.body.classList.toggle("paused", !running);
});
document.querySelector("#reset")!.addEventListener("click", () => simulation.reset());
document.querySelector("#seed")!.addEventListener("click", () => {
  for (let i = 0; i < 500; i++) simulation.paint(
    Math.floor(Math.random() * simulation.width),
    Math.floor(Math.random() * simulation.height),
    CellType.Food,
  );
});
document.querySelector<HTMLInputElement>("#speed")!.addEventListener("input", (event) => {
  ticksPerFrame = Number((event.target as HTMLInputElement).value);
  document.querySelector("#speed-value")!.textContent = String(ticksPerFrame);
});
document.querySelector<HTMLInputElement>("#food-rate")!.addEventListener("input", (event) => {
  const value = Number((event.target as HTMLInputElement).value);
  simulation.foodChance = value / 100;
  document.querySelector("#food-value")!.textContent = `${value}%`;
});
document.querySelectorAll<HTMLButtonElement>(".tool").forEach((button) => button.addEventListener("click", () => {
  document.querySelector(".tool.active")?.classList.remove("active");
  button.classList.add("active");
  tool = Number(button.dataset.tool) as CellType;
}));
canvas.addEventListener("pointerdown", (event) => {
  const bounds = canvas.getBoundingClientRect();
  simulation.paint(Math.floor((event.clientX - bounds.left) / cellSize), Math.floor((event.clientY - bounds.top) / cellSize), tool);
});

window.addEventListener("resize", resize);
resize();
frame();

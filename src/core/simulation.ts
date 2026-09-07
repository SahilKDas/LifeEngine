import { FEATURE_CATEGORIES, Nnue, POSITION_COUNT, SENSOR_RADIUS, type BrainSeed } from "./nnue";
import { CellType, DIRECTIONS, type LocalCell, type Metrics } from "./types";

interface Organism {
  id: number;
  x: number;
  y: number;
  cells: LocalCell[];
  brain: Nnue;
  food: number;
  age: number;
  damage: number;
  mutationRate: number;
}

export interface SimulationOptions {
  width: number;
  height: number;
  foodChance: number;
  lifespan: number;
}

export class Simulation {
  readonly width: number;
  readonly height: number;
  readonly cells: Uint8Array;
  readonly owners: Int32Array;
  private organisms = new Map<number, Organism>();
  private nextId = 1;
  private ticks = 0;
  private generation = 0;
  private record = 0;
  private largest = 0;
  private seed?: BrainSeed;
  foodChance: number;
  lifespan: number;

  constructor(options: SimulationOptions, seed?: BrainSeed) {
    this.width = options.width;
    this.height = options.height;
    this.cells = new Uint8Array(this.width * this.height);
    this.owners = new Int32Array(this.width * this.height);
    this.owners.fill(-1);
    this.foodChance = options.foodChance;
    this.lifespan = options.lifespan;
    this.seed = seed;
    this.reset();
  }

  reset(): void {
    this.cells.fill(CellType.Empty);
    this.owners.fill(-1);
    this.organisms.clear();
    this.ticks = 0;
    this.generation++;
    const organism: Organism = {
      id: this.nextId++,
      x: Math.floor(this.width / 2),
      y: Math.floor(this.height / 2),
      cells: [
        { type: CellType.Mouth, x: 0, y: 0 },
        { type: CellType.Producer, x: -1, y: 0 },
        { type: CellType.Mover, x: 1, y: 0 },
      ],
      brain: new Nnue(this.seed),
      food: 0,
      age: 0,
      damage: 0,
      mutationRate: 0.05,
    };
    this.add(organism);
  }

  step(count = 1): void {
    for (let iteration = 0; iteration < count; iteration++) {
      this.ticks++;
      for (const organism of [...this.organisms.values()]) this.updateOrganism(organism);
      if (this.organisms.size === 0) this.reset();
      this.record = Math.max(this.record, this.organisms.size);
    }
  }

  paint(x: number, y: number, type: CellType): void {
    if (!this.valid(x, y)) return;
    const index = this.index(x, y);
    const owner = this.owners[index] ?? -1;
    if (owner >= 0) this.kill(owner);
    this.cells[index] = type;
    this.owners[index] = -1;
  }

  metrics(): Metrics {
    const mutation = [...this.organisms.values()].reduce((sum, organism) => sum + organism.mutationRate, 0);
    return {
      organisms: this.organisms.size,
      record: this.record,
      generation: this.generation,
      ticks: this.ticks,
      largest: this.largest,
      averageMutation: this.organisms.size ? mutation / this.organisms.size : 0,
    };
  }

  private updateOrganism(organism: Organism): void {
    if (!this.organisms.has(organism.id)) return;
    organism.age++;
    if (organism.age > organism.cells.length * this.lifespan || organism.damage >= organism.cells.length) {
      this.kill(organism.id);
      return;
    }
    for (const local of organism.cells) {
      if (local.type === CellType.Mouth) this.eat(organism, local);
      if (local.type === CellType.Producer) this.produce(organism, local);
      if (local.type === CellType.Killer) this.attack(organism, local);
    }
    if (organism.food >= organism.cells.length) this.reproduce(organism);
    if (organism.cells.some((cell) => cell.type === CellType.Mover)) {
      const action = organism.brain.action(this.features(organism));
      if (action < 4) this.move(organism, action);
    }
  }

  private eat(organism: Organism, local: LocalCell): void {
    for (const [dx, dy] of DIRECTIONS) {
      const x = organism.x + local.x + dx;
      const y = organism.y + local.y + dy;
      if (this.at(x, y) === CellType.Food) {
        const index = this.index(x, y);
        this.cells[index] = CellType.Empty;
        organism.food++;
      }
    }
  }

  private produce(organism: Organism, local: LocalCell): void {
    if (Math.random() >= this.foodChance) return;
    const [dx, dy] = DIRECTIONS[Math.floor(Math.random() * DIRECTIONS.length)]!;
    const x = organism.x + local.x + dx;
    const y = organism.y + local.y + dy;
    if (this.at(x, y) === CellType.Empty) this.cells[this.index(x, y)] = CellType.Food;
  }

  private attack(organism: Organism, local: LocalCell): void {
    for (const [dx, dy] of DIRECTIONS) {
      const index = this.safeIndex(organism.x + local.x + dx, organism.y + local.y + dy);
      if (index < 0) continue;
      const victimId = this.owners[index] ?? -1;
      const victim = this.organisms.get(victimId);
      if (victim && victim.id !== organism.id && this.cells[index] !== CellType.Armor) victim.damage++;
    }
  }

  private move(organism: Organism, action: number): void {
    const [dx, dy] = DIRECTIONS[action]!;
    if (!organism.cells.every((cell) => this.passable(organism.x + cell.x + dx, organism.y + cell.y + dy, organism.id))) return;
    this.clearBody(organism);
    organism.x += dx;
    organism.y += dy;
    this.placeBody(organism);
  }

  private reproduce(parent: Organism): void {
    parent.food -= parent.cells.length;
    const action = Math.floor(Math.random() * 4);
    const [dx, dy] = DIRECTIONS[action]!;
    const child: Organism = {
      ...parent,
      id: this.nextId++,
      x: parent.x + dx * (parent.cells.length + 2),
      y: parent.y + dy * (parent.cells.length + 2),
      cells: parent.cells.map((cell) => ({ ...cell })),
      brain: parent.brain.clone(),
      food: 0,
      age: 0,
      damage: 0,
    };
    if (Math.random() < child.mutationRate) this.mutate(child);
    if (child.cells.every((cell) => this.passable(child.x + cell.x, child.y + cell.y, child.id))) this.add(child);
  }

  private mutate(organism: Organism): void {
    organism.brain.mutate();
    organism.mutationRate = Math.min(0.5, Math.max(0.005, organism.mutationRate + (Math.random() - 0.5) * 0.02));
    const choice = Math.floor(Math.random() * 3);
    if (choice === 0 && organism.cells.length < 32) {
      const base = organism.cells[Math.floor(Math.random() * organism.cells.length)]!;
      const [dx, dy] = DIRECTIONS[Math.floor(Math.random() * 4)]!;
      const x = base.x + dx;
      const y = base.y + dy;
      if (!organism.cells.some((cell) => cell.x === x && cell.y === y)) {
        organism.cells.push({ type: CellType.Mouth + Math.floor(Math.random() * 5), x, y });
      }
    } else if (choice === 1) {
      organism.cells[Math.floor(Math.random() * organism.cells.length)]!.type = CellType.Mouth + Math.floor(Math.random() * 5);
    } else if (organism.cells.length > 1) {
      organism.cells.splice(1 + Math.floor(Math.random() * (organism.cells.length - 1)), 1);
    }
  }

  private features(organism: Organism): number[] {
    const features: number[] = [];
    let square = 0;
    for (let dy = -SENSOR_RADIUS; dy <= SENSOR_RADIUS; dy++) {
      for (let dx = -SENSOR_RADIUS; dx <= SENSOR_RADIUS; dx++) {
        if (dx === 0 && dy === 0) continue;
        const index = this.safeIndex(organism.x + dx, organism.y + dy);
        let category = 5;
        if (index >= 0) {
          const type = this.cells[index] as CellType;
          const owner = this.owners[index] ?? -1;
          category = owner === organism.id ? 3 : owner >= 0 ? 4 : type === CellType.Food ? 1 : type === CellType.Wall ? 2 : 0;
        }
        features.push(square++ * FEATURE_CATEGORIES + category);
      }
    }
    features.push(POSITION_COUNT * FEATURE_CATEGORIES + (organism.food >= organism.cells.length ? 1 : 0));
    features.push(POSITION_COUNT * FEATURE_CATEGORIES + 2 + (organism.damage > 0 ? 1 : 0));
    return features;
  }

  private add(organism: Organism): void {
    this.organisms.set(organism.id, organism);
    this.largest = Math.max(this.largest, organism.cells.length);
    this.placeBody(organism);
  }

  private kill(id: number): void {
    const organism = this.organisms.get(id);
    if (!organism) return;
    for (const cell of organism.cells) {
      const index = this.safeIndex(organism.x + cell.x, organism.y + cell.y);
      if (index >= 0 && this.owners[index] === id) {
        this.cells[index] = CellType.Food;
        this.owners[index] = -1;
      }
    }
    this.organisms.delete(id);
  }

  private placeBody(organism: Organism): void {
    for (const cell of organism.cells) {
      const index = this.index(organism.x + cell.x, organism.y + cell.y);
      this.cells[index] = cell.type;
      this.owners[index] = organism.id;
    }
  }

  private clearBody(organism: Organism): void {
    for (const cell of organism.cells) {
      const index = this.index(organism.x + cell.x, organism.y + cell.y);
      this.cells[index] = CellType.Empty;
      this.owners[index] = -1;
    }
  }

  private passable(x: number, y: number, owner: number): boolean {
    const index = this.safeIndex(x, y);
    return index >= 0 && ((this.cells[index] === CellType.Empty) || this.owners[index] === owner);
  }

  private at(x: number, y: number): CellType | undefined {
    const index = this.safeIndex(x, y);
    return index < 0 ? undefined : this.cells[index] as CellType;
  }

  private valid(x: number, y: number): boolean {
    return x >= 0 && y >= 0 && x < this.width && y < this.height;
  }

  private safeIndex(x: number, y: number): number {
    return this.valid(x, y) ? this.index(x, y) : -1;
  }

  private index(x: number, y: number): number {
    return y * this.width + x;
  }
}

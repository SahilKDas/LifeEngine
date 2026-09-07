export const SENSOR_RADIUS = 2;
export const FEATURE_CATEGORIES = 7;
export const HIDDEN_SIZE = 16;
export const OUTPUT_SIZE = 5;
export const POSITION_COUNT = 24;
export const INPUT_SIZE = POSITION_COUNT * FEATURE_CATEGORIES + 4;

export interface BrainSeed {
  inputWeights: number[][];
  hiddenBias: number[];
  outputWeights: number[][];
  outputBias: number[];
}

const randomWeight = (scale: number): number => (Math.random() * 2 - 1) * scale;

export class Nnue {
  readonly inputWeights: Float32Array[];
  readonly hiddenBias: Float32Array;
  readonly outputWeights: Float32Array[];
  readonly outputBias: Float32Array;
  private accumulator: Float32Array;
  private active = new Set<number>();

  constructor(seed?: BrainSeed) {
    this.inputWeights = seed
      ? seed.inputWeights.map((row) => Float32Array.from(row))
      : Array.from({ length: INPUT_SIZE }, () => Float32Array.from({ length: HIDDEN_SIZE }, () => randomWeight(0.12)));
    this.hiddenBias = seed ? Float32Array.from(seed.hiddenBias) : Float32Array.from({ length: HIDDEN_SIZE }, () => randomWeight(0.05));
    this.outputWeights = seed
      ? seed.outputWeights.map((row) => Float32Array.from(row))
      : Array.from({ length: OUTPUT_SIZE }, () => Float32Array.from({ length: HIDDEN_SIZE }, () => randomWeight(0.18)));
    this.outputBias = seed ? Float32Array.from(seed.outputBias) : Float32Array.from({ length: OUTPUT_SIZE }, () => randomWeight(0.05));
    this.accumulator = new Float32Array(this.hiddenBias);
  }

  clone(): Nnue {
    return new Nnue(this.toSeed());
  }

  toSeed(): BrainSeed {
    return {
      inputWeights: this.inputWeights.map((row) => Array.from(row)),
      hiddenBias: Array.from(this.hiddenBias),
      outputWeights: this.outputWeights.map((row) => Array.from(row)),
      outputBias: Array.from(this.outputBias),
    };
  }

  evaluate(features: readonly number[]): Float32Array {
    const next = new Set(features);
    for (const feature of this.active) if (!next.has(feature)) this.apply(feature, -1);
    for (const feature of next) if (!this.active.has(feature)) this.apply(feature, 1);
    this.active = next;

    return Float32Array.from({ length: OUTPUT_SIZE }, (_, output) => {
      let value = this.outputBias[output] ?? 0;
      const row = this.outputWeights[output]!;
      for (let hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
        value += Math.max(0, this.accumulator[hidden] ?? 0) * (row[hidden] ?? 0);
      }
      return value;
    });
  }

  action(features: readonly number[]): number {
    const output = this.evaluate(features);
    let best = 0;
    for (let i = 1; i < output.length; i++) if (output[i]! > output[best]!) best = i;
    return best;
  }

  mutate(probability = 0.03, magnitude = 0.2): void {
    let changed = false;
    const perturb = (values: Float32Array): void => {
      for (let i = 0; i < values.length; i++) {
        if (Math.random() < probability) {
          values[i] = (values[i] ?? 0) + randomWeight(magnitude);
          changed = true;
        }
      }
    };
    this.inputWeights.forEach(perturb);
    perturb(this.hiddenBias);
    this.outputWeights.forEach(perturb);
    perturb(this.outputBias);
    if (!changed) this.outputBias[Math.floor(Math.random() * OUTPUT_SIZE)]! += randomWeight(magnitude);
    this.reset();
  }

  private apply(feature: number, sign: number): void {
    const weights = this.inputWeights[feature];
    if (!weights) return;
    for (let hidden = 0; hidden < HIDDEN_SIZE; hidden++) {
      this.accumulator[hidden] = (this.accumulator[hidden] ?? 0) + sign * (weights[hidden] ?? 0);
    }
  }

  private reset(): void {
    this.active.clear();
    this.accumulator = new Float32Array(this.hiddenBias);
  }
}

import { describe, expect, it } from "vitest";
import { HIDDEN_SIZE, Nnue, OUTPUT_SIZE } from "./nnue";

describe("Nnue", () => {
  it("returns stable output when sparse features do not change", () => {
    const brain = new Nnue();
    expect(Array.from(brain.evaluate([0, 7, 18]))).toEqual(Array.from(brain.evaluate([0, 7, 18])));
  });

  it("deep clones heritable weights", () => {
    const parent = new Nnue();
    const child = parent.clone();
    expect(child).not.toBe(parent);
    expect(child.toSeed()).toEqual(parent.toSeed());
  });

  it("has the expected compact topology", () => {
    const seed = new Nnue().toSeed();
    expect(seed.hiddenBias).toHaveLength(HIDDEN_SIZE);
    expect(seed.outputBias).toHaveLength(OUTPUT_SIZE);
  });
});

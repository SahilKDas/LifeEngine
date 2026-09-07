const assert = require("assert");
const NNUE = require("../src/Organism/NNUE");
const Organism = require("../src/Organism/Organism");
const Cell = require("../src/Organism/Cell/Cell");
const CellTypes = require("../src/Organism/Cell/CellTypes");

function makeEnvironment(cols=7, rows=7) {
    const grid = [];
    for (let c=0; c<cols; c++) {
        grid[c] = [];
        for (let r=0; r<rows; r++) grid[c][r] = new Cell(CellTypes.empty, c, r, c, r);
    }
    return {
        grid_map: {
            cellAt(c, r) {
                return c >= 0 && r >= 0 && c < cols && r < rows ? grid[c][r] : null;
            }
        },
        changeCell(c, r, type, owner) {
            const cell = this.grid_map.cellAt(c, r);
            cell.setType(type);
            cell.owner = owner;
        },
        addOrganism() {}
    };
}

function closeEnough(actual, expected, message) {
    assert.strictEqual(actual.length, expected.length);
    for (let i=0; i<actual.length; i++) {
        assert.ok(Math.abs(actual[i] - expected[i]) < 1e-5,
            `${message} at ${i}: ${actual[i]} != ${expected[i]}`);
    }
}

function testSparseAccumulator() {
    const brain = new NNUE(() => 0.75);
    brain.updateAccumulator([0, 7, 18]);
    brain.updateAccumulator([7, 18, 25]);

    const expected = new Float32Array(brain.hidden_bias);
    for (const feature of [7, 18, 25]) {
        for (let hidden=0; hidden<brain.hidden_size; hidden++) {
            expected[hidden] += brain.input_weights[feature][hidden];
        }
    }
    closeEnough(brain.accumulator, expected, "incremental accumulator diverged");
}

function testFeatureEncoding() {
    const env = makeEnvironment();
    const organism = new Organism(3, 3, env);
    organism.addCell(CellTypes.mover, 0, 0);
    organism.updateGrid();
    env.changeCell(3, 2, CellTypes.food, null);

    const features = organism.brain.featuresFor(organism);
    assert.strictEqual(features.length, 26, "5x5 view and state must be one-hot encoded");
    assert.strictEqual(new Set(features).size, features.length, "active features must be unique");
}

function testBrainInheritanceAndMutation() {
    const env = makeEnvironment();
    const parent = new Organism(3, 3, env);
    parent.addCell(CellTypes.mover, 0, 0);
    const child = new Organism(3, 3, env, parent);

    assert.notStrictEqual(child.brain, parent.brain, "offspring must own a separate brain");
    assert.deepStrictEqual(Array.from(child.brain.output_bias), Array.from(parent.brain.output_bias));

    child.brain.mutate(1, 0.2, () => 0.75);
    assert.notDeepStrictEqual(Array.from(child.brain.output_bias), Array.from(parent.brain.output_bias));
}

function testActionSelection() {
    const env = makeEnvironment();
    const organism = new Organism(3, 3, env);
    organism.addCell(CellTypes.mover, 0, 0);
    for (const row of organism.brain.input_weights) row.fill(0);
    for (const row of organism.brain.output_weights) row.fill(0);
    organism.brain.hidden_bias.fill(0);
    organism.brain.output_bias.fill(0);
    organism.brain.output_bias[3] = 1;

    assert.strictEqual(organism.chooseMovement(), 3, "highest-scoring NNUE action must be selected");
}

testSparseAccumulator();
testFeatureEncoding();
testBrainInheritanceAndMutation();
testActionSelection();
console.log("NNUE tests passed");

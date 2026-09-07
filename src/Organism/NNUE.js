const CellTypes = require("./Cell/CellTypes");

// A small, sparse efficiently-updatable neural network. Each square in the
// organism's local 5x5 view contributes exactly one active feature. When the
// view changes, only added/removed feature weights are applied to the cached
// hidden-layer accumulator.
class NNUE {
    constructor(random=Math.random) {
        this.input_size = NNUE.inputSize();
        this.hidden_size = NNUE.hiddenSize;
        this.output_size = NNUE.outputSize;

        this.input_weights = NNUE.createMatrix(this.input_size, this.hidden_size,
            () => NNUE.randomWeight(random, 0.12));
        this.hidden_bias = NNUE.createVector(this.hidden_size,
            () => NNUE.randomWeight(random, 0.05));
        this.output_weights = NNUE.createMatrix(this.output_size, this.hidden_size,
            () => NNUE.randomWeight(random, 0.18));
        this.output_bias = NNUE.createVector(this.output_size,
            () => NNUE.randomWeight(random, 0.05));

        this.resetAccumulator();
    }

    static inputSize() {
        const width = NNUE.sensorRadius * 2 + 1;
        const sensedSquares = width * width - 1;
        return sensedSquares * NNUE.featureCategories + NNUE.stateFeatures;
    }

    static createVector(length, valueAt) {
        const result = new Float32Array(length);
        for (let i=0; i<length; i++) result[i] = valueAt(i);
        return result;
    }

    static createMatrix(rows, columns, valueAt) {
        const result = new Array(rows);
        for (let r=0; r<rows; r++) {
            result[r] = NNUE.createVector(columns, c => valueAt(r, c));
        }
        return result;
    }

    static randomWeight(random, scale) {
        return (random() * 2 - 1) * scale;
    }

    resetAccumulator() {
        this.active_features = new Set();
        this.accumulator = new Float32Array(this.hidden_bias);
    }

    clone() {
        const copy = Object.create(NNUE.prototype);
        copy.input_size = this.input_size;
        copy.hidden_size = this.hidden_size;
        copy.output_size = this.output_size;
        copy.input_weights = this.input_weights.map(row => new Float32Array(row));
        copy.hidden_bias = new Float32Array(this.hidden_bias);
        copy.output_weights = this.output_weights.map(row => new Float32Array(row));
        copy.output_bias = new Float32Array(this.output_bias);
        copy.resetAccumulator();
        return copy;
    }

    featuresFor(organism) {
        const features = [];
        let squareIndex = 0;

        for (let row=-NNUE.sensorRadius; row<=NNUE.sensorRadius; row++) {
            for (let col=-NNUE.sensorRadius; col<=NNUE.sensorRadius; col++) {
                if (col === 0 && row === 0) continue;
                const cell = organism.env.grid_map.cellAt(organism.c + col, organism.r + row);
                const category = this.categoryFor(cell, organism);
                features.push(squareIndex * NNUE.featureCategories + category);
                squareIndex++;
            }
        }

        const stateOffset = squareIndex * NNUE.featureCategories;
        features.push(stateOffset + (organism.food_collected >= organism.foodNeeded() ? 1 : 0));
        features.push(stateOffset + 2 + (organism.damage > 0 ? 1 : 0));
        return features;
    }

    categoryFor(cell, organism) {
        if (cell == null) return NNUE.categories.outOfBounds;
        if (cell.owner === organism) return NNUE.categories.self;
        if (cell.owner != null || cell.isLiving()) return NNUE.categories.organism;
        if (cell.type === CellTypes.food) return NNUE.categories.food;
        if (cell.type === CellTypes.wall) return NNUE.categories.wall;
        return NNUE.categories.empty;
    }

    updateAccumulator(features) {
        const next = new Set(features);

        for (const feature of this.active_features) {
            if (!next.has(feature)) this.applyFeature(feature, -1);
        }
        for (const feature of next) {
            if (!this.active_features.has(feature)) this.applyFeature(feature, 1);
        }

        this.active_features = next;
    }

    applyFeature(feature, sign) {
        const weights = this.input_weights[feature];
        for (let hidden=0; hidden<this.hidden_size; hidden++) {
            this.accumulator[hidden] += sign * weights[hidden];
        }
    }

    evaluate(features) {
        this.updateAccumulator(features);
        const outputs = new Float32Array(this.output_size);

        for (let output=0; output<this.output_size; output++) {
            let value = this.output_bias[output];
            for (let hidden=0; hidden<this.hidden_size; hidden++) {
                value += Math.max(0, this.accumulator[hidden]) * this.output_weights[output][hidden];
            }
            outputs[output] = value;
        }
        return outputs;
    }

    chooseAction(organism) {
        const outputs = this.evaluate(this.featuresFor(organism));
        let best = 0;
        for (let i=1; i<outputs.length; i++) {
            if (outputs[i] > outputs[best]) best = i;
        }
        return best;
    }

    mutate(probability, magnitude, random=Math.random) {
        let changed = false;
        const perturb = vector => {
            for (let i=0; i<vector.length; i++) {
                if (random() < probability) {
                    vector[i] += NNUE.randomWeight(random, magnitude);
                    changed = true;
                }
            }
        };

        for (const row of this.input_weights) perturb(row);
        perturb(this.hidden_bias);
        for (const row of this.output_weights) perturb(row);
        perturb(this.output_bias);

        // A mutation event must always make a heritable neural change, even
        // when all probability checks happened to miss.
        if (!changed) {
            const output = Math.floor(random() * this.output_size);
            this.output_bias[output] += NNUE.randomWeight(random, magnitude);
        }
        this.resetAccumulator();
    }
}

NNUE.sensorRadius = 2;
NNUE.hiddenSize = 16;
NNUE.outputSize = 5; // up, down, left, right, wait
NNUE.featureCategories = 6;
NNUE.stateFeatures = 4; // hungry/full and healthy/damaged one-hot pairs
NNUE.categories = {
    empty: 0,
    food: 1,
    wall: 2,
    self: 3,
    organism: 4,
    outOfBounds: 5
};
NNUE.waitAction = 4;

module.exports = NNUE;

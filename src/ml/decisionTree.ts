import { DecisionTreeNode, FeatureImportance } from './types.ts';

function calculateVariance(y: number[]): number {
  const n = y.length;
  if (n <= 1) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += y[i];
  const mean = sum / n;
  let varSum = 0;
  for (let i = 0; i < n; i++) {
    const d = y[i] - mean;
    varSum += d * d;
  }
  return varSum / n;
}

function calculateMean(y: number[]): number {
  if (y.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < y.length; i++) sum += y[i];
  return sum / y.length;
}

export class DecisionTreeRegressor {
  root: DecisionTreeNode | null = null;
  maxDepth: number;
  minSamplesSplit: number;
  featureNames: string[] = [];
  featureImportanceMap: Map<number, number> = new Map();

  constructor(maxDepth = 6, minSamplesSplit = 10) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  train(
    X: number[][],
    y: number[],
    featureNames: string[],
    featureSubsetIndices?: number[]
  ) {
    this.featureNames = featureNames;
    this.featureImportanceMap.clear();

    const nSamples = X.length;
    if (nSamples === 0) return;

    const sampleIndices = Array.from({ length: nSamples }, (_, i) => i);
    const availableFeatures =
      featureSubsetIndices || Array.from({ length: X[0].length }, (_, i) => i);

    this.root = this.buildTree(X, y, sampleIndices, 0, availableFeatures);
  }

  private buildTree(
    X: number[][],
    y: number[],
    indices: number[],
    depth: number,
    availableFeatures: number[]
  ): DecisionTreeNode {
    const nodeSamples = indices.map((i) => y[i]);
    const nodeMean = calculateMean(nodeSamples);

    // Stop conditions
    if (
      depth >= this.maxDepth ||
      indices.length < this.minSamplesSplit ||
      calculateVariance(nodeSamples) < 1e-4
    ) {
      return {
        value: nodeMean,
        samplesCount: indices.length,
      };
    }

    const currentVar = calculateVariance(nodeSamples);
    let bestVarianceReduction = -1;
    let bestFeature = -1;
    let bestThreshold = 0;
    let bestLeftIndices: number[] = [];
    let bestRightIndices: number[] = [];

    // Search across candidate features
    for (const fIdx of availableFeatures) {
      // Find candidate thresholds (quantiles to be efficient)
      const values = indices.map((i) => X[i][fIdx]);
      values.sort((a, b) => a - b);

      // Pick up to 10 split candidates across percentiles
      const numSteps = Math.min(10, values.length);
      const step = Math.max(1, Math.floor(values.length / numSteps));

      for (let s = 1; s < numSteps; s++) {
        const threshold = values[s * step];
        const left: number[] = [];
        const right: number[] = [];

        for (const idx of indices) {
          if (X[idx][fIdx] <= threshold) {
            left.push(idx);
          } else {
            right.push(idx);
          }
        }

        if (left.length === 0 || right.length === 0) continue;

        const leftY = left.map((i) => y[i]);
        const rightY = right.map((i) => y[i]);
        const leftVar = calculateVariance(leftY);
        const rightVar = calculateVariance(rightY);

        const weightedChildVar =
          (left.length / indices.length) * leftVar +
          (right.length / indices.length) * rightVar;
        const reduction = currentVar - weightedChildVar;

        if (reduction > bestVarianceReduction) {
          bestVarianceReduction = reduction;
          bestFeature = fIdx;
          bestThreshold = threshold;
          bestLeftIndices = left;
          bestRightIndices = right;
        }
      }
    }

    if (bestVarianceReduction <= 1e-6 || bestLeftIndices.length === 0 || bestRightIndices.length === 0) {
      return {
        value: nodeMean,
        samplesCount: indices.length,
      };
    }

    // Accumulate feature importance
    const prevImportance = this.featureImportanceMap.get(bestFeature) || 0;
    this.featureImportanceMap.set(
      bestFeature,
      prevImportance + bestVarianceReduction * indices.length
    );

    const leftNode = this.buildTree(
      X,
      y,
      bestLeftIndices,
      depth + 1,
      availableFeatures
    );
    const rightNode = this.buildTree(
      X,
      y,
      bestRightIndices,
      depth + 1,
      availableFeatures
    );

    return {
      featureIndex: bestFeature,
      featureName: this.featureNames[bestFeature],
      threshold: bestThreshold,
      value: nodeMean,
      varianceReduction: bestVarianceReduction,
      left: leftNode,
      right: rightNode,
      samplesCount: indices.length,
    };
  }

  predict(x: number[]): number {
    if (!this.root) return 0;
    let node: DecisionTreeNode = this.root;

    while (node.left && node.right && node.featureIndex !== undefined && node.threshold !== undefined) {
      if (x[node.featureIndex] <= node.threshold) {
        node = node.left;
      } else {
        node = node.right;
      }
    }

    return node.value ?? 0;
  }

  predictBatch(X: number[][]): number[] {
    return X.map((row) => this.predict(row));
  }

  getFeatureImportances(): FeatureImportance[] {
    let total = 0;
    this.featureImportanceMap.forEach((val) => {
      total += val;
    });

    const safeTotal = total > 0 ? total : 1;

    return this.featureNames.map((name, i) => {
      const imp = (this.featureImportanceMap.get(i) || 0) / safeTotal;
      return {
        feature: name,
        label: name,
        importance: imp,
      };
    }).sort((a, b) => b.importance - a.importance);
  }
}

import { DecisionTreeRegressor } from './decisionTree.ts';
import { FeatureImportance } from './types.ts';

export class RandomForestRegressor {
  trees: DecisionTreeRegressor[] = [];
  numTrees: number;
  maxDepth: number;
  featureSubsampleRatio: number;
  featureNames: string[] = [];

  constructor(numTrees = 20, maxDepth = 6, featureSubsampleRatio = 0.7) {
    this.numTrees = numTrees;
    this.maxDepth = maxDepth;
    this.featureSubsampleRatio = featureSubsampleRatio;
  }

  train(X: number[][], y: number[], featureNames: string[]) {
    this.featureNames = featureNames;
    this.trees = [];
    const nSamples = X.length;
    if (nSamples === 0) return;
    const nFeatures = X[0].length;
    const numSubsampleFeatures = Math.max(3, Math.floor(nFeatures * this.featureSubsampleRatio));

    for (let t = 0; t < this.numTrees; t++) {
      // 1. Bootstrap sample with replacement
      const bootstrapX: number[][] = [];
      const bootstrapY: number[] = [];

      for (let s = 0; s < nSamples; s++) {
        const randIdx = Math.floor(Math.random() * nSamples);
        bootstrapX.push(X[randIdx]);
        bootstrapY.push(y[randIdx]);
      }

      // 2. Random feature subset
      const allFeatureIndices = Array.from({ length: nFeatures }, (_, i) => i);
      // Shuffle indices
      for (let i = allFeatureIndices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [allFeatureIndices[i], allFeatureIndices[j]] = [allFeatureIndices[j], allFeatureIndices[i]];
      }
      const selectedFeatures = allFeatureIndices.slice(0, numSubsampleFeatures);

      // 3. Train individual tree
      const tree = new DecisionTreeRegressor(this.maxDepth, 6);
      tree.train(bootstrapX, bootstrapY, featureNames, selectedFeatures);
      this.trees.push(tree);
    }
  }

  predict(x: number[]): { mean: number; std: number } {
    if (this.trees.length === 0) return { mean: 0, std: 0 };

    const preds = this.trees.map((tree) => tree.predict(x));
    let sum = 0;
    for (let p of preds) sum += p;
    const mean = sum / preds.length;

    let varSum = 0;
    for (let p of preds) {
      const diff = p - mean;
      varSum += diff * diff;
    }
    const std = Math.sqrt(varSum / preds.length);

    return { mean, std };
  }

  predictBatch(X: number[][]): number[] {
    return X.map((row) => this.predict(row).mean);
  }

  getFeatureImportances(): FeatureImportance[] {
    const totalImportance = new Map<string, number>();

    for (const tree of this.trees) {
      const treeImportances = tree.getFeatureImportances();
      for (const item of treeImportances) {
        const current = totalImportance.get(item.feature) || 0;
        totalImportance.set(item.feature, current + item.importance);
      }
    }

    let grandTotal = 0;
    totalImportance.forEach((val) => {
      grandTotal += val;
    });

    const safeTotal = grandTotal > 0 ? grandTotal : 1;

    return this.featureNames.map((name) => {
      const imp = (totalImportance.get(name) || 0) / safeTotal;
      return {
        feature: name,
        label: name,
        importance: imp,
      };
    }).sort((a, b) => b.importance - a.importance);
  }
}

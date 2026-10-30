import { PropertyRecord, TrainingConfig, Hyperparameters } from './types.ts';
import { extractFeatureVector, FEATURE_NAMES, detectOutliersIQR } from './preprocessing.ts';
import { LinearRegressionModel } from './linearRegression.ts';
import { DecisionTreeRegressor } from './decisionTree.ts';
import { RandomForestRegressor } from './randomForest.ts';
import { calculateMetrics } from './evaluation.ts';

export interface FoldMetric {
  foldNumber: number;
  valCount: number;
  r2: number;
  mae: number;
  rmse: number;
  mape: number;
}

export interface ModelCVResult {
  modelName: string;
  folds: FoldMetric[];
  meanR2: number;
  stdR2: number;
  meanMae: number;
  stdMae: number;
  meanRmse: number;
  stdRmse: number;
  meanMape: number;
  stdMape: number;
  ci95LowR2: number;
  ci95HighR2: number;
}

export interface FoldComparisonRow {
  foldNumber: number;
  foldLabel: string;
  valCount: number;
  rfR2: number;
  rfMae: number;
  rfRmse: number;
  linR2: number;
  linMae: number;
  linRmse: number;
  dtR2: number;
  dtMae: number;
  dtRmse: number;
}

export interface CrossValidationSuiteResult {
  k: number;
  models: Record<string, ModelCVResult>;
  foldRows: FoldComparisonRow[];
  totalTimeMs: number;
}

export function performKFoldCrossValidation(
  rawDataset: PropertyRecord[],
  config: TrainingConfig,
  hyperparams: Hyperparameters,
  kFolds = 5
): CrossValidationSuiteResult {
  const t0 = performance.now();

  let dataset = [...rawDataset];
  if (config.filterOutliers) {
    const { cleaned } = detectOutliersIQR(dataset, config.outlierThresholdIQR);
    dataset = cleaned;
  }

  const n = dataset.length;
  const foldSize = Math.floor(n / kFolds);

  // Initialize accumulators
  const rfFolds: FoldMetric[] = [];
  const linFolds: FoldMetric[] = [];
  const dtFolds: FoldMetric[] = [];
  const foldRows: FoldComparisonRow[] = [];

  for (let f = 0; f < kFolds; f++) {
    const startIdx = f * foldSize;
    const endIdx = f === kFolds - 1 ? n : (f + 1) * foldSize;

    const valSubset = dataset.slice(startIdx, endIdx);
    const trainSubset = [...dataset.slice(0, startIdx), ...dataset.slice(endIdx)];

    // Vectorize features
    const X_train_raw = trainSubset.map(extractFeatureVector);
    const y_train = trainSubset.map((d) => (config.logTransformTarget ? Math.log(d.salePrice) : d.salePrice));

    const X_val_raw = valSubset.map(extractFeatureVector);
    const y_val = valSubset.map((d) => (config.logTransformTarget ? Math.log(d.salePrice) : d.salePrice));

    // Standardize using train statistics to prevent leakage
    const numFeatures = FEATURE_NAMES.length;
    const means: number[] = new Array(numFeatures).fill(0);
    const stds: number[] = new Array(numFeatures).fill(1);

    if (config.standardizeFeatures && X_train_raw.length > 0) {
      for (let j = 0; j < numFeatures; j++) {
        let sum = 0;
        for (let i = 0; i < X_train_raw.length; i++) sum += X_train_raw[i][j];
        means[j] = sum / X_train_raw.length;

        let varSum = 0;
        for (let i = 0; i < X_train_raw.length; i++) {
          const diff = X_train_raw[i][j] - means[j];
          varSum += diff * diff;
        }
        const s = Math.sqrt(varSum / X_train_raw.length);
        stds[j] = s > 1e-6 ? s : 1;
      }
    }

    const scale = (v: number[]) => {
      if (!config.standardizeFeatures) return v;
      return v.map((val, idx) => (val - means[idx]) / stds[idx]);
    };

    const X_train = X_train_raw.map(scale);
    const X_val = X_val_raw.map(scale);

    // 1. Train Ridge Linear Regression
    const lin = new LinearRegressionModel();
    lin.train(X_train, y_train, FEATURE_NAMES, hyperparams.linearRidgeLambda);
    const linPreds = lin.predictBatch(X_val);
    const linMetrics = calculateMetrics(y_val, linPreds);

    linFolds.push({
      foldNumber: f + 1,
      valCount: valSubset.length,
      r2: parseFloat(linMetrics.r2.toFixed(4)),
      mae: Math.round(linMetrics.mae),
      rmse: Math.round(linMetrics.rmse),
      mape: parseFloat(linMetrics.mape.toFixed(2)),
    });

    // 2. Train Decision Tree
    const dt = new DecisionTreeRegressor(hyperparams.treeMaxDepth, hyperparams.treeMinSamplesSplit);
    dt.train(X_train, y_train, FEATURE_NAMES);
    const dtPreds = dt.predictBatch(X_val);
    const dtMetrics = calculateMetrics(y_val, dtPreds);

    dtFolds.push({
      foldNumber: f + 1,
      valCount: valSubset.length,
      r2: parseFloat(dtMetrics.r2.toFixed(4)),
      mae: Math.round(dtMetrics.mae),
      rmse: Math.round(dtMetrics.rmse),
      mape: parseFloat(dtMetrics.mape.toFixed(2)),
    });

    // 3. Train Random Forest (using fewer trees for fast responsive CV execution)
    const cvTrees = Math.min(14, hyperparams.forestNumTrees);
    const rf = new RandomForestRegressor(cvTrees, hyperparams.forestMaxDepth, hyperparams.forestFeatureSubsampleRatio);
    rf.train(X_train, y_train, FEATURE_NAMES);
    const rfPreds = rf.predictBatch(X_val);
    const rfMetrics = calculateMetrics(y_val, rfPreds);

    rfFolds.push({
      foldNumber: f + 1,
      valCount: valSubset.length,
      r2: parseFloat(rfMetrics.r2.toFixed(4)),
      mae: Math.round(rfMetrics.mae),
      rmse: Math.round(rfMetrics.rmse),
      mape: parseFloat(rfMetrics.mape.toFixed(2)),
    });

    foldRows.push({
      foldNumber: f + 1,
      foldLabel: `Fold ${f + 1}`,
      valCount: valSubset.length,
      rfR2: parseFloat((rfMetrics.r2 * 100).toFixed(2)),
      rfMae: Math.round(rfMetrics.mae),
      rfRmse: Math.round(rfMetrics.rmse),
      linR2: parseFloat((linMetrics.r2 * 100).toFixed(2)),
      linMae: Math.round(linMetrics.mae),
      linRmse: Math.round(linMetrics.rmse),
      dtR2: parseFloat((dtMetrics.r2 * 100).toFixed(2)),
      dtMae: Math.round(dtMetrics.mae),
      dtRmse: Math.round(dtMetrics.rmse),
    });
  }

  const buildSummary = (name: string, folds: FoldMetric[]): ModelCVResult => {
    const k = folds.length;
    const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / k;
    const std = (arr: number[], m: number) =>
      Math.sqrt(arr.reduce((a, b) => a + Math.pow(b - m, 2), 0) / (k > 1 ? k - 1 : 1));

    const r2s = folds.map((f) => f.r2);
    const maes = folds.map((f) => f.mae);
    const rmses = folds.map((f) => f.rmse);
    const mapes = folds.map((f) => f.mape);

    const meanR2 = mean(r2s);
    const stdR2 = std(r2s, meanR2);

    const meanMae = mean(maes);
    const stdMae = std(maes, meanMae);

    const meanRmse = mean(rmses);
    const stdRmse = std(rmses, meanRmse);

    const meanMape = mean(mapes);
    const stdMape = std(mapes, meanMape);

    // 95% Confidence Interval
    const margin = 1.96 * (stdR2 / Math.sqrt(k));

    return {
      modelName: name,
      folds,
      meanR2: parseFloat(meanR2.toFixed(4)),
      stdR2: parseFloat(stdR2.toFixed(4)),
      meanMae: Math.round(meanMae),
      stdMae: Math.round(stdMae),
      meanRmse: Math.round(meanRmse),
      stdRmse: Math.round(stdRmse),
      meanMape: parseFloat(meanMape.toFixed(2)),
      stdMape: parseFloat(stdMape.toFixed(2)),
      ci95LowR2: parseFloat(Math.max(0, meanR2 - margin).toFixed(4)),
      ci95HighR2: parseFloat(Math.min(1, meanR2 + margin).toFixed(4)),
    };
  };

  const t1 = performance.now();

  return {
    k: kFolds,
    models: {
      'Random Forest Regressor': buildSummary('Random Forest Regressor', rfFolds),
      'Linear Regression (Ridge)': buildSummary('Linear Regression (Ridge)', linFolds),
      'Decision Tree Regressor': buildSummary('Decision Tree Regressor', dtFolds),
    },
    foldRows,
    totalTimeMs: Math.round(t1 - t0),
  };
}

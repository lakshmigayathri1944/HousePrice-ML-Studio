export interface PropertyRecord {
  id: string;
  sqftLiving: number;
  bedrooms: number;
  bathrooms: number;
  lotSizeSqft: number;
  yearBuilt: number;
  yearRemodeled: number;
  overallQuality: number; // 1 to 10
  overallCondition: number; // 1 to 10
  neighborhood: string;
  garageCars: number;
  distanceToCenterKm: number;
  waterfront: boolean;
  hasBasement: boolean;
  hasPool: boolean;
  salePrice: number;
}

export type Neighborhood =
  | 'Westside Heights'
  | 'Downtown Urban'
  | 'Oakridge Suburbs'
  | 'Riverside Vista'
  | 'Pine Crest'
  | 'Sunset Valley'
  | 'Lakeview Terrace';

export interface PreprocessedData {
  featureNames: string[];
  rawTrain: PropertyRecord[];
  rawTest: PropertyRecord[];
  X_train: number[][];
  y_train: number[];
  X_test: number[][];
  y_test: number[];
  means: number[];
  stds: number[];
}

export interface TrainingConfig {
  trainRatio: number; // e.g. 0.8
  randomSeed: number;
  filterOutliers: boolean;
  outlierThresholdIQR: number; // e.g. 1.5
  standardizeFeatures: boolean;
  missingValueStrategy: 'median' | 'mean' | 'drop';
  logTransformTarget: boolean;
}

export interface ModelMetrics {
  mae: number;
  rmse: number;
  r2: number;
  mape: number; // percentage
  trainMae: number;
  trainRmse: number;
  trainR2: number;
}

export interface PredictionPoint {
  id: string;
  actual: number;
  predicted: number;
  residual: number;
  neighborhood: string;
  sqftLiving: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number; // 0 to 1
  label: string;
}

export interface ModelResult {
  name: string;
  type: 'linear' | 'decisionTree' | 'randomForest';
  metrics: ModelMetrics;
  predictions: PredictionPoint[];
  featureImportances: FeatureImportance[];
  trainingTimeMs: number;
}

export interface DecisionTreeNode {
  featureIndex?: number;
  featureName?: string;
  threshold?: number;
  value?: number;
  left?: DecisionTreeNode;
  right?: DecisionTreeNode;
  samplesCount: number;
  varianceReduction?: number;
}

export interface Hyperparameters {
  linearRidgeLambda: number;
  treeMaxDepth: number;
  treeMinSamplesSplit: number;
  forestNumTrees: number;
  forestMaxDepth: number;
  forestFeatureSubsampleRatio: number;
}

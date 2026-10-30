import { PropertyRecord, PreprocessedData, TrainingConfig, Neighborhood } from './types.ts';
import { NEIGHBORHOODS } from './dataset.ts';

// Deterministic shuffle using seed
function seededShuffle<T>(array: T[], seed: number): T[] {
  const arr = [...array];
  let m = arr.length;
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;

  const nextRand = () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };

  while (m) {
    const i = Math.floor(nextRand() * m--);
    const t = arr[m];
    arr[m] = arr[i];
    arr[i] = t;
  }
  return arr;
}

export function detectOutliersIQR(
  data: PropertyRecord[],
  threshold = 1.5
): { cleaned: PropertyRecord[]; outliers: PropertyRecord[] } {
  if (data.length < 10) return { cleaned: data, outliers: [] };

  const prices = data.map((d) => d.salePrice).sort((a, b) => a - b);
  const q1Index = Math.floor(prices.length * 0.25);
  const q3Index = Math.floor(prices.length * 0.75);
  const q1 = prices[q1Index];
  const q3 = prices[q3Index];
  const iqr = q3 - q1;
  const lowerBound = Math.max(50000, q1 - threshold * iqr);
  const upperBound = q3 + threshold * iqr;

  const cleaned: PropertyRecord[] = [];
  const outliers: PropertyRecord[] = [];

  for (const item of data) {
    if (item.salePrice >= lowerBound && item.salePrice <= upperBound) {
      cleaned.push(item);
    } else {
      outliers.push(item);
    }
  }

  return { cleaned, outliers };
}

// Convert a single property into a flat numerical feature vector
export function extractFeatureVector(record: PropertyRecord): number[] {
  const vector: number[] = [
    record.sqftLiving,
    record.bedrooms,
    record.bathrooms,
    record.lotSizeSqft,
    record.yearBuilt,
    record.yearRemodeled,
    record.overallQuality,
    record.overallCondition,
    record.garageCars,
    record.distanceToCenterKm,
    record.waterfront ? 1 : 0,
    record.hasBasement ? 1 : 0,
    record.hasPool ? 1 : 0,
  ];

  // One-hot encode neighborhoods
  for (const n of NEIGHBORHOODS) {
    vector.push(record.neighborhood === n.name ? 1 : 0);
  }

  return vector;
}

export const FEATURE_NAMES: string[] = [
  'SqFt Living',
  'Bedrooms',
  'Bathrooms',
  'Lot Size SqFt',
  'Year Built',
  'Year Remodeled',
  'Overall Quality',
  'Overall Condition',
  'Garage Cars',
  'Distance Center (km)',
  'Waterfront',
  'Has Basement',
  'Has Pool',
  ...NEIGHBORHOODS.map((n) => `Neighborhood: ${n.name}`),
];

export function preprocessDataset(
  rawDataset: PropertyRecord[],
  config: TrainingConfig
): PreprocessedData {
  let dataset = [...rawDataset];

  // 1. Outlier filtering
  if (config.filterOutliers) {
    const { cleaned } = detectOutliersIQR(dataset, config.outlierThresholdIQR);
    dataset = cleaned;
  }

  // 2. Train/Test shuffle & split
  const shuffled = seededShuffle(dataset, config.randomSeed);
  const splitIndex = Math.floor(shuffled.length * config.trainRatio);

  const rawTrain = shuffled.slice(0, splitIndex);
  const rawTest = shuffled.slice(splitIndex);

  // 3. Extract raw matrices
  const X_train_raw = rawTrain.map(extractFeatureVector);
  const y_train_raw = rawTrain.map((d) => (config.logTransformTarget ? Math.log(d.salePrice) : d.salePrice));

  const X_test_raw = rawTest.map(extractFeatureVector);
  const y_test_raw = rawTest.map((d) => (config.logTransformTarget ? Math.log(d.salePrice) : d.salePrice));

  const numFeatures = FEATURE_NAMES.length;
  const means: number[] = new Array(numFeatures).fill(0);
  const stds: number[] = new Array(numFeatures).fill(1);

  // 4. Calculate means and stds on train set (to prevent data leakage)
  if (config.standardizeFeatures && X_train_raw.length > 0) {
    for (let f = 0; f < numFeatures; f++) {
      let sum = 0;
      for (let i = 0; i < X_train_raw.length; i++) {
        sum += X_train_raw[i][f];
      }
      means[f] = sum / X_train_raw.length;

      let varSum = 0;
      for (let i = 0; i < X_train_raw.length; i++) {
        const diff = X_train_raw[i][f] - means[f];
        varSum += diff * diff;
      }
      const stdev = Math.sqrt(varSum / X_train_raw.length);
      stds[f] = stdev > 1e-6 ? stdev : 1;
    }
  }

  // 5. Apply scaling
  const scaleVector = (v: number[]) => {
    if (!config.standardizeFeatures) return v;
    return v.map((val, idx) => (val - means[idx]) / stds[idx]);
  };

  const X_train = X_train_raw.map(scaleVector);
  const X_test = X_test_raw.map(scaleVector);

  return {
    featureNames: FEATURE_NAMES,
    rawTrain,
    rawTest,
    X_train,
    y_train: y_train_raw,
    X_test,
    y_test: y_test_raw,
    means,
    stds,
  };
}

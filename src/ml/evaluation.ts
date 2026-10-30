import { ModelMetrics, PredictionPoint, PropertyRecord } from './types.ts';

export function calculateMetrics(actuals: number[], predictions: number[]): {
  mae: number;
  rmse: number;
  r2: number;
  mape: number;
} {
  const n = actuals.length;
  if (n === 0) return { mae: 0, rmse: 0, r2: 0, mape: 0 };

  let sumActual = 0;
  for (let i = 0; i < n; i++) sumActual += actuals[i];
  const meanActual = sumActual / n;

  let sumAbsErr = 0;
  let sumSqErr = 0;
  let sumTotSq = 0;
  let sumPctErr = 0;

  for (let i = 0; i < n; i++) {
    const act = actuals[i];
    const pred = predictions[i];
    const err = pred - act;
    const absErr = Math.abs(err);

    sumAbsErr += absErr;
    sumSqErr += err * err;
    sumTotSq += (act - meanActual) * (act - meanActual);

    if (act !== 0) {
      sumPctErr += Math.abs(err / act);
    }
  }

  const mae = sumAbsErr / n;
  const rmse = Math.sqrt(sumSqErr / n);
  const r2 = sumTotSq > 0 ? 1 - sumSqErr / sumTotSq : 0;
  const mape = (sumPctErr / n) * 100;

  return { mae, rmse, r2, mape };
}

export function buildPredictionPoints(
  rawRecords: PropertyRecord[],
  actuals: number[],
  predictions: number[],
  isLogScale = false
): PredictionPoint[] {
  return rawRecords.map((rec, i) => {
    let act = actuals[i];
    let pred = predictions[i];

    if (isLogScale) {
      act = Math.exp(act);
      pred = Math.exp(pred);
    }

    return {
      id: rec.id,
      actual: Math.round(act),
      predicted: Math.round(pred),
      residual: Math.round(pred - act),
      neighborhood: rec.neighborhood,
      sqftLiving: rec.sqftLiving,
    };
  });
}

import { FeatureImportance } from './types.ts';

// Matrix operations for multivariate regression with L2 Ridge regularizer
function transpose(A: number[][]): number[][] {
  const rows = A.length;
  const cols = A[0].length;
  const res: number[][] = Array.from({ length: cols }, () => new Array(rows).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      res[j][i] = A[i][j];
    }
  }
  return res;
}

function matMul(A: number[][], B: number[][]): number[][] {
  const rowsA = A.length;
  const colsA = A[0].length;
  const colsB = B[0].length;
  const res: number[][] = Array.from({ length: rowsA }, () => new Array(colsB).fill(0));

  for (let i = 0; i < rowsA; i++) {
    for (let k = 0; k < colsA; k++) {
      const aik = A[i][k];
      for (let j = 0; j < colsB; j++) {
        res[i][j] += aik * B[k][j];
      }
    }
  }
  return res;
}

// Invert matrix using Gauss-Jordan with partial pivoting
function invertMatrix(M: number[][]): number[][] | null {
  const n = M.length;
  const A: number[][] = M.map((row) => [...row]);
  const I: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );

  for (let i = 0; i < n; i++) {
    // Find pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) {
        maxRow = k;
      }
    }

    if (Math.abs(A[maxRow][i]) < 1e-12) {
      return null; // Singular matrix
    }

    // Swap rows
    const tempA = A[i];
    A[i] = A[maxRow];
    A[maxRow] = tempA;

    const tempI = I[i];
    I[i] = I[maxRow];
    I[maxRow] = tempI;

    // Normalize pivot row
    const pivot = A[i][i];
    for (let j = 0; j < n; j++) {
      A[i][j] /= pivot;
      I[i][j] /= pivot;
    }

    // Eliminate other rows
    for (let k = 0; k < n; k++) {
      if (k !== i) {
        const factor = A[k][i];
        for (let j = 0; j < n; j++) {
          A[k][j] -= factor * A[i][j];
          I[k][j] -= factor * I[i][j];
        }
      }
    }
  }

  return I;
}

export class LinearRegressionModel {
  weights: number[] = [];
  intercept = 0;
  featureNames: string[] = [];

  train(X: number[][], y: number[], featureNames: string[], ridgeLambda = 1.0) {
    this.featureNames = featureNames;
    const n = X.length;
    if (n === 0) return;
    const p = X[0].length;

    // Append 1s column for intercept: X_bias = [1, x1, x2, ...]
    const X_bias: number[][] = X.map((row) => [1, ...row]);
    const Xt = transpose(X_bias); // (p+1) x n
    const XtX = matMul(Xt, X_bias); // (p+1) x (p+1)

    // Add Ridge L2 penalty lambda * I (don't penalize intercept index 0)
    for (let j = 1; j <= p; j++) {
      XtX[j][j] += ridgeLambda;
    }

    const XtX_inv = invertMatrix(XtX);
    if (!XtX_inv) {
      console.warn('Matrix inversion fallback');
      this.weights = new Array(p).fill(0);
      this.intercept = y.reduce((a, b) => a + b, 0) / n;
      return;
    }

    // y_mat as n x 1
    const y_mat: number[][] = y.map((val) => [val]);
    const XtY = matMul(Xt, y_mat); // (p+1) x 1
    const beta = matMul(XtX_inv, XtY); // (p+1) x 1

    this.intercept = beta[0][0];
    this.weights = beta.slice(1).map((row) => row[0]);
  }

  predict(x: number[]): number {
    let pred = this.intercept;
    for (let i = 0; i < this.weights.length; i++) {
      pred += (this.weights[i] || 0) * (x[i] || 0);
    }
    return pred;
  }

  predictBatch(X: number[][]): number[] {
    return X.map((row) => this.predict(row));
  }

  getFeatureImportances(): FeatureImportance[] {
    const absWeights = this.weights.map((w) => Math.abs(w));
    const maxWeight = Math.max(...absWeights, 1e-6);

    return this.featureNames.map((name, i) => ({
      feature: name,
      label: name,
      importance: (absWeights[i] || 0) / maxWeight,
    })).sort((a, b) => b.importance - a.importance);
  }
}

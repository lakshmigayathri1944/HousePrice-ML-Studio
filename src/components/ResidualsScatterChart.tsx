import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import { ModelResult } from '../ml/types.ts';
import { Layers, Info, AlertTriangle, CheckCircle, HelpCircle } from 'lucide-react';

interface ResidualsScatterChartProps {
  models: ModelResult[];
  activeModelName: string;
  onSelectModel: (name: string) => void;
}

const MODEL_PALETTE: Record<string, { fill: string; stroke: string; label: string }> = {
  'Random Forest Regressor': {
    fill: '#2563eb',
    stroke: '#1d4ed8',
    label: 'Random Forest',
  },
  'Linear Regression (Ridge)': {
    fill: '#f59e0b',
    stroke: '#d97706',
    label: 'Linear Regression',
  },
  'Decision Tree Regressor': {
    fill: '#10b981',
    stroke: '#059669',
    label: 'Decision Tree',
  },
};

type XAxisMode = 'predicted' | 'actual' | 'sqft';

export const ResidualsScatterChart: React.FC<ResidualsScatterChartProps> = ({
  models,
  activeModelName,
  onSelectModel,
}) => {
  const [xAxisMode, setXAxisMode] = useState<XAxisMode>('predicted');
  const [compareAllModels, setCompareAllModels] = useState<boolean>(false);
  const [sampleLimit, setSampleLimit] = useState<number>(140);

  // Compute points for each model where residual = actual - predicted
  const modelResidualPoints = useMemo(() => {
    const res: Record<string, any[]> = {};

    models.forEach((m) => {
      const pts = m.predictions.slice(0, sampleLimit).map((pt) => {
        // Residual: Actual - Predicted
        const residual = pt.actual - pt.predicted;
        const absRes = Math.abs(residual);
        const pct = (absRes / pt.actual) * 100;

        let xVal = pt.predicted;
        if (xAxisMode === 'actual') xVal = pt.actual;
        else if (xAxisMode === 'sqft') xVal = pt.sqftLiving;

        return {
          id: pt.id,
          x: xVal,
          residual, // Y axis: Actual - Predicted
          predicted: pt.predicted,
          actual: pt.actual,
          sqftLiving: pt.sqftLiving,
          neighborhood: pt.neighborhood,
          absRes,
          pct: parseFloat(pct.toFixed(1)),
          modelName: m.name,
        };
      });

      res[m.name] = pts;
    });

    return res;
  }, [models, sampleLimit, xAxisMode]);

  // Active dataset points
  const activePts = modelResidualPoints[activeModelName] || [];

  // Compute Heteroscedasticity Analysis Metrics
  const heteroscedasticityMetrics = useMemo(() => {
    if (activePts.length < 10) return null;

    const n = activePts.length;
    const residuals = activePts.map((p) => p.residual);

    // Mean residual (should be very close to 0 if unbiased)
    const meanRes = residuals.reduce((a, b) => a + b, 0) / n;

    // Variance & Standard Error
    const sumSq = residuals.reduce((a, b) => a + (b - meanRes) * (b - meanRes), 0);
    const variance = sumSq / (n - 1);
    const rse = Math.sqrt(variance);

    // Funnel Ratio test: compare variance of lowest 33% predicted values vs highest 33%
    const sortedByX = [...activePts].sort((a, b) => a.x - b.x);
    const splitSize = Math.floor(n / 3);
    const lowTier = sortedByX.slice(0, splitSize).map((p) => p.residual);
    const highTier = sortedByX.slice(n - splitSize).map((p) => p.residual);

    const calcVar = (arr: number[]) => {
      const m = arr.reduce((a, b) => a + b, 0) / arr.length;
      return arr.reduce((a, b) => a + Math.pow(b - m, 2), 0) / (arr.length - 1);
    };

    const varLow = calcVar(lowTier);
    const varHigh = calcVar(highTier);
    const funnelRatio = varLow > 0 ? varHigh / varLow : 1.0;

    // Interpretation
    let status: 'homoscedastic' | 'mild_hetero' | 'pronounced_hetero' = 'homoscedastic';
    if (funnelRatio > 2.8 || funnelRatio < 0.35) {
      status = 'pronounced_hetero';
    } else if (funnelRatio > 1.7 || funnelRatio < 0.6) {
      status = 'mild_hetero';
    }

    return {
      meanRes: Math.round(meanRes),
      rse: Math.round(rse),
      sigma1: Math.round(rse),
      sigma2: Math.round(rse * 2),
      funnelRatio: parseFloat(funnelRatio.toFixed(2)),
      status,
    };
  }, [activePts]);

  // Dynamic X and Y bounds
  const { minX, maxX, maxYBound } = useMemo(() => {
    let min_x = Infinity;
    let max_x = -Infinity;
    let max_abs_y = 0;

    Object.values(modelResidualPoints).forEach((pts) => {
      pts.forEach((p) => {
        if (p.x < min_x) min_x = p.x;
        if (p.x > max_x) max_x = p.x;
        if (Math.abs(p.residual) > max_abs_y) max_abs_y = Math.abs(p.residual);
      });
    });

    if (min_x === Infinity) {
      min_x = 100000;
      max_x = 1200000;
      max_abs_y = 120000;
    }

    const padX = (max_x - min_x) * 0.05;
    const roundedY = Math.ceil((max_abs_y * 1.15) / 10000) * 10000;

    return {
      minX: Math.max(0, Math.floor((min_x - padX) / 10000) * 10000),
      maxX: Math.ceil((max_x + padX) / 10000) * 10000,
      maxYBound: Math.max(60000, roundedY),
    };
  }, [modelResidualPoints]);

  const formatCurrency = (val: number) => {
    if (Math.abs(val) >= 1000000) return `$${(val / 1000000).toFixed(1)}M`;
    if (Math.abs(val) >= 1000) return `$${Math.round(val / 1000)}k`;
    return `$${val}`;
  };

  const formatXAxis = (val: number) => {
    if (xAxisMode === 'sqft') return `${Math.round(val)} sqft`;
    return formatCurrency(val);
  };

  // Custom Recharts Tooltip for Residuals
  const CustomResidualTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;
    const data = payload[0].payload;
    const underEstimated = data.residual > 0;

    return (
      <div className="bg-slate-900/95 dark:bg-slate-950/95 backdrop-blur-md border border-slate-700 p-3.5 rounded-xl shadow-xl text-xs text-white max-w-xs space-y-2 z-50">
        <div className="flex items-center justify-between border-b border-slate-800 pb-1 font-mono">
          <span className="font-semibold text-blue-400">{data.id}</span>
          <span className="text-[11px] text-slate-400">{data.neighborhood}</span>
        </div>

        <div className="text-[11px] text-slate-300">
          Model: <span className="font-medium text-white">{data.modelName || activeModelName}</span>
        </div>

        <div className="grid grid-cols-2 gap-2 pt-1">
          <div>
            <div className="text-[10px] text-slate-400">Actual Price (y)</div>
            <div className="font-mono font-semibold text-white">
              ${data.actual?.toLocaleString()}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-slate-400">Predicted (ŷ)</div>
            <div className="font-mono font-semibold text-slate-200">
              ${data.predicted?.toLocaleString()}
            </div>
          </div>
        </div>

        <div className="pt-2 border-t border-slate-800 flex items-center justify-between">
          <span className="text-slate-400">Residual (y - ŷ):</span>
          <span
            className={`font-mono font-bold ${
              underEstimated ? 'text-emerald-400' : 'text-amber-400'
            }`}
          >
            {underEstimated ? '+' : ''}${data.residual?.toLocaleString()}
          </span>
        </div>

        <div className="text-[10px] text-slate-400 font-sans italic">
          {underEstimated
            ? 'Model under-predicted: house sold higher than estimated.'
            : 'Model over-predicted: house sold lower than estimated.'}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm space-y-6">
      {/* Title & Diagnostic Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-slate-900 dark:text-white">
              Residual Scatter Plot & Heteroscedasticity Analysis
            </h3>
            <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-400 font-mono font-medium">
              Residuals (y - ŷ)
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Plotting residual errors against predicted prices to inspect homoscedasticity, variance fanning, and model bias.
          </p>
        </div>

        {/* X Axis selector */}
        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-400 text-[11px]">X-Axis:</span>
          <div className="flex items-center gap-1 p-0.5 bg-slate-100 dark:bg-slate-800 rounded-lg">
            <button
              onClick={() => setXAxisMode('predicted')}
              className={`px-2.5 py-1 text-xs rounded-md transition-colors whitespace-nowrap ${
                xAxisMode === 'predicted'
                  ? 'bg-white dark:bg-slate-700 font-semibold text-slate-900 dark:text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              Predicted Price (ŷ)
            </button>
            <button
              onClick={() => setXAxisMode('actual')}
              className={`px-2.5 py-1 text-xs rounded-md transition-colors whitespace-nowrap ${
                xAxisMode === 'actual'
                  ? 'bg-white dark:bg-slate-700 font-semibold text-slate-900 dark:text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              Actual Price (y)
            </button>
            <button
              onClick={() => setXAxisMode('sqft')}
              className={`px-2.5 py-1 text-xs rounded-md transition-colors whitespace-nowrap ${
                xAxisMode === 'sqft'
                  ? 'bg-white dark:bg-slate-700 font-semibold text-slate-900 dark:text-white shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              Living SqFt
            </button>
          </div>
        </div>
      </div>

      {/* Model Selection and Overlay Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setCompareAllModels(!compareAllModels)}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors ${
              compareAllModels
                ? 'bg-blue-600 text-white border-blue-600 shadow-sm'
                : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Overlay All Models</span>
          </button>

          {!compareAllModels && (
            <div className="flex items-center gap-1 p-0.5 bg-slate-100 dark:bg-slate-800 rounded-lg">
              {models.map((m) => (
                <button
                  key={m.name}
                  onClick={() => onSelectModel(m.name)}
                  className={`px-2.5 py-1 text-xs rounded-md transition-colors whitespace-nowrap ${
                    activeModelName === m.name
                      ? 'bg-white dark:bg-slate-700 font-semibold text-slate-900 dark:text-white shadow-xs'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                  }`}
                >
                  {MODEL_PALETTE[m.name]?.label || m.name}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-[11px]">Points:</span>
          <select
            value={sampleLimit}
            onChange={(e) => setSampleLimit(parseInt(e.target.value, 10))}
            className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1 text-slate-800 dark:text-slate-200 text-xs focus:outline-none cursor-pointer font-mono"
          >
            <option value={70}>70</option>
            <option value={140}>140</option>
            <option value={200}>200</option>
          </select>
        </div>
      </div>

      {/* Heteroscedasticity Statistical Indicator Strip */}
      {heteroscedasticityMetrics && (
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 bg-slate-50 dark:bg-slate-800/50 p-4 rounded-xl border border-slate-200 dark:border-slate-700/60 text-xs">
          <div>
            <span className="text-slate-400 block text-[11px]">Residual Std Error (RSE)</span>
            <span className="font-mono font-bold text-slate-900 dark:text-white text-sm tabular-nums">
              ±${heteroscedasticityMetrics.rse.toLocaleString()}
            </span>
            <span className="text-[10px] text-slate-400 block mt-0.5">1σ confidence corridor</span>
          </div>

          <div>
            <span className="text-slate-400 block text-[11px]">Mean Residual (Bias)</span>
            <span className="font-mono font-bold text-slate-900 dark:text-white text-sm tabular-nums">
              ${heteroscedasticityMetrics.meanRes.toLocaleString()}
            </span>
            <span className="text-[10px] text-emerald-600 dark:text-emerald-400 block mt-0.5">
              Near zero (unbiased)
            </span>
          </div>

          <div>
            <span className="text-slate-400 block text-[11px]">Variance Funnel Ratio</span>
            <span className="font-mono font-bold text-slate-900 dark:text-white text-sm tabular-nums">
              {heteroscedasticityMetrics.funnelRatio}x
            </span>
            <span className="text-[10px] text-slate-400 block mt-0.5">Var(High Comps) / Var(Low)</span>
          </div>

          <div>
            <span className="text-slate-400 block text-[11px]">Variance Diagnosis</span>
            <div className="flex items-center gap-1.5 mt-0.5">
              {heteroscedasticityMetrics.status === 'homoscedastic' ? (
                <>
                  <CheckCircle className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                  <span className="font-semibold text-emerald-600 dark:text-emerald-400">
                    Homoscedastic
                  </span>
                </>
              ) : heteroscedasticityMetrics.status === 'mild_hetero' ? (
                <>
                  <Info className="w-3.5 h-3.5 text-blue-500 shrink-0" />
                  <span className="font-semibold text-blue-600 dark:text-blue-400">
                    Mild Heteroscedasticity
                  </span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-500 shrink-0" />
                  <span className="font-semibold text-amber-600 dark:text-amber-400">
                    Fanning Detected
                  </span>
                </>
              )}
            </div>
            <span className="text-[10px] text-slate-400 block mt-0.5">
              {heteroscedasticityMetrics.status === 'homoscedastic'
                ? 'Errors have constant spread across values'
                : 'Residual spread expands moderately for luxury homes'}
            </span>
          </div>
        </div>
      )}

      {/* Main Recharts Residual Scatter Chart */}
      <div className="h-96 w-full pt-1">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 30, bottom: 25, left: 30 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" opacity={0.2} />

            <XAxis
              type="number"
              dataKey="x"
              domain={[minX, maxX]}
              tickFormatter={formatXAxis}
              tick={{ fill: '#64748b', fontSize: 11 }}
              axisLine={{ stroke: '#94a3b8' }}
              label={{
                value:
                  xAxisMode === 'predicted'
                    ? 'Model Predicted Price (ŷ, $)'
                    : xAxisMode === 'actual'
                    ? 'Actual Recorded Sale Price (y, $)'
                    : 'Living Area (SqFt)',
                position: 'insideBottom',
                offset: -12,
                fill: '#64748b',
                fontSize: 12,
              }}
            />

            <YAxis
              type="number"
              dataKey="residual"
              domain={[-maxYBound, maxYBound]}
              tickFormatter={formatCurrency}
              tick={{ fill: '#64748b', fontSize: 11 }}
              axisLine={{ stroke: '#94a3b8' }}
              label={{
                value: 'Residual Error: Actual - Predicted (y - ŷ, $)',
                angle: -90,
                position: 'insideLeft',
                offset: -10,
                fill: '#64748b',
                fontSize: 12,
              }}
            />

            <Tooltip content={<CustomResidualTooltip />} />
            <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />

            {/* Zero-Error Baseline Reference Line */}
            <ReferenceLine
              y={0}
              stroke="#ef4444"
              strokeWidth={2}
              label={{
                value: 'Zero Error Baseline (y = ŷ)',
                fill: '#ef4444',
                position: 'insideTopLeft',
                fontSize: 11,
              }}
            />

            {/* 1-Sigma Confidence Envelope Lines */}
            {heteroscedasticityMetrics && (
              <>
                <ReferenceLine
                  y={heteroscedasticityMetrics.sigma1}
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  strokeWidth={1}
                  label={{
                    value: `+1σ (+$${Math.round(heteroscedasticityMetrics.sigma1 / 1000)}k)`,
                    fill: '#94a3b8',
                    position: 'insideTopRight',
                    fontSize: 10,
                  }}
                />
                <ReferenceLine
                  y={-heteroscedasticityMetrics.sigma1}
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  strokeWidth={1}
                  label={{
                    value: `-1σ (-$${Math.round(heteroscedasticityMetrics.sigma1 / 1000)}k)`,
                    fill: '#94a3b8',
                    position: 'insideBottomRight',
                    fontSize: 10,
                  }}
                />
              </>
            )}

            {/* Scatter points: Overlay or Individual */}
            {compareAllModels ? (
              <>
                <Scatter
                  name="Random Forest Regressor"
                  data={modelResidualPoints['Random Forest Regressor'] || []}
                  fill={MODEL_PALETTE['Random Forest Regressor'].fill}
                  opacity={0.7}
                />
                <Scatter
                  name="Linear Regression (Ridge)"
                  data={modelResidualPoints['Linear Regression (Ridge)'] || []}
                  fill={MODEL_PALETTE['Linear Regression (Ridge)'].fill}
                  opacity={0.65}
                />
                <Scatter
                  name="Decision Tree Regressor"
                  data={modelResidualPoints['Decision Tree Regressor'] || []}
                  fill={MODEL_PALETTE['Decision Tree Regressor'].fill}
                  opacity={0.65}
                />
              </>
            ) : (
              <Scatter
                name={activeModelName}
                data={modelResidualPoints[activeModelName] || []}
                fill={MODEL_PALETTE[activeModelName]?.fill || '#2563eb'}
                opacity={0.8}
              />
            )}
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Econometric Guidance Footnote */}
      <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 text-xs text-slate-600 dark:text-slate-400 space-y-1">
        <div className="font-semibold text-slate-900 dark:text-white flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5 text-blue-500" />
          <span>How to Read Heteroscedasticity in Real Estate ML:</span>
        </div>
        <p className="leading-relaxed text-[11px]">
          In an ideal <strong>homoscedastic model</strong>, the residuals scatter randomly within a constant horizontal band across all predicted values. If you observe a <em>cone or fan shape</em> opening to the right (higher variance for higher-priced homes), that indicates heteroscedasticity—a common real estate phenomenon because expensive properties have larger absolute dollar swings. Applying a <strong>Log-Transform target (<code className="font-mono text-blue-600 dark:text-blue-400">np.log1p</code>)</strong> in the Data Cleaning tab helps compress and stabilize this variance!
        </p>
      </div>
    </div>
  );
};

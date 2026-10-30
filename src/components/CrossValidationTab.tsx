import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import { PropertyRecord, TrainingConfig, Hyperparameters } from '../ml/types.ts';
import {
  performKFoldCrossValidation,
  CrossValidationSuiteResult,
  ModelCVResult,
} from '../ml/crossValidation.ts';
import {
  RefreshCw,
  Trophy,
  Activity,
  ShieldCheck,
  Info,
  CheckCircle2,
  TrendingUp,
  Cpu,
} from 'lucide-react';

interface CrossValidationTabProps {
  dataset: PropertyRecord[];
  config: TrainingConfig;
  hyperparams: Hyperparameters;
}

type MetricMode = 'r2' | 'mae' | 'rmse';

const MODEL_STYLE = {
  rf: {
    name: 'Random Forest Regressor',
    label: 'Random Forest',
    stroke: '#2563eb',
    keyR2: 'rfR2',
    keyMae: 'rfMae',
    keyRmse: 'rfRmse',
  },
  lin: {
    name: 'Linear Regression (Ridge)',
    label: 'Linear Regression',
    stroke: '#f59e0b',
    keyR2: 'linR2',
    keyMae: 'linMae',
    keyRmse: 'linRmse',
  },
  dt: {
    name: 'Decision Tree Regressor',
    label: 'Decision Tree',
    stroke: '#10b981',
    keyR2: 'dtR2',
    keyMae: 'dtMae',
    keyRmse: 'dtRmse',
  },
};

export const CrossValidationTab: React.FC<CrossValidationTabProps> = ({
  dataset,
  config,
  hyperparams,
}) => {
  const [kFolds, setKFolds] = useState<number>(5);
  const [metricMode, setMetricMode] = useState<MetricMode>('r2');
  const [isRunning, setIsRunning] = useState<boolean>(false);

  // Cross-Validation Results State
  const [cvResult, setCvResult] = useState<CrossValidationSuiteResult>(() => {
    return performKFoldCrossValidation(dataset, config, hyperparams, 5);
  });

  const handleRunCV = () => {
    setIsRunning(true);
    setTimeout(() => {
      try {
        const res = performKFoldCrossValidation(dataset, config, hyperparams, kFolds);
        setCvResult(res);
      } catch (err) {
        console.error('Cross-validation execution error:', err);
      } finally {
        setIsRunning(false);
      }
    }, 50);
  };

  // Identify most stable model (lowest R2 standard deviation)
  const mostStableModel = useMemo(() => {
    const modelsArr = Object.values(cvResult.models);
    return [...modelsArr].sort((a, b) => a.stdR2 - b.stdR2)[0];
  }, [cvResult]);

  // Format currency
  const formatCurrency = (val: number) => {
    if (val >= 1000) return `$${Math.round(val / 1000)}k`;
    return `$${Math.round(val)}`;
  };

  const formatYAxis = (val: number) => {
    if (metricMode === 'r2') return `${val}%`;
    return formatCurrency(val);
  };

  // Custom Recharts Tooltip for Fold Trajectory
  const CustomCVTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;

    return (
      <div className="bg-slate-900/95 dark:bg-slate-950/95 backdrop-blur-md border border-slate-700 p-3.5 rounded-xl shadow-xl text-xs text-white max-w-xs space-y-2 z-50">
        <div className="flex items-center justify-between border-b border-slate-800 pb-1 font-semibold text-blue-400">
          <span>{label} Validation Result</span>
          <span className="text-[10px] text-slate-400 font-mono">
            K = {cvResult.k} Folds
          </span>
        </div>

        <div className="space-y-1.5 font-mono text-xs tabular-nums">
          {payload.map((item: any) => (
            <div key={item.name} className="flex justify-between items-center">
              <span className="flex items-center gap-1.5">
                <span
                  className="w-2.5 h-2.5 rounded-full inline-block"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-slate-300 font-sans">{item.name}:</span>
              </span>
              <span className="font-bold text-white">
                {metricMode === 'r2' ? `${item.value}%` : `$${Number(item.value).toLocaleString()}`}
              </span>
            </div>
          ))}
        </div>

        <div className="pt-2 border-t border-slate-800 text-[10px] text-slate-400 font-sans">
          Validation set size: ~{Math.round(dataset.length / cvResult.k)} properties
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Top Banner & Execution Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-bold text-slate-900 dark:text-white">
              K-Fold Cross-Validation Suite
            </h2>
            <span className="text-xs px-2 py-0.5 rounded-full bg-blue-50 dark:bg-blue-950 text-blue-600 dark:text-blue-400 font-mono font-medium">
              Out-of-Sample Stability
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Evaluate model resilience across non-overlapping data slices to prevent train/test split variance bias.
          </p>
        </div>

        {/* Folds Selector & Run Button */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 bg-slate-100 dark:bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-xs">
            <span className="text-slate-500 font-medium">Folds (K):</span>
            <select
              value={kFolds}
              onChange={(e) => setKFolds(parseInt(e.target.value, 10))}
              className="bg-transparent font-semibold text-slate-900 dark:text-white focus:outline-none cursor-pointer"
            >
              <option value={3}>3-Fold</option>
              <option value={5}>5-Fold (Standard)</option>
              <option value={8}>8-Fold</option>
              <option value={10}>10-Fold (Rigorous)</option>
            </select>
          </div>

          <button
            onClick={handleRunCV}
            disabled={isRunning}
            className="inline-flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-lg shadow-sm transition-colors whitespace-nowrap shrink-0 disabled:opacity-60"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isRunning ? 'animate-spin' : ''}`} />
            <span>{isRunning ? 'Evaluating Folds...' : `Run ${kFolds}-Fold CV`}</span>
          </button>
        </div>
      </div>

      {/* Cross-Validation Robust Model Leaderboard */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {Object.values(cvResult.models).map((model) => {
          const isStableWinner = mostStableModel && model.modelName === mostStableModel.modelName;

          return (
            <div
              key={model.modelName}
              className={`p-5 rounded-xl border bg-white dark:bg-slate-900 transition-all ${
                isStableWinner
                  ? 'border-blue-500/60 shadow-md ring-1 ring-blue-500/20'
                  : 'border-slate-200 dark:border-slate-800'
              }`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-1.5">
                    <span className="font-semibold text-sm text-slate-900 dark:text-white">
                      {model.modelName}
                    </span>
                    {isStableWinner && (
                      <span className="text-[10px] text-amber-600 dark:text-amber-400 font-semibold flex items-center gap-0.5">
                        <Trophy className="w-3 h-3 fill-current" /> Most Stable
                      </span>
                    )}
                  </div>
                  <span className="text-[11px] text-slate-400">
                    K = {cvResult.k} Folds (N = {Math.round(dataset.length / cvResult.k)} per fold)
                  </span>
                </div>

                <div className="text-right">
                  <div className="text-xl font-bold font-mono text-blue-600 dark:text-blue-400 tabular-nums">
                    {(model.meanR2 * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-slate-400 font-mono">
                    ±{(model.stdR2 * 100).toFixed(2)}% σ
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 mt-4 pt-3 border-t border-slate-100 dark:border-slate-800 text-xs font-mono tabular-nums">
                <div>
                  <div className="text-[10px] text-slate-400 font-sans">Mean MAE (± σ)</div>
                  <div className="font-medium text-slate-800 dark:text-slate-200">
                    ${model.meanMae.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-slate-400">
                    ±${model.stdMae.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-400 font-sans">Mean RMSE (± σ)</div>
                  <div className="font-medium text-slate-800 dark:text-slate-200">
                    ${model.meanRmse.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-slate-400">
                    ±${model.stdRmse.toLocaleString()}
                  </div>
                </div>
              </div>

              <div className="mt-3 pt-2.5 border-t border-slate-100 dark:border-slate-800/80 flex items-center justify-between text-[11px]">
                <span className="text-slate-500">95% Confidence Band:</span>
                <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">
                  {(model.ci95LowR2 * 100).toFixed(1)}% – {(model.ci95HighR2 * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Main Recharts Line Chart: Metrics Variance Across Folds */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-slate-900 dark:text-white">
                Metric Stability Across Folds (Recharts Trajectory)
              </h3>
              <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 font-mono">
                {cvResult.k} Folds
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Flat, tight lines indicate low variance and high generalization capability on unseen test slices.
            </p>
          </div>

          {/* Metric Selector Buttons */}
          <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg self-start sm:self-auto">
            <button
              onClick={() => setMetricMode('r2')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                metricMode === 'r2'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-semibold'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              R² Score (%)
            </button>
            <button
              onClick={() => setMetricMode('mae')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                metricMode === 'mae'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-semibold'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              MAE ($)
            </button>
            <button
              onClick={() => setMetricMode('rmse')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                metricMode === 'rmse'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-semibold'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              RMSE ($)
            </button>
          </div>
        </div>

        {/* Recharts Line Chart Viewport */}
        <div className="h-80 w-full pt-1">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={cvResult.foldRows}
              margin={{ top: 20, right: 30, bottom: 25, left: 25 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" opacity={0.2} />

              <XAxis
                dataKey="foldLabel"
                tick={{ fill: '#64748b', fontSize: 11 }}
                axisLine={{ stroke: '#94a3b8' }}
                label={{
                  value: 'Cross-Validation Folds',
                  position: 'insideBottom',
                  offset: -12,
                  fill: '#64748b',
                  fontSize: 12,
                }}
              />

              <YAxis
                tickFormatter={formatYAxis}
                tick={{ fill: '#64748b', fontSize: 11 }}
                domain={['auto', 'auto']}
                axisLine={{ stroke: '#94a3b8' }}
                label={{
                  value:
                    metricMode === 'r2'
                      ? 'R² Score (% Variance Explained)'
                      : metricMode === 'mae'
                      ? 'Mean Absolute Error ($)'
                      : 'Root Mean Squared Error ($)',
                  angle: -90,
                  position: 'insideLeft',
                  offset: -5,
                  fill: '#64748b',
                  fontSize: 12,
                }}
              />

              <Tooltip content={<CustomCVTooltip />} />
              <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />

              {/* Grand Mean Line for Random Forest */}
              {metricMode === 'r2' && (
                <ReferenceLine
                  y={parseFloat((cvResult.models['Random Forest Regressor'].meanR2 * 100).toFixed(2))}
                  stroke="#2563eb"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                  label={{
                    value: `RF CV Mean: ${(cvResult.models['Random Forest Regressor'].meanR2 * 100).toFixed(1)}%`,
                    fill: '#2563eb',
                    position: 'insideTopLeft',
                    fontSize: 10,
                  }}
                />
              )}

              {/* Lines for each algorithm */}
              <Line
                type="monotone"
                dataKey={
                  metricMode === 'r2'
                    ? MODEL_STYLE.rf.keyR2
                    : metricMode === 'mae'
                    ? MODEL_STYLE.rf.keyMae
                    : MODEL_STYLE.rf.keyRmse
                }
                name={MODEL_STYLE.rf.label}
                stroke={MODEL_STYLE.rf.stroke}
                strokeWidth={2.5}
                activeDot={{ r: 6 }}
                dot={{ r: 4 }}
              />

              <Line
                type="monotone"
                dataKey={
                  metricMode === 'r2'
                    ? MODEL_STYLE.lin.keyR2
                    : metricMode === 'mae'
                    ? MODEL_STYLE.lin.keyMae
                    : MODEL_STYLE.lin.keyRmse
                }
                name={MODEL_STYLE.lin.label}
                stroke={MODEL_STYLE.lin.stroke}
                strokeWidth={2}
                strokeDasharray="4 2"
                dot={{ r: 3.5 }}
              />

              <Line
                type="monotone"
                dataKey={
                  metricMode === 'r2'
                    ? MODEL_STYLE.dt.keyR2
                    : metricMode === 'mae'
                    ? MODEL_STYLE.dt.keyMae
                    : MODEL_STYLE.dt.keyRmse
                }
                name={MODEL_STYLE.dt.label}
                stroke={MODEL_STYLE.dt.stroke}
                strokeWidth={2}
                strokeDasharray="2 2"
                dot={{ r: 3.5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Footnote on Line Chart Observations */}
        <div className="flex items-center justify-between text-xs text-slate-500 pt-2 border-t border-slate-100 dark:border-slate-800">
          <div className="flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-blue-500 shrink-0" />
            <span>
              Low variance across folds confirms the model has not memorized particular random splits. Notice how bagging in Random Forest smooths out fold dips compared to single Decision Trees.
            </span>
          </div>
          <span className="font-mono text-slate-400 shrink-0">
            Computed in {cvResult.totalTimeMs}ms
          </span>
        </div>
      </div>

      {/* Fold-by-Fold Performance Breakdown Table */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm space-y-4">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
          Fold-by-Fold Cross-Validation Performance Matrix
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 dark:bg-slate-800/60 text-slate-600 dark:text-slate-400 font-medium border-b border-slate-200 dark:border-slate-700">
              <tr>
                <th className="py-2.5 px-3">Fold Iteration</th>
                <th className="py-2.5 px-3 text-right">Holdout Samples</th>
                <th className="py-2.5 px-3 text-right font-semibold text-blue-600 dark:text-blue-400">
                  Random Forest R²
                </th>
                <th className="py-2.5 px-3 text-right font-semibold text-amber-600 dark:text-amber-400">
                  Linear Ridge R²
                </th>
                <th className="py-2.5 px-3 text-right font-semibold text-emerald-600 dark:text-emerald-400">
                  Decision Tree R²
                </th>
                <th className="py-2.5 px-3 text-right">RF MAE ($)</th>
                <th className="py-2.5 px-3 text-right">Linear MAE ($)</th>
                <th className="py-2.5 px-3 text-right">Tree MAE ($)</th>
                <th className="py-2.5 px-3 text-right">Top Fold Model</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800 font-mono tabular-nums">
              {cvResult.foldRows.map((row) => {
                const maxR2 = Math.max(row.rfR2, row.linR2, row.dtR2);
                const winner =
                  maxR2 === row.rfR2
                    ? 'Random Forest'
                    : maxR2 === row.linR2
                    ? 'Linear Ridge'
                    : 'Decision Tree';

                return (
                  <tr
                    key={row.foldNumber}
                    className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors"
                  >
                    <td className="py-2 px-3 font-sans font-medium text-slate-900 dark:text-white">
                      {row.foldLabel}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-500">{row.valCount}</td>
                    <td className="py-2 px-3 text-right font-bold text-blue-600 dark:text-blue-400">
                      {row.rfR2}%
                    </td>
                    <td className="py-2 px-3 text-right text-slate-700 dark:text-slate-300">
                      {row.linR2}%
                    </td>
                    <td className="py-2 px-3 text-right text-slate-700 dark:text-slate-300">
                      {row.dtR2}%
                    </td>
                    <td className="py-2 px-3 text-right text-slate-600 dark:text-slate-400">
                      ${row.rfMae.toLocaleString()}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-600 dark:text-slate-400">
                      ${row.linMae.toLocaleString()}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-600 dark:text-slate-400">
                      ${row.dtMae.toLocaleString()}
                    </td>
                    <td className="py-2 px-3 text-right font-sans font-medium text-emerald-600 dark:text-emerald-400">
                      {winner}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

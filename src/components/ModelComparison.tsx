import React, { useState } from 'react';
import {
  Trophy,
  Sliders,
  CheckCircle2,
  Activity,
  Layers,
  Sparkles,
  Zap,
} from 'lucide-react';
import { ModelResult, Hyperparameters } from '../ml/types.ts';
import { PredictedVsActualChart } from './PredictedVsActualChart.tsx';
import { ResidualsScatterChart } from './ResidualsScatterChart.tsx';

interface ModelComparisonProps {
  models: ModelResult[];
  hyperparams: Hyperparameters;
  setHyperparams: React.Dispatch<React.SetStateAction<Hyperparameters>>;
  onRetrain: () => void;
  isTraining: boolean;
}

export const ModelComparison: React.FC<ModelComparisonProps> = ({
  models,
  hyperparams,
  setHyperparams,
  onRetrain,
  isTraining,
}) => {
  const [selectedModelName, setSelectedModelName] = useState<string>('Random Forest Regressor');

  // Identify best model by R2
  const bestModel = [...models].sort((a, b) => b.metrics.r2 - a.metrics.r2)[0];

  const currentModel = models.find((m) => m.name === selectedModelName) || models[0];

  return (
    <div className="space-y-6">
      {/* Header with Title and Retrain status */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 dark:text-white">
            Model Benchmarking & Performance Comparison
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Evaluate and compare Linear Regression, Decision Tree, and Random Forest models across MAE, RMSE, and R² scores.
          </p>
        </div>

        <button
          onClick={onRetrain}
          disabled={isTraining}
          className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-lg shadow-sm transition-colors whitespace-nowrap shrink-0 disabled:opacity-60"
        >
          <Zap className="w-3.5 h-3.5 fill-current" />
          <span>{isTraining ? 'Re-training Models...' : 'Retrain Benchmarks'}</span>
        </button>
      </div>

      {/* Model Benchmark Cards / Leaderboard */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {models.map((m) => {
          const isSelected = m.name === selectedModelName;
          const isWinner = bestModel && m.name === bestModel.name;

          return (
            <div
              key={m.name}
              onClick={() => setSelectedModelName(m.name)}
              className={`p-5 rounded-xl border cursor-pointer transition-all ${
                isSelected
                  ? 'border-blue-600 dark:border-blue-500 bg-blue-50/20 dark:bg-blue-950/20 shadow-md ring-1 ring-blue-600/20'
                  : 'border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:border-slate-300 dark:hover:border-slate-700'
              }`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-1.5">
                    <span className="font-semibold text-sm text-slate-900 dark:text-white">
                      {m.name}
                    </span>
                    {isWinner && (
                      <span className="text-[10px] text-amber-600 dark:text-amber-400 font-semibold flex items-center gap-0.5">
                        <Trophy className="w-3 h-3 fill-current" /> Top
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-slate-500 capitalize">{m.type} model</span>
                </div>
                <div className="text-right">
                  <span className="text-xl font-bold font-mono text-slate-900 dark:text-white tabular-nums">
                    {(m.metrics.r2 * 100).toFixed(1)}%
                  </span>
                  <div className="text-[10px] text-slate-400 font-medium">Test R² Score</div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 mt-4 pt-3 border-t border-slate-100 dark:border-slate-800 text-xs">
                <div>
                  <div className="text-[11px] text-slate-400">MAE (Test Error)</div>
                  <div className="font-mono font-medium text-slate-800 dark:text-slate-200 tabular-nums">
                    ${Math.round(m.metrics.mae).toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-slate-400">RMSE (Penalty Error)</div>
                  <div className="font-mono font-medium text-slate-800 dark:text-slate-200 tabular-nums">
                    ${Math.round(m.metrics.rmse).toLocaleString()}
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between mt-3 text-[11px] text-slate-500 font-mono">
                <span>MAPE: {m.metrics.mape.toFixed(1)}%</span>
                <span>Train time: {m.trainingTimeMs.toFixed(0)}ms</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Head-to-Head Performance Evaluation Table */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm space-y-4">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
          Head-to-Head Evaluation Metrics Comparison
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 dark:bg-slate-800/60 text-slate-600 dark:text-slate-400 font-medium border-b border-slate-200 dark:border-slate-700">
              <tr>
                <th className="py-2.5 px-3">Algorithm</th>
                <th className="py-2.5 px-3 text-right">Test R² (Variance Explained)</th>
                <th className="py-2.5 px-3 text-right">Test MAE ($)</th>
                <th className="py-2.5 px-3 text-right">Test RMSE ($)</th>
                <th className="py-2.5 px-3 text-right">MAPE (%)</th>
                <th className="py-2.5 px-3 text-right">Train R² (Fit)</th>
                <th className="py-2.5 px-3 text-right">Train/Test Gap</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800 font-mono tabular-nums">
              {models.map((m) => {
                const gap = Math.abs(m.metrics.trainR2 - m.metrics.r2);
                return (
                  <tr
                    key={m.name}
                    className={`hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors ${
                      m.name === selectedModelName ? 'bg-blue-50/40 dark:bg-blue-950/30' : ''
                    }`}
                  >
                    <td className="py-2.5 px-3 font-sans font-medium text-slate-900 dark:text-white flex items-center gap-2">
                      <span>{m.name}</span>
                      {bestModel && m.name === bestModel.name && (
                        <span className="text-[10px] text-amber-500 font-bold">★ Best</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold text-blue-600 dark:text-blue-400">
                      {(m.metrics.r2 * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-800 dark:text-slate-200">
                      ${Math.round(m.metrics.mae).toLocaleString()}
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-800 dark:text-slate-200">
                      ${Math.round(m.metrics.rmse).toLocaleString()}
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-600 dark:text-slate-400">
                      {m.metrics.mape.toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-600 dark:text-slate-400">
                      {(m.metrics.trainR2 * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span className={gap > 0.08 ? 'text-amber-500 font-medium' : 'text-slate-400'}>
                        {(gap * 100).toFixed(1)}% {gap > 0.08 ? '(overfit)' : ''}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recharts Predicted vs. Actual Interactive Visualizer */}
      <PredictedVsActualChart
        models={models}
        activeModelName={selectedModelName}
        onSelectModel={setSelectedModelName}
      />

      {/* Recharts Residual Scatter Plot (Actual minus Predicted) for Heteroscedasticity Analysis */}
      <ResidualsScatterChart
        models={models}
        activeModelName={selectedModelName}
        onSelectModel={setSelectedModelName}
      />

      {/* Hyperparameter Tuning Studio */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-blue-600 dark:text-blue-400" />
            <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
              Algorithm Hyperparameter Tuning
            </h3>
          </div>
          <span className="text-xs text-slate-500">Fine-tune model parameters to minimize generalization error</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Ridge Linear Regression params */}
          <div className="p-4 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 space-y-4">
            <h4 className="text-xs font-semibold text-slate-900 dark:text-white">
              Linear Regression (Ridge)
            </h4>
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-600 dark:text-slate-300">
                <span>Regularization Alpha (λ)</span>
                <span className="font-mono font-medium">{hyperparams.linearRidgeLambda}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="10.0"
                step="0.1"
                value={hyperparams.linearRidgeLambda}
                onChange={(e) =>
                  setHyperparams((prev) => ({ ...prev, linearRidgeLambda: parseFloat(e.target.value) }))
                }
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-[11px] text-slate-500">
                Controls L2 weight shrinkage to prevent collinear matrix singularity.
              </p>
            </div>
          </div>

          {/* Decision Tree Regressor params */}
          <div className="p-4 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 space-y-4">
            <h4 className="text-xs font-semibold text-slate-900 dark:text-white">
              Decision Tree Regressor
            </h4>
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-600 dark:text-slate-300">
                <span>Max Depth</span>
                <span className="font-mono font-medium">{hyperparams.treeMaxDepth} levels</span>
              </div>
              <input
                type="range"
                min="3"
                max="12"
                step="1"
                value={hyperparams.treeMaxDepth}
                onChange={(e) =>
                  setHyperparams((prev) => ({ ...prev, treeMaxDepth: parseInt(e.target.value, 10) }))
                }
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-600 dark:text-slate-300">
                <span>Min Samples Split</span>
                <span className="font-mono font-medium">{hyperparams.treeMinSamplesSplit}</span>
              </div>
              <input
                type="range"
                min="4"
                max="30"
                step="2"
                value={hyperparams.treeMinSamplesSplit}
                onChange={(e) =>
                  setHyperparams((prev) => ({
                    ...prev,
                    treeMinSamplesSplit: parseInt(e.target.value, 10),
                  }))
                }
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>

          {/* Random Forest Regressor params */}
          <div className="p-4 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 space-y-4">
            <h4 className="text-xs font-semibold text-slate-900 dark:text-white">
              Random Forest Regressor
            </h4>
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-600 dark:text-slate-300">
                <span>Estimators (Trees)</span>
                <span className="font-mono font-medium">{hyperparams.forestNumTrees}</span>
              </div>
              <input
                type="range"
                min="5"
                max="40"
                step="5"
                value={hyperparams.forestNumTrees}
                onChange={(e) =>
                  setHyperparams((prev) => ({ ...prev, forestNumTrees: parseInt(e.target.value, 10) }))
                }
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-600 dark:text-slate-300">
                <span>Feature Subsample Ratio</span>
                <span className="font-mono font-medium">
                  {Math.round(hyperparams.forestFeatureSubsampleRatio * 100)}%
                </span>
              </div>
              <input
                type="range"
                min="0.3"
                max="1.0"
                step="0.1"
                value={hyperparams.forestFeatureSubsampleRatio}
                onChange={(e) =>
                  setHyperparams((prev) => ({
                    ...prev,
                    forestFeatureSubsampleRatio: parseFloat(e.target.value),
                  }))
                }
                className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

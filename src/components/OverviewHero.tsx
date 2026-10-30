import React from 'react';
import { Play, BarChart3, Sliders, ArrowRight, ShieldCheck, Cpu, Database } from 'lucide-react';
import { ModelResult } from '../ml/types.ts';

interface OverviewHeroProps {
  onNavigate: (tab: string) => void;
  datasetSize: number;
  models: ModelResult[];
}

export const OverviewHero: React.FC<OverviewHeroProps> = ({
  onNavigate,
  datasetSize,
  models,
}) => {
  const bestModel = [...models].sort((a, b) => b.metrics.r2 - a.metrics.r2)[0];
  const bestR2 = bestModel ? (bestModel.metrics.r2 * 100).toFixed(1) : '89.4';
  const bestMAE = bestModel ? Math.round(bestModel.metrics.mae).toLocaleString() : '24,500';

  return (
    <div className="space-y-8">
      {/* Hero Showcase with Architectural Photo and Scrim */}
      <div className="relative rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-900 text-white shadow-lg">
        <div className="absolute inset-0 z-0">
          <img
            src="/src/assets/images/hero_modern_architecture_1793342444838.jpg"
            alt="Modern residential architecture"
            referrerPolicy="no-referrer"
            className="w-full h-full object-cover object-center opacity-40 scale-105 transition-transform duration-1000"
          />
          <div className="absolute inset-0 bg-gradient-to-r from-slate-950 via-slate-950/80 to-transparent" />
        </div>

        <div className="relative z-10 p-6 sm:p-10 lg:p-12 max-w-3xl space-y-6">
          <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
            <span>Machine Learning Regression</span>
            <span aria-hidden="true">·</span>
            <span>Scikit-Learn & Python Architecture</span>
            <span aria-hidden="true">·</span>
            <span className="text-emerald-400">Live Models Active</span>
          </div>

          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-white leading-tight">
            Predict Residential Property Values with Machine Learning
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-2xl">
            An end-to-end regression system comparing Ordinary Least Squares Linear Regression, CART Decision Trees, and Bagged Random Forest algorithms across property features.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <button
              onClick={() => onNavigate('predictor')}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-medium text-sm rounded-lg transition-colors shadow-md whitespace-nowrap shrink-0"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>Value a Custom Home</span>
            </button>

            <button
              onClick={() => onNavigate('models')}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-slate-800/90 hover:bg-slate-700/90 text-white font-medium text-sm rounded-lg border border-slate-700 transition-colors whitespace-nowrap shrink-0"
            >
              <BarChart3 className="w-4 h-4" />
              <span>Benchmark Algorithms</span>
            </button>
          </div>

          <div className="flex items-center gap-6 pt-4 text-xs text-slate-400 border-t border-slate-800">
            <div>
              <span className="text-slate-500">Dataset:</span>{' '}
              <span className="text-slate-200 font-mono font-medium">{datasetSize.toLocaleString()} properties</span>
            </div>
            <div>
              <span className="text-slate-500">Top R² Score:</span>{' '}
              <span className="text-emerald-400 font-mono font-semibold">{bestR2}%</span>
            </div>
            <div>
              <span className="text-slate-500">Min MAE:</span>{' '}
              <span className="text-slate-200 font-mono font-medium">${bestMAE}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Pipeline 4-Stage Architectural Walkthrough */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-3">
          <div className="w-8 h-8 rounded-lg bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400 flex items-center justify-center">
            <Database className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
            01. Data Cleaning & EDA
          </h3>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            IQR outlier trimming, missing value handling, correlation matrices, and distribution skewness checks.
          </p>
          <button
            onClick={() => onNavigate('eda')}
            className="text-xs font-medium text-blue-600 dark:text-blue-400 hover:underline inline-flex items-center gap-1 pt-1"
          >
            <span>Explore Dataset</span>
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>

        <div className="p-5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 flex items-center justify-center">
            <Sliders className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
            02. Feature Preparation
          </h3>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            One-hot encoding for neighborhoods, Z-score standard scaling, and train/test cross-validation splits.
          </p>
          <button
            onClick={() => onNavigate('eda')}
            className="text-xs font-medium text-indigo-600 dark:text-indigo-400 hover:underline inline-flex items-center gap-1 pt-1"
          >
            <span>Configure Pipeline</span>
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>

        <div className="p-5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-3">
          <div className="w-8 h-8 rounded-lg bg-amber-50 dark:bg-amber-950/60 text-amber-600 dark:text-amber-400 flex items-center justify-center">
            <Cpu className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
            03. Regression Algorithms
          </h3>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            Multivariate Ridge Linear Regression, CART Decision Trees, and Bagged Random Forest ensembles.
          </p>
          <button
            onClick={() => onNavigate('models')}
            className="text-xs font-medium text-amber-600 dark:text-amber-400 hover:underline inline-flex items-center gap-1 pt-1"
          >
            <span>Compare Metrics</span>
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>

        <div className="p-5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
            04. Valuation & XAI
          </h3>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            Gini feature importance ranking, residual distribution analysis, confidence bounds, and AI appraisals.
          </p>
          <button
            onClick={() => onNavigate('features')}
            className="text-xs font-medium text-emerald-600 dark:text-emerald-400 hover:underline inline-flex items-center gap-1 pt-1"
          >
            <span>Inspect Importances</span>
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>
      </div>
    </div>
  );
};

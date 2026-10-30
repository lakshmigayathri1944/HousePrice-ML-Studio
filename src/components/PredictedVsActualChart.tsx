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
  LineChart,
  Line,
  BarChart,
  Bar,
  Cell,
} from 'recharts';
import { ModelResult } from '../ml/types.ts';
import { Check, Filter, Layers, Info, TrendingUp, Maximize2 } from 'lucide-react';

interface PredictedVsActualChartProps {
  models: ModelResult[];
  activeModelName: string;
  onSelectModel: (name: string) => void;
}

const MODEL_COLORS: Record<string, { fill: string; stroke: string; label: string }> = {
  'Random Forest Regressor': {
    fill: '#2563eb', // blue-600
    stroke: '#1d4ed8',
    label: 'Random Forest',
  },
  'Linear Regression (Ridge)': {
    fill: '#f59e0b', // amber-500
    stroke: '#d97706',
    label: 'Linear Regression',
  },
  'Decision Tree Regressor': {
    fill: '#10b981', // emerald-500
    stroke: '#059669',
    label: 'Decision Tree',
  },
};

type ViewMode = 'scatter' | 'sortedCurve' | 'residualSpread';
type PriceFilter = 'all' | 'entry' | 'mid' | 'luxury';

export const PredictedVsActualChart: React.FC<PredictedVsActualChartProps> = ({
  models,
  activeModelName,
  onSelectModel,
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('scatter');
  const [compareAllModels, setCompareAllModels] = useState<boolean>(true);
  const [priceFilter, setPriceFilter] = useState<PriceFilter>('all');
  const [selectedNeighborhood, setSelectedNeighborhood] = useState<string>('all');
  const [sampleLimit, setSampleLimit] = useState<number>(120);

  // Extract unique neighborhoods from test data
  const neighborhoods = useMemo(() => {
    const set = new Set<string>();
    models.forEach((m) => {
      m.predictions.forEach((p) => {
        if (p.neighborhood) set.add(p.neighborhood);
      });
    });
    return Array.from(set).sort();
  }, [models]);

  // Filter test sample points based on user selections
  const filteredPredictionsByModel = useMemo(() => {
    const res: Record<string, any[]> = {};

    models.forEach((model) => {
      let pts = model.predictions;

      // Price filter
      if (priceFilter === 'entry') {
        pts = pts.filter((p) => p.actual < 380000);
      } else if (priceFilter === 'mid') {
        pts = pts.filter((p) => p.actual >= 380000 && p.actual <= 720000);
      } else if (priceFilter === 'luxury') {
        pts = pts.filter((p) => p.actual > 720000);
      }

      // Neighborhood filter
      if (selectedNeighborhood !== 'all') {
        pts = pts.filter((p) => p.neighborhood === selectedNeighborhood);
      }

      // Limit samples to maintain smooth rendering
      pts = pts.slice(0, sampleLimit);

      res[model.name] = pts.map((pt) => {
        const error = pt.predicted - pt.actual;
        const absError = Math.abs(error);
        const errorPct = (absError / pt.actual) * 100;
        return {
          id: pt.id,
          actual: pt.actual,
          predicted: pt.predicted,
          residual: error,
          absError,
          errorPct: parseFloat(errorPct.toFixed(1)),
          neighborhood: pt.neighborhood,
          sqftLiving: pt.sqftLiving,
          modelName: model.name,
        };
      });
    });

    return res;
  }, [models, priceFilter, selectedNeighborhood, sampleLimit]);

  // Dataset for Sorted Valuation Curve (where actual price is ordered ascending)
  const sortedCurveData = useMemo(() => {
    const rfPts = filteredPredictionsByModel['Random Forest Regressor'] || [];
    const linPts = filteredPredictionsByModel['Linear Regression (Ridge)'] || [];
    const dtPts = filteredPredictionsByModel['Decision Tree Regressor'] || [];

    // Map by property id
    const linMap = new Map(linPts.map((p) => [p.id, p.predicted]));
    const dtMap = new Map(dtPts.map((p) => [p.id, p.predicted]));

    const combined = rfPts.map((pt) => ({
      id: pt.id,
      actual: pt.actual,
      randomForest: pt.predicted,
      linearRegression: linMap.get(pt.id) || pt.predicted,
      decisionTree: dtMap.get(pt.id) || pt.predicted,
      neighborhood: pt.neighborhood,
      sqftLiving: pt.sqftLiving,
    }));

    return combined.sort((a, b) => a.actual - b.actual);
  }, [filteredPredictionsByModel]);

  // Overall min and max bounds for coordinate axes
  const { minVal, maxVal } = useMemo(() => {
    let min = Infinity;
    let max = -Infinity;

    Object.values(filteredPredictionsByModel).forEach((pts) => {
      pts.forEach((p) => {
        if (p.actual < min) min = p.actual;
        if (p.actual > max) max = p.actual;
        if (p.predicted < min) min = p.predicted;
        if (p.predicted > max) max = p.predicted;
      });
    });

    if (min === Infinity) return { minVal: 100000, maxVal: 1200000 };

    const padding = (max - min) * 0.05;
    return {
      minVal: Math.max(50000, Math.floor((min - padding) / 50000) * 50000),
      maxVal: Math.ceil((max + padding) / 50000) * 50000,
    };
  }, [filteredPredictionsByModel]);

  // Inspection stats for current view
  const activeModelPts = filteredPredictionsByModel[activeModelName] || [];
  const currentViewStats = useMemo(() => {
    if (activeModelPts.length === 0) return null;
    const n = activeModelPts.length;
    const sumAbs = activeModelPts.reduce((acc, p) => acc + p.absError, 0);
    const meanAbs = sumAbs / n;

    // Within 10% threshold
    const within10Pct = activeModelPts.filter((p) => p.errorPct <= 10).length;
    const accuracyRate = (within10Pct / n) * 100;

    const maxOver = Math.max(...activeModelPts.map((p) => p.residual));
    const maxUnder = Math.min(...activeModelPts.map((p) => p.residual));

    return {
      count: n,
      mae: Math.round(meanAbs),
      accuracyRate: accuracyRate.toFixed(1),
      maxOver: Math.round(maxOver),
      maxUnder: Math.round(maxUnder),
    };
  }, [activeModelPts]);

  // Format currency
  const formatCurrency = (val: number) => {
    if (val >= 1000000) return `$${(val / 1000000).toFixed(1)}M`;
    if (val >= 1000) return `$${Math.round(val / 1000)}k`;
    return `$${val}`;
  };

  // Custom Scatter Tooltip
  const CustomScatterTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;
    const data = payload[0].payload;
    const isOver = data.residual > 0;

    return (
      <div className="bg-slate-900/95 dark:bg-slate-950/95 backdrop-blur-md border border-slate-700 p-3 rounded-xl shadow-xl text-xs text-white max-w-xs space-y-1.5 z-50">
        <div className="flex items-center justify-between border-b border-slate-700/80 pb-1 font-mono">
          <span className="font-semibold text-blue-400">{data.id}</span>
          <span className="text-[11px] text-slate-400">{data.neighborhood}</span>
        </div>

        <div className="text-[11px] text-slate-300">
          Model: <span className="font-medium text-white">{data.modelName || activeModelName}</span>
        </div>

        <div className="grid grid-cols-2 gap-2 pt-1">
          <div>
            <div className="text-[10px] text-slate-400">Actual Price</div>
            <div className="font-mono font-semibold text-white">
              ${data.actual?.toLocaleString()}
            </div>
          </div>
          <div>
            <div className="text-[10px] text-slate-400">Predicted Price</div>
            <div className="font-mono font-semibold text-blue-400">
              ${data.predicted?.toLocaleString()}
            </div>
          </div>
        </div>

        <div className="pt-1 border-t border-slate-800 flex items-center justify-between text-[11px]">
          <span className="text-slate-400">Prediction Variance:</span>
          <span className={`font-mono font-medium ${isOver ? 'text-amber-400' : 'text-emerald-400'}`}>
            {isOver ? '+' : ''}${Math.round(data.residual)?.toLocaleString()} ({data.errorPct}%)
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm space-y-6">
      {/* Header and Controls Strip */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-slate-900 dark:text-white">
              Predicted vs. Actual Price Diagnostic Visualizer
            </h3>
            <span className="text-xs px-2 py-0.5 rounded-full bg-blue-50 dark:bg-blue-950 text-blue-600 dark:text-blue-400 font-mono font-medium">
              Recharts Engine
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Evaluate empirical accuracy, variance dispersion, and deviation from the ideal 45° regression line ($y = x$).
          </p>
        </div>

        {/* View Mode Switcher */}
        <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg self-start lg:self-auto">
          <button
            onClick={() => setViewMode('scatter')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              viewMode === 'scatter'
                ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-semibold'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Scatter (45° Diagonal)
          </button>
          <button
            onClick={() => setViewMode('sortedCurve')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              viewMode === 'sortedCurve'
                ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-semibold'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Ordered Valuation Lines
          </button>
          <button
            onClick={() => setViewMode('residualSpread')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              viewMode === 'residualSpread'
                ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-semibold'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Residual Error Spread
          </button>
        </div>
      </div>

      {/* Filter and Model Selector Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
        {/* Model Selection Tabs / Overlay Toggle */}
        <div className="flex flex-wrap items-center gap-2">
          {viewMode === 'scatter' && (
            <button
              onClick={() => setCompareAllModels(!compareAllModels)}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors ${
                compareAllModels
                  ? 'bg-blue-600 text-white border-blue-600 shadow-sm'
                  : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Overlay All 3 Models</span>
            </button>
          )}

          {(!compareAllModels || viewMode !== 'scatter') && (
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
                  {MODEL_COLORS[m.name]?.label || m.name}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Filter dropdowns */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Price Range Filter */}
          <div className="flex items-center gap-1 bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1">
            <span className="text-slate-400 text-[11px]">Price:</span>
            <select
              value={priceFilter}
              onChange={(e) => setPriceFilter(e.target.value as PriceFilter)}
              className="bg-transparent text-slate-800 dark:text-slate-200 text-xs focus:outline-none cursor-pointer"
            >
              <option value="all">All Values</option>
              <option value="entry">Starter (&lt; $380k)</option>
              <option value="mid">Mid-Tier ($380k - $720k)</option>
              <option value="luxury">Luxury (&gt; $720k)</option>
            </select>
          </div>

          {/* Neighborhood Filter */}
          <div className="flex items-center gap-1 bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1">
            <span className="text-slate-400 text-[11px]">Area:</span>
            <select
              value={selectedNeighborhood}
              onChange={(e) => setSelectedNeighborhood(e.target.value)}
              className="bg-transparent text-slate-800 dark:text-slate-200 text-xs focus:outline-none cursor-pointer"
            >
              <option value="all">All Neighborhoods</option>
              {neighborhoods.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>

          {/* Sample Size */}
          <div className="flex items-center gap-1 bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1">
            <span className="text-slate-400 text-[11px]">Points:</span>
            <select
              value={sampleLimit}
              onChange={(e) => setSampleLimit(parseInt(e.target.value, 10))}
              className="bg-transparent text-slate-800 dark:text-slate-200 text-xs focus:outline-none cursor-pointer font-mono"
            >
              <option value={60}>60</option>
              <option value={120}>120</option>
              <option value={200}>200</option>
            </select>
          </div>
        </div>
      </div>

      {/* Accuracy Performance Metrics Banner */}
      {currentViewStats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-50 dark:bg-slate-800/50 p-3.5 rounded-lg border border-slate-200 dark:border-slate-700/60 text-xs">
          <div>
            <span className="text-slate-400 block text-[11px]">Viewport Mean Error</span>
            <span className="font-mono font-bold text-slate-900 dark:text-white tabular-nums">
              ${currentViewStats.mae.toLocaleString()}
            </span>
          </div>
          <div>
            <span className="text-slate-400 block text-[11px]">Within ±10% Accuracy</span>
            <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400 tabular-nums">
              {currentViewStats.accuracyRate}%
            </span>
          </div>
          <div>
            <span className="text-slate-400 block text-[11px]">Max Over-Prediction</span>
            <span className="font-mono font-medium text-amber-600 dark:text-amber-400 tabular-nums">
              +${currentViewStats.maxOver.toLocaleString()}
            </span>
          </div>
          <div>
            <span className="text-slate-400 block text-[11px]">Max Under-Prediction</span>
            <span className="font-mono font-medium text-blue-600 dark:text-blue-400 tabular-nums">
              -${Math.abs(currentViewStats.maxUnder).toLocaleString()}
            </span>
          </div>
        </div>
      )}

      {/* Main Chart Container */}
      <div className="h-96 w-full pt-2">
        {/* VIEW 1: SCATTER PLOT WITH 45 DEGREE LINE */}
        {viewMode === 'scatter' && (
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 30, bottom: 25, left: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" opacity={0.2} />

              <XAxis
                type="number"
                dataKey="actual"
                name="Actual Price"
                domain={[minVal, maxVal]}
                tickFormatter={formatCurrency}
                tick={{ fill: '#64748b', fontSize: 11 }}
                axisLine={{ stroke: '#94a3b8' }}
                label={{
                  value: 'Actual Recorded Sale Price ($)',
                  position: 'insideBottom',
                  offset: -12,
                  fill: '#64748b',
                  fontSize: 12,
                }}
              />

              <YAxis
                type="number"
                dataKey="predicted"
                name="Predicted Price"
                domain={[minVal, maxVal]}
                tickFormatter={formatCurrency}
                tick={{ fill: '#64748b', fontSize: 11 }}
                axisLine={{ stroke: '#94a3b8' }}
                label={{
                  value: 'Model Predicted Valuation ($)',
                  angle: -90,
                  position: 'insideLeft',
                  offset: 0,
                  fill: '#64748b',
                  fontSize: 12,
                }}
              />

              <Tooltip content={<CustomScatterTooltip />} />
              <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />

              {/* 45 Degree Perfect Line: segment from minVal to maxVal */}
              <ReferenceLine
                segment={[
                  { x: minVal, y: minVal },
                  { x: maxVal, y: maxVal },
                ]}
                stroke="#ef4444"
                strokeWidth={1.8}
                strokeDasharray="4 4"
                label={{
                  value: 'Ideal y = x (Zero Error Line)',
                  fill: '#ef4444',
                  position: 'insideTopLeft',
                  fontSize: 11,
                }}
              />

              {/* Multi-model Scatters or Single Model Scatter */}
              {compareAllModels ? (
                <>
                  <Scatter
                    name="Random Forest Regressor"
                    data={filteredPredictionsByModel['Random Forest Regressor'] || []}
                    fill={MODEL_COLORS['Random Forest Regressor'].fill}
                    opacity={0.7}
                  />
                  <Scatter
                    name="Linear Regression (Ridge)"
                    data={filteredPredictionsByModel['Linear Regression (Ridge)'] || []}
                    fill={MODEL_COLORS['Linear Regression (Ridge)'].fill}
                    opacity={0.65}
                  />
                  <Scatter
                    name="Decision Tree Regressor"
                    data={filteredPredictionsByModel['Decision Tree Regressor'] || []}
                    fill={MODEL_COLORS['Decision Tree Regressor'].fill}
                    opacity={0.65}
                  />
                </>
              ) : (
                <Scatter
                  name={activeModelName}
                  data={filteredPredictionsByModel[activeModelName] || []}
                  fill={MODEL_COLORS[activeModelName]?.fill || '#2563eb'}
                  opacity={0.8}
                />
              )}
            </ScatterChart>
          </ResponsiveContainer>
        )}

        {/* VIEW 2: ORDERED VALUATION CURVE */}
        {viewMode === 'sortedCurve' && (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sortedCurveData} margin={{ top: 20, right: 30, bottom: 25, left: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" opacity={0.2} />
              <XAxis
                dataKey="id"
                tick={false}
                axisLine={{ stroke: '#94a3b8' }}
                label={{
                  value: `Properties Ordered by Actual Price (Low to High, N = ${sortedCurveData.length})`,
                  position: 'insideBottom',
                  offset: -10,
                  fill: '#64748b',
                  fontSize: 12,
                }}
              />
              <YAxis
                tickFormatter={formatCurrency}
                tick={{ fill: '#64748b', fontSize: 11 }}
                domain={['auto', 'auto']}
                axisLine={{ stroke: '#94a3b8' }}
                label={{
                  value: 'Price ($)',
                  angle: -90,
                  position: 'insideLeft',
                  offset: 0,
                  fill: '#64748b',
                  fontSize: 12,
                }}
              />
              <Tooltip
                formatter={(value: any, name: any) => [
                  `$${Number(value).toLocaleString()}`,
                  name === 'actual'
                    ? 'Ground Truth (Actual)'
                    : name === 'randomForest'
                    ? 'Random Forest'
                    : name === 'linearRegression'
                    ? 'Linear Regression'
                    : 'Decision Tree',
                ]}
                labelFormatter={(id) => `Property ${id}`}
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '0.75rem',
                  fontSize: '11px',
                  color: '#fff',
                }}
              />
              <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />

              {/* Actual price smooth curve */}
              <Line
                type="monotone"
                dataKey="actual"
                name="Actual Recorded Price"
                stroke="#0f172a"
                strokeWidth={2.5}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="randomForest"
                name="Random Forest"
                stroke="#2563eb"
                strokeWidth={1.8}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="linearRegression"
                name="Linear Regression"
                stroke="#f59e0b"
                strokeWidth={1.5}
                strokeDasharray="4 2"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="decisionTree"
                name="Decision Tree"
                stroke="#10b981"
                strokeWidth={1.5}
                strokeDasharray="2 2"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}

        {/* VIEW 3: RESIDUAL SPREAD BAR CHART */}
        {viewMode === 'residualSpread' && (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={filteredPredictionsByModel[activeModelName] || []}
              margin={{ top: 20, right: 30, bottom: 25, left: 25 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" opacity={0.2} />
              <XAxis
                dataKey="id"
                tick={false}
                axisLine={{ stroke: '#94a3b8' }}
                label={{
                  value: `Test Properties (${activeModelName})`,
                  position: 'insideBottom',
                  offset: -10,
                  fill: '#64748b',
                  fontSize: 12,
                }}
              />
              <YAxis
                tickFormatter={formatCurrency}
                tick={{ fill: '#64748b', fontSize: 11 }}
                axisLine={{ stroke: '#94a3b8' }}
                label={{
                  value: 'Residual Error (Predicted - Actual, $)',
                  angle: -90,
                  position: 'insideLeft',
                  offset: 0,
                  fill: '#64748b',
                  fontSize: 12,
                }}
              />
              <ReferenceLine y={0} stroke="#ef4444" strokeWidth={1.5} />
              <Tooltip
                formatter={(val: any) => [`$${Number(val).toLocaleString()}`, 'Residual Error']}
                labelFormatter={(id) => `Property ${id}`}
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '0.75rem',
                  fontSize: '11px',
                  color: '#fff',
                }}
              />
              <Bar dataKey="residual" name="Residual Error ($)">
                {(filteredPredictionsByModel[activeModelName] || []).map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.residual >= 0 ? '#3b82f6' : '#f59e0b'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Explanatory Footnote */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 pt-3 border-t border-slate-100 dark:border-slate-800 text-xs text-slate-500">
        <div className="flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5 text-blue-500 shrink-0" />
          <span>
            {viewMode === 'scatter' &&
              'Tight clustering around the red dashed 45° line demonstrates high goodness-of-fit with minimal heteroscedastic spread.'}
            {viewMode === 'sortedCurve' &&
              'Tracks the predictive tracking response across price ranks: notice how Random Forest smooths variance better than single Decision Trees.'}
            {viewMode === 'residualSpread' &&
              'Blue bars represent over-valuations and amber bars indicate under-valuations. Symmetrical distribution denotes unbiased estimator performance.'}
          </span>
        </div>
        <div className="font-mono text-slate-400 shrink-0">
          Showing {activeModelPts.length} test comparisons
        </div>
      </div>
    </div>
  );
};

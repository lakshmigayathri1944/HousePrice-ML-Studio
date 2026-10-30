import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  Cell,
} from 'recharts';
import { PropertyRecord } from '../ml/types.ts';
import { NEIGHBORHOODS } from '../ml/dataset.ts';
import { Sliders, AlertTriangle, Info, CheckCircle2, Layers } from 'lucide-react';

interface FeatureDistributionVisualizerProps {
  dataset: PropertyRecord[];
}

type TargetFeature = 'salePrice' | 'sqftLiving' | 'lotSizeSqft' | 'yearBuilt' | 'overallQuality';

interface FeatureConfig {
  key: TargetFeature;
  label: string;
  unit: string;
  isCurrency: boolean;
  numBins: number;
}

const FEATURE_CONFIGS: FeatureConfig[] = [
  {
    key: 'salePrice',
    label: 'Sale Price',
    unit: '$',
    isCurrency: true,
    numBins: 20,
  },
  {
    key: 'sqftLiving',
    label: 'Living Area',
    unit: 'sq ft',
    isCurrency: false,
    numBins: 18,
  },
  {
    key: 'lotSizeSqft',
    label: 'Lot Size',
    unit: 'sq ft',
    isCurrency: false,
    numBins: 18,
  },
  {
    key: 'yearBuilt',
    label: 'Year Built',
    unit: 'year',
    isCurrency: false,
    numBins: 16,
  },
  {
    key: 'overallQuality',
    label: 'Overall Quality',
    unit: '/10',
    isCurrency: false,
    numBins: 8,
  },
];

export const FeatureDistributionVisualizer: React.FC<FeatureDistributionVisualizerProps> = ({
  dataset,
}) => {
  const [activeFeatureKey, setActiveFeatureKey] = useState<TargetFeature>('salePrice');
  const [viewType, setViewType] = useState<'histogram' | 'boxPlot'>('histogram');
  const [iqrMultiplier, setIqrMultiplier] = useState<number>(1.5);

  const activeConfig =
    FEATURE_CONFIGS.find((f) => f.key === activeFeatureKey) || FEATURE_CONFIGS[0];

  // Calculate 5-number summary & outliers for active feature
  const summaryStats = useMemo(() => {
    if (dataset.length === 0) return null;

    const values = dataset.map((d) => Number(d[activeConfig.key])).sort((a, b) => a - b);
    const n = values.length;
    const sum = values.reduce((a, b) => a + b, 0);
    const mean = sum / n;

    const q1 = values[Math.floor(n * 0.25)];
    const median = values[Math.floor(n * 0.5)];
    const q3 = values[Math.floor(n * 0.75)];
    const iqr = q3 - q1;

    const lowerBound = q1 - iqrMultiplier * iqr;
    const upperBound = q3 + iqrMultiplier * iqr;

    const min = values[0];
    const max = values[n - 1];

    // Standard deviation and Pearson skewness coefficient
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
    const std = Math.sqrt(variance);
    const skewness = std > 0 ? (3 * (mean - median)) / std : 0;

    // Outliers
    const outliers = values.filter((v) => v < lowerBound || v > upperBound);

    return {
      n,
      mean,
      std,
      min,
      q1,
      median,
      q3,
      max,
      iqr,
      lowerBound,
      upperBound,
      skewness: parseFloat(skewness.toFixed(2)),
      outlierCount: outliers.length,
      outlierPct: parseFloat(((outliers.length / n) * 100).toFixed(1)),
    };
  }, [dataset, activeConfig, iqrMultiplier]);

  // Format values according to unit
  const formatVal = (val: number) => {
    if (activeConfig.isCurrency) {
      if (Math.abs(val) >= 1000000) return `$${(val / 1000000).toFixed(1)}M`;
      if (Math.abs(val) >= 1000) return `$${Math.round(val / 1000)}k`;
      return `$${Math.round(val)}`;
    }
    if (activeConfig.key === 'yearBuilt') return `${Math.round(val)}`;
    if (val >= 10000) return `${Math.round(val / 1000)}k`;
    return `${Math.round(val).toLocaleString()}`;
  };

  // Recharts Histogram Bins Data
  const histogramData = useMemo(() => {
    if (!summaryStats || dataset.length === 0) return [];

    const values = dataset.map((d) => Number(d[activeConfig.key]));
    const minVal = summaryStats.min;
    const maxVal = summaryStats.max;
    const numBins = activeConfig.numBins;
    const step = (maxVal - minVal) / numBins;

    const bins = Array.from({ length: numBins }, (_, i) => {
      const start = minVal + i * step;
      const end = minVal + (i + 1) * step;
      return {
        binIndex: i,
        start,
        end,
        mid: (start + end) / 2,
        label: `${formatVal(start)} - ${formatVal(end)}`,
        count: 0,
        isOutlierBin: false,
      };
    });

    values.forEach((v) => {
      const idx = Math.min(numBins - 1, Math.max(0, Math.floor((v - minVal) / step)));
      bins[idx].count++;
    });

    // Flag outlier bins
    return bins.map((b) => {
      const isOutlier =
        b.end < summaryStats.lowerBound || b.start > summaryStats.upperBound;
      return {
        ...b,
        isOutlierBin: isOutlier && b.count > 0,
        pct: parseFloat(((b.count / dataset.length) * 100).toFixed(1)),
      };
    });
  }, [dataset, activeConfig, summaryStats]);

  // Recharts Box Plot Breakdown Data by Neighborhood
  const neighborhoodBoxPlotData = useMemo(() => {
    if (dataset.length === 0) return [];

    return NEIGHBORHOODS.map((n) => {
      const subset = dataset
        .filter((d) => d.neighborhood === n.name)
        .map((d) => Number(d[activeConfig.key]))
        .sort((a, b) => a - b);

      if (subset.length === 0) {
        return {
          neighborhood: n.name,
          min: 0,
          q1: 0,
          median: 0,
          q3: 0,
          max: 0,
          outliers: 0,
          count: 0,
        };
      }

      const len = subset.length;
      const q1 = subset[Math.floor(len * 0.25)];
      const median = subset[Math.floor(len * 0.5)];
      const q3 = subset[Math.floor(len * 0.75)];
      const iqr = q3 - q1;
      const lower = q1 - iqrMultiplier * iqr;
      const upper = q3 + iqrMultiplier * iqr;

      const outliers = subset.filter((v) => v < lower || v > upper).length;

      // Whiskers (min and max values within normal IQR fence)
      const nonOutliers = subset.filter((v) => v >= lower && v <= upper);
      const minWhisker = nonOutliers.length ? nonOutliers[0] : subset[0];
      const maxWhisker = nonOutliers.length ? nonOutliers[nonOutliers.length - 1] : subset[len - 1];

      return {
        neighborhood: n.name,
        min: minWhisker,
        bottomWhiskerSpan: q1 - minWhisker,
        q1,
        boxSpan: q3 - q1, // Interquartile range height
        median,
        q3,
        topWhiskerSpan: maxWhisker - q3,
        max: maxWhisker,
        outliers,
        count: len,
      };
    });
  }, [dataset, activeConfig, iqrMultiplier]);

  // Custom Histogram Tooltip
  const CustomHistogramTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;
    const data = payload[0].payload;

    return (
      <div className="bg-slate-900/95 dark:bg-slate-950/95 backdrop-blur-md border border-slate-700 p-3.5 rounded-xl shadow-xl text-xs text-white max-w-xs space-y-1.5 z-50">
        <div className="flex items-center justify-between border-b border-slate-800 pb-1 font-mono">
          <span className="font-semibold text-blue-400">{activeConfig.label} Interval</span>
          {data.isOutlierBin && (
            <span className="text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/40 px-1.5 py-0.5 rounded font-sans font-medium flex items-center gap-1">
              <AlertTriangle className="w-3 h-3 text-amber-400" />
              <span>Outlier Fence</span>
            </span>
          )}
        </div>

        <div className="font-mono text-sm font-bold text-white pt-1">
          {data.label}
        </div>

        <div className="grid grid-cols-2 gap-2 pt-1">
          <div>
            <div className="text-[10px] text-slate-400">Frequency</div>
            <div className="font-mono font-semibold text-blue-400">
              {data.count} properties
            </div>
          </div>
          <div>
            <div className="text-[10px] text-slate-400">Percentage</div>
            <div className="font-mono font-semibold text-slate-200">
              {data.pct}%
            </div>
          </div>
        </div>

        <div className="text-[10px] text-slate-400 pt-1 border-t border-slate-800">
          {data.isOutlierBin
            ? 'Values falling in this bin sit beyond the IQR threshold and are flagged for potential data cleaning or transformation.'
            : 'Values inside the normal distribution IQR boundary.'}
        </div>
      </div>
    );
  };

  // Custom Box Plot Tooltip
  const CustomBoxPlotTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;
    const data = payload[0].payload;

    return (
      <div className="bg-slate-900/95 dark:bg-slate-950/95 backdrop-blur-md border border-slate-700 p-3.5 rounded-xl shadow-xl text-xs text-white max-w-xs space-y-2 z-50">
        <div className="flex items-center justify-between border-b border-slate-800 pb-1 font-semibold text-blue-400">
          <span>{data.neighborhood}</span>
          <span className="text-[11px] text-slate-400 font-mono font-normal">
            N = {data.count}
          </span>
        </div>

        <div className="space-y-1 font-mono text-xs tabular-nums">
          <div className="flex justify-between">
            <span className="text-slate-400">Upper Whisker (Max):</span>
            <span className="text-white">{formatVal(data.max)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Q3 (75th Percentile):</span>
            <span className="text-white">{formatVal(data.q3)}</span>
          </div>
          <div className="flex justify-between font-bold text-blue-400">
            <span className="text-slate-400 font-normal">Median (50th):</span>
            <span>{formatVal(data.median)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Q1 (25th Percentile):</span>
            <span className="text-white">{formatVal(data.q1)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Lower Whisker (Min):</span>
            <span className="text-white">{formatVal(data.min)}</span>
          </div>
        </div>

        <div className="pt-2 border-t border-slate-800 flex justify-between items-center text-[11px]">
          <span className="text-slate-400">Outliers ({iqrMultiplier}x IQR):</span>
          <span
            className={`font-mono font-semibold ${
              data.outliers > 0 ? 'text-amber-400' : 'text-emerald-400'
            }`}
          >
            {data.outliers} properties
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm space-y-6">
      {/* Header and Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-slate-900 dark:text-white">
              Recharts Feature Distribution & Outlier Visualizer
            </h3>
            <span className="text-xs px-2 py-0.5 rounded-full bg-blue-50 dark:bg-blue-950 text-blue-600 dark:text-blue-400 font-mono font-medium">
              Recharts
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Identify skewness, distribution density, and potential outliers across core housing features using IQR boundaries.
          </p>
        </div>

        {/* View Mode Switcher */}
        <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg self-start lg:self-auto">
          <button
            onClick={() => setViewType('histogram')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              viewType === 'histogram'
                ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-semibold'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Histogram & Outlier Bins
          </button>
          <button
            onClick={() => setViewType('boxPlot')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              viewType === 'boxPlot'
                ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-semibold'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Box Plot Spread by Area
          </button>
        </div>
      </div>

      {/* Feature Selector Tabs & Outlier Sensitivity Slider */}
      <div className="flex flex-wrap items-center justify-between gap-4 text-xs">
        {/* Feature selection tabs */}
        <div className="flex flex-wrap items-center gap-1.5">
          {FEATURE_CONFIGS.map((f) => (
            <button
              key={f.key}
              onClick={() => setActiveFeatureKey(f.key)}
              className={`px-3 py-1.5 rounded-lg border text-xs transition-colors whitespace-nowrap shrink-0 ${
                activeFeatureKey === f.key
                  ? 'bg-blue-600 text-white border-blue-600 font-semibold shadow-xs'
                  : 'bg-slate-50 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* IQR Sensitivity Slider */}
        <div className="flex items-center gap-2 bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5">
          <span className="text-slate-500 font-medium text-[11px]">Outlier Fence:</span>
          <span className="font-mono font-semibold text-slate-900 dark:text-white text-xs">
            {iqrMultiplier}x IQR
          </span>
          <input
            type="range"
            min="1.0"
            max="3.0"
            step="0.1"
            value={iqrMultiplier}
            onChange={(e) => setIqrMultiplier(parseFloat(e.target.value))}
            className="w-20 h-1 bg-slate-300 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>

      {/* Summary Statistics Strip */}
      {summaryStats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3 bg-slate-50 dark:bg-slate-800/50 p-3.5 rounded-xl border border-slate-200 dark:border-slate-700/60 text-xs">
          <div>
            <span className="text-slate-400 block text-[11px]">Median (Q2)</span>
            <span className="font-mono font-bold text-blue-600 dark:text-blue-400 text-sm tabular-nums">
              {formatVal(summaryStats.median)}
            </span>
          </div>
          <div>
            <span className="text-slate-400 block text-[11px]">Mean Average</span>
            <span className="font-mono font-bold text-slate-900 dark:text-white text-sm tabular-nums">
              {formatVal(summaryStats.mean)}
            </span>
          </div>
          <div>
            <span className="text-slate-400 block text-[11px]">IQR (Q3 - Q1)</span>
            <span className="font-mono font-medium text-slate-800 dark:text-slate-200 text-sm tabular-nums">
              {formatVal(summaryStats.iqr)}
            </span>
          </div>
          <div>
            <span className="text-slate-400 block text-[11px]">Upper Outlier Fence</span>
            <span className="font-mono font-medium text-amber-600 dark:text-amber-400 text-sm tabular-nums">
              &gt; {formatVal(summaryStats.upperBound)}
            </span>
          </div>
          <div>
            <span className="text-slate-400 block text-[11px]">Detected Outliers</span>
            <span className="font-mono font-bold text-amber-600 dark:text-amber-400 text-sm tabular-nums">
              {summaryStats.outlierCount} ({summaryStats.outlierPct}%)
            </span>
          </div>
          <div>
            <span className="text-slate-400 block text-[11px]">Skewness</span>
            <span className="font-mono font-bold text-slate-800 dark:text-slate-200 text-sm tabular-nums">
              {summaryStats.skewness > 0 ? `+${summaryStats.skewness}` : summaryStats.skewness}
            </span>
            <span className="text-[10px] text-slate-400 block">
              {summaryStats.skewness > 0.4 ? 'Right-skewed' : 'Symmetric'}
            </span>
          </div>
        </div>
      )}

      {/* Recharts Chart Viewport */}
      <div className="h-80 w-full pt-1">
        {viewType === 'histogram' ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histogramData} margin={{ top: 20, right: 30, bottom: 25, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" opacity={0.2} />

              <XAxis
                dataKey="mid"
                tickFormatter={formatVal}
                tick={{ fill: '#64748b', fontSize: 11 }}
                axisLine={{ stroke: '#94a3b8' }}
                label={{
                  value: `${activeConfig.label} (${activeConfig.unit})`,
                  position: 'insideBottom',
                  offset: -12,
                  fill: '#64748b',
                  fontSize: 12,
                }}
              />

              <YAxis
                tick={{ fill: '#64748b', fontSize: 11 }}
                axisLine={{ stroke: '#94a3b8' }}
                label={{
                  value: 'Frequency (Home Count)',
                  angle: -90,
                  position: 'insideLeft',
                  offset: 0,
                  fill: '#64748b',
                  fontSize: 12,
                }}
              />

              <Tooltip content={<CustomHistogramTooltip />} />

              {/* Vertical Reference Line for Median */}
              {summaryStats && (
                <ReferenceLine
                  x={summaryStats.median}
                  stroke="#2563eb"
                  strokeWidth={2}
                  strokeDasharray="4 2"
                  label={{
                    value: `Median: ${formatVal(summaryStats.median)}`,
                    fill: '#2563eb',
                    position: 'insideTopLeft',
                    fontSize: 11,
                  }}
                />
              )}

              {/* Vertical Reference Line for Upper Outlier Fence */}
              {summaryStats && summaryStats.upperBound < summaryStats.max && (
                <ReferenceLine
                  x={summaryStats.upperBound}
                  stroke="#f59e0b"
                  strokeWidth={2}
                  strokeDasharray="3 3"
                  label={{
                    value: `Upper Outlier Fence: ${formatVal(summaryStats.upperBound)}`,
                    fill: '#f59e0b',
                    position: 'insideTopRight',
                    fontSize: 10,
                  }}
                />
              )}

              {/* Histogram Bars with conditional Outlier coloring */}
              <Bar dataKey="count" name="Frequency Count" radius={[4, 4, 0, 0]}>
                {histogramData.map((entry, index) => (
                  <Cell
                    key={`hist-cell-${index}`}
                    fill={entry.isOutlierBin ? '#f59e0b' : '#3b82f6'}
                    opacity={entry.isOutlierBin ? 0.95 : 0.85}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          /* Recharts Box Plot Visualizer by Neighborhood */
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={neighborhoodBoxPlotData}
              margin={{ top: 20, right: 30, bottom: 35, left: 25 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" opacity={0.2} />

              <XAxis
                dataKey="neighborhood"
                tick={{ fill: '#64748b', fontSize: 11 }}
                axisLine={{ stroke: '#94a3b8' }}
                interval={0}
                angle={-15}
                textAnchor="end"
              />

              <YAxis
                tickFormatter={formatVal}
                tick={{ fill: '#64748b', fontSize: 11 }}
                axisLine={{ stroke: '#94a3b8' }}
                label={{
                  value: `${activeConfig.label} (${activeConfig.unit})`,
                  angle: -90,
                  position: 'insideLeft',
                  offset: 0,
                  fill: '#64748b',
                  fontSize: 12,
                }}
              />

              <Tooltip content={<CustomBoxPlotTooltip />} />

              {/* Stacked Bars to represent Box Plot (Min to Q1 transparent, Q1 to Q3 colored box) */}
              <Bar dataKey="q1" stackId="boxplot" fill="transparent" />
              <Bar
                dataKey="boxSpan"
                stackId="boxplot"
                name="Interquartile Range (Q1 - Q3)"
                fill="#3b82f6"
                radius={[4, 4, 4, 4]}
                opacity={0.85}
              />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Explanatory Guide Footnote */}
      <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 text-xs text-slate-600 dark:text-slate-400 space-y-1">
        <div className="font-semibold text-slate-900 dark:text-white flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5 text-blue-500" />
          <span>Outlier Identification & Machine Learning Impact:</span>
        </div>
        <p className="leading-relaxed text-[11px]">
          Bins highlighted in <strong className="text-amber-600 dark:text-amber-400">Amber</strong> fall beyond the{' '}
          <code className="font-mono text-blue-600 dark:text-blue-400">Q3 + {iqrMultiplier}×IQR</code> fence. In residential housing data, high square footage or luxury sales create long right tails (positive skewness). Linear Regression is highly sensitive to these extreme points as squared error penalties pull the regression hyperplane. You can enable <strong>IQR Outlier Filtering</strong> or <strong>Target Log-Transformation (<code className="font-mono">np.log1p</code>)</strong> in the Data Cleaning & Pipeline tab to normalize the distribution before fitting!
        </p>
      </div>
    </div>
  );
};

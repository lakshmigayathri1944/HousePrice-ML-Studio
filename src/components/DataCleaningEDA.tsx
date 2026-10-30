import React, { useState, useMemo } from 'react';
import {
  Search,
  Download,
  Upload,
  RotateCcw,
  Sliders,
  BarChart2,
  Table as TableIcon,
  HelpCircle,
} from 'lucide-react';
import { PropertyRecord, TrainingConfig } from '../ml/types.ts';
import { NEIGHBORHOODS, exportDatasetToCSV, parseCSVToDataset } from '../ml/dataset.ts';
import { detectOutliersIQR } from '../ml/preprocessing.ts';
import { D3CorrelationHeatmap } from './D3CorrelationHeatmap.tsx';
import { FeatureDistributionVisualizer } from './FeatureDistributionVisualizer.tsx';

interface DataCleaningEDAProps {
  dataset: PropertyRecord[];
  setDataset: (data: PropertyRecord[]) => void;
  config: TrainingConfig;
  setConfig: React.Dispatch<React.SetStateAction<TrainingConfig>>;
  onResetDefaultData: () => void;
}

export const DataCleaningEDA: React.FC<DataCleaningEDAProps> = ({
  dataset,
  setDataset,
  config,
  setConfig,
  onResetDefaultData,
}) => {
  const [activeTab, setActiveTab] = useState<'eda' | 'cleaning' | 'data'>('eda');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedNeighborhood, setSelectedNeighborhood] = useState<string>('all');
  const [sortField, setSortField] = useState<keyof PropertyRecord>('salePrice');
  const [sortAsc, setSortAsc] = useState(false);

  // Outlier detection preview
  const outlierInfo = useMemo(() => {
    return detectOutliersIQR(dataset, config.outlierThresholdIQR);
  }, [dataset, config.outlierThresholdIQR]);

  // Statistical calculations
  const stats = useMemo(() => {
    if (dataset.length === 0) return null;

    const calc = (arr: number[]) => {
      const sorted = [...arr].sort((a, b) => a - b);
      const count = sorted.length;
      const sum = sorted.reduce((a, b) => a + b, 0);
      const mean = sum / count;
      const variance = sorted.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / count;
      const std = Math.sqrt(variance);
      const min = sorted[0];
      const p25 = sorted[Math.floor(count * 0.25)];
      const median = sorted[Math.floor(count * 0.5)];
      const p75 = sorted[Math.floor(count * 0.75)];
      const max = sorted[count - 1];
      return { count, mean, std, min, p25, median, p75, max };
    };

    return {
      price: calc(dataset.map((d) => d.salePrice)),
      sqft: calc(dataset.map((d) => d.sqftLiving)),
      beds: calc(dataset.map((d) => d.bedrooms)),
      baths: calc(dataset.map((d) => d.bathrooms)),
      quality: calc(dataset.map((d) => d.overallQuality)),
      year: calc(dataset.map((d) => d.yearBuilt)),
    };
  }, [dataset]);

  // Correlation calculation between key numerical variables
  const correlations = useMemo(() => {
    const keys: (keyof PropertyRecord)[] = [
      'salePrice',
      'sqftLiving',
      'overallQuality',
      'bathrooms',
      'bedrooms',
      'garageCars',
      'yearBuilt',
    ];

    const labels: Record<string, string> = {
      salePrice: 'Price',
      sqftLiving: 'SqFt',
      overallQuality: 'Quality',
      bathrooms: 'Baths',
      bedrooms: 'Beds',
      garageCars: 'Garage',
      yearBuilt: 'Year',
    };

    const getCorr = (k1: keyof PropertyRecord, k2: keyof PropertyRecord) => {
      const n = dataset.length;
      if (n <= 1) return 0;
      const x = dataset.map((d) => Number(d[k1]));
      const y = dataset.map((d) => Number(d[k2]));
      const meanX = x.reduce((a, b) => a + b, 0) / n;
      const meanY = y.reduce((a, b) => a + b, 0) / n;

      let num = 0;
      let denX = 0;
      let denY = 0;
      for (let i = 0; i < n; i++) {
        const dx = x[i] - meanX;
        const dy = y[i] - meanY;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
      }
      const den = Math.sqrt(denX * denY);
      return den > 0 ? num / den : 0;
    };

    const matrix = keys.map((k1) =>
      keys.map((k2) => ({
        k1,
        k2,
        corr: getCorr(k1, k2),
      }))
    );

    return { keys, labels, matrix };
  }, [dataset]);

  // Histogram buckets for sale price
  const priceHistogram = useMemo(() => {
    if (dataset.length === 0) return [];
    const min = 100000;
    const max = 1500000;
    const bucketCount = 14;
    const step = (max - min) / bucketCount;

    const buckets = Array.from({ length: bucketCount }, (_, i) => ({
      rangeStart: min + i * step,
      rangeEnd: min + (i + 1) * step,
      count: 0,
    }));

    dataset.forEach((d) => {
      const idx = Math.min(bucketCount - 1, Math.max(0, Math.floor((d.salePrice - min) / step)));
      buckets[idx].count++;
    });

    const maxCount = Math.max(...buckets.map((b) => b.count), 1);
    return buckets.map((b) => ({ ...b, relativeHeight: (b.count / maxCount) * 100 }));
  }, [dataset]);

  // Filtered & sorted records for the data table
  const filteredRecords = useMemo(() => {
    return dataset
      .filter((rec) => {
        const matchesNeighborhood =
          selectedNeighborhood === 'all' || rec.neighborhood === selectedNeighborhood;
        const matchesSearch =
          searchQuery.trim() === '' ||
          rec.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
          rec.neighborhood.toLowerCase().includes(searchQuery.toLowerCase()) ||
          rec.sqftLiving.toString().includes(searchQuery);
        return matchesNeighborhood && matchesSearch;
      })
      .sort((a, b) => {
        const valA = a[sortField];
        const valB = b[sortField];
        if (typeof valA === 'number' && typeof valB === 'number') {
          return sortAsc ? valA - valB : valB - valA;
        }
        return sortAsc
          ? String(valA).localeCompare(String(valB))
          : String(valB).localeCompare(String(valA));
      });
  }, [dataset, selectedNeighborhood, searchQuery, sortField, sortAsc]);

  // CSV download
  const handleDownloadCSV = () => {
    const csv = exportDatasetToCSV(dataset);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'housing_dataset_clean.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // CSV upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      const text = evt.target?.result as string;
      const parsed = parseCSVToDataset(text);
      if (parsed.length > 0) {
        setDataset(parsed);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="space-y-6">
      {/* Sub-navigation Segmented Control */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-4">
        <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg">
          <button
            onClick={() => setActiveTab('eda')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'eda'
                ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-semibold'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <span className="flex items-center gap-1.5">
              <BarChart2 className="w-3.5 h-3.5" />
              <span>Exploratory Data Analysis (EDA)</span>
            </span>
          </button>
          <button
            onClick={() => setActiveTab('cleaning')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'cleaning'
                ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-semibold'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <span className="flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5" />
              <span>Data Cleaning & Pipeline</span>
            </span>
          </button>
          <button
            onClick={() => setActiveTab('data')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'data'
                ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-semibold'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <span className="flex items-center gap-1.5">
              <TableIcon className="w-3.5 h-3.5" />
              <span>Dataset Inspection ({dataset.length})</span>
            </span>
          </button>
        </div>

        <div className="flex items-center gap-2">
          <label className="cursor-pointer inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-300 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700/60 rounded-lg transition-colors whitespace-nowrap shrink-0">
            <Upload className="w-3.5 h-3.5" />
            <span>Upload CSV</span>
            <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
          </label>
          <button
            onClick={handleDownloadCSV}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-300 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700/60 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download CSV</span>
          </button>
          <button
            onClick={onResetDefaultData}
            title="Reload default 1,000 residential homes"
            className="p-1.5 text-slate-500 hover:text-slate-900 dark:hover:text-white rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* TAB 1: EXPLORATORY DATA ANALYSIS (EDA) */}
      {activeTab === 'eda' && (
        <div className="space-y-6">
          {/* Statistical Summary Grid */}
          {stats && (
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                  Descriptive Feature Statistics (Pandas .describe() equivalent)
                </h3>
                <span className="text-xs text-slate-500 font-mono">N = {dataset.length} samples</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-50 dark:bg-slate-800/60 text-slate-600 dark:text-slate-400 font-medium border-b border-slate-200 dark:border-slate-700">
                    <tr>
                      <th className="py-2.5 px-3">Feature</th>
                      <th className="py-2.5 px-3 text-right">Mean</th>
                      <th className="py-2.5 px-3 text-right">Std Dev</th>
                      <th className="py-2.5 px-3 text-right">Min</th>
                      <th className="py-2.5 px-3 text-right">25% (Q1)</th>
                      <th className="py-2.5 px-3 text-right">Median (50%)</th>
                      <th className="py-2.5 px-3 text-right">75% (Q3)</th>
                      <th className="py-2.5 px-3 text-right">Max</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800 font-mono tabular-nums">
                    <tr>
                      <td className="py-2 px-3 font-sans font-medium text-slate-900 dark:text-white">
                        Sale Price ($)
                      </td>
                      <td className="py-2 px-3 text-right">${Math.round(stats.price.mean).toLocaleString()}</td>
                      <td className="py-2 px-3 text-right">${Math.round(stats.price.std).toLocaleString()}</td>
                      <td className="py-2 px-3 text-right">${Math.round(stats.price.min).toLocaleString()}</td>
                      <td className="py-2 px-3 text-right">${Math.round(stats.price.p25).toLocaleString()}</td>
                      <td className="py-2 px-3 text-right font-semibold text-blue-600 dark:text-blue-400">
                        ${Math.round(stats.price.median).toLocaleString()}
                      </td>
                      <td className="py-2 px-3 text-right">${Math.round(stats.price.p75).toLocaleString()}</td>
                      <td className="py-2 px-3 text-right">${Math.round(stats.price.max).toLocaleString()}</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-sans font-medium text-slate-900 dark:text-white">
                        Living Area (SqFt)
                      </td>
                      <td className="py-2 px-3 text-right">{Math.round(stats.sqft.mean)}</td>
                      <td className="py-2 px-3 text-right">{Math.round(stats.sqft.std)}</td>
                      <td className="py-2 px-3 text-right">{stats.sqft.min}</td>
                      <td className="py-2 px-3 text-right">{stats.sqft.p25}</td>
                      <td className="py-2 px-3 text-right">{stats.sqft.median}</td>
                      <td className="py-2 px-3 text-right">{stats.sqft.p75}</td>
                      <td className="py-2 px-3 text-right">{stats.sqft.max}</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-sans font-medium text-slate-900 dark:text-white">
                        Overall Quality (1-10)
                      </td>
                      <td className="py-2 px-3 text-right">{stats.quality.mean.toFixed(2)}</td>
                      <td className="py-2 px-3 text-right">{stats.quality.std.toFixed(2)}</td>
                      <td className="py-2 px-3 text-right">{stats.quality.min}</td>
                      <td className="py-2 px-3 text-right">{stats.quality.p25}</td>
                      <td className="py-2 px-3 text-right">{stats.quality.median}</td>
                      <td className="py-2 px-3 text-right">{stats.quality.p75}</td>
                      <td className="py-2 px-3 text-right">{stats.quality.max}</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-sans font-medium text-slate-900 dark:text-white">
                        Bedrooms
                      </td>
                      <td className="py-2 px-3 text-right">{stats.beds.mean.toFixed(1)}</td>
                      <td className="py-2 px-3 text-right">{stats.beds.std.toFixed(1)}</td>
                      <td className="py-2 px-3 text-right">{stats.beds.min}</td>
                      <td className="py-2 px-3 text-right">{stats.beds.p25}</td>
                      <td className="py-2 px-3 text-right">{stats.beds.median}</td>
                      <td className="py-2 px-3 text-right">{stats.beds.p75}</td>
                      <td className="py-2 px-3 text-right">{stats.beds.max}</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-sans font-medium text-slate-900 dark:text-white">
                        Bathrooms
                      </td>
                      <td className="py-2 px-3 text-right">{stats.baths.mean.toFixed(1)}</td>
                      <td className="py-2 px-3 text-right">{stats.baths.std.toFixed(1)}</td>
                      <td className="py-2 px-3 text-right">{stats.baths.min}</td>
                      <td className="py-2 px-3 text-right">{stats.baths.p25}</td>
                      <td className="py-2 px-3 text-right">{stats.baths.median}</td>
                      <td className="py-2 px-3 text-right">{stats.baths.p75}</td>
                      <td className="py-2 px-3 text-right">{stats.baths.max}</td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-sans font-medium text-slate-900 dark:text-white">
                        Year Built
                      </td>
                      <td className="py-2 px-3 text-right">{Math.round(stats.year.mean)}</td>
                      <td className="py-2 px-3 text-right">{Math.round(stats.year.std)}</td>
                      <td className="py-2 px-3 text-right">{stats.year.min}</td>
                      <td className="py-2 px-3 text-right">{stats.year.p25}</td>
                      <td className="py-2 px-3 text-right">{stats.year.median}</td>
                      <td className="py-2 px-3 text-right">{stats.year.p75}</td>
                      <td className="py-2 px-3 text-right">{stats.year.max}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Recharts Feature Distribution & Outlier Visualizer (Histogram & Box Plot) */}
          <FeatureDistributionVisualizer dataset={dataset} />

          {/* D3.js Pearson Correlation Heatmap Component */}
          <D3CorrelationHeatmap dataset={dataset} />

          {/* Neighborhood Price Spread Box Breakdown */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm space-y-4">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
              Price Distribution by Neighborhood
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {NEIGHBORHOODS.map((n) => {
                const subset = dataset.filter((d) => d.neighborhood === n.name);
                const prices = subset.map((d) => d.salePrice).sort((a, b) => a - b);
                const median = prices.length ? prices[Math.floor(prices.length * 0.5)] : 0;
                const min = prices.length ? prices[0] : 0;
                const max = prices.length ? prices[prices.length - 1] : 0;
                return (
                  <div key={n.name} className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-slate-900 dark:text-white">{n.name}</span>
                      <span className="text-slate-500 font-mono">{subset.length} homes</span>
                    </div>
                    <div className="text-lg font-bold text-slate-900 dark:text-white font-mono tabular-nums">
                      ${Math.round(median).toLocaleString()}
                    </div>
                    <div className="text-[11px] text-slate-500 flex justify-between font-mono">
                      <span>Min: ${Math.round(min / 1000)}k</span>
                      <span>Max: ${Math.round(max / 1000)}k</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: DATA CLEANING & PIPELINE CONFIGURATION */}
      {activeTab === 'cleaning' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm space-y-6">
              <div>
                <h3 className="text-base font-semibold text-slate-900 dark:text-white">
                  Data Cleaning & Preprocessing Pipeline
                </h3>
                <p className="text-xs text-slate-500 mt-1">
                  Configure feature preparation, outlier trimming, scaling, and train/test splits.
                </p>
              </div>

              {/* IQR Outlier Detection Card */}
              <div className="p-4 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <div className="text-sm font-semibold text-slate-900 dark:text-white flex items-center gap-2">
                      <span>Interquartile Range (IQR) Outlier Filtering</span>
                    </div>
                    <p className="text-xs text-slate-500">
                      Eliminates extreme anomalies beyond [Q1 - k*IQR, Q3 + k*IQR] on sale price.
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={config.filterOutliers}
                    onChange={(e) => setConfig((prev) => ({ ...prev, filterOutliers: e.target.checked }))}
                    className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500"
                  />
                </div>

                {config.filterOutliers && (
                  <div className="space-y-3 pt-2 border-t border-slate-200 dark:border-slate-700">
                    <div className="flex justify-between text-xs text-slate-600 dark:text-slate-400">
                      <span>Multiplier (k):</span>
                      <span className="font-mono font-semibold">{config.outlierThresholdIQR}x</span>
                    </div>
                    <input
                      type="range"
                      min="1.0"
                      max="3.0"
                      step="0.1"
                      value={config.outlierThresholdIQR}
                      onChange={(e) =>
                        setConfig((prev) => ({ ...prev, outlierThresholdIQR: parseFloat(e.target.value) }))
                      }
                      className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-500">Outliers detected & excluded:</span>
                      <span className="font-mono font-semibold text-amber-600 dark:text-amber-400">
                        {outlierInfo.outliers.length} properties ({((outlierInfo.outliers.length / dataset.length) * 100).toFixed(1)}%)
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Train / Test Split Slider */}
              <div className="p-4 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 space-y-3">
                <div className="flex justify-between items-center text-sm font-semibold text-slate-900 dark:text-white">
                  <span>Train / Test Split Ratio</span>
                  <span className="font-mono text-blue-600 dark:text-blue-400">
                    {Math.round(config.trainRatio * 100)}% Train / {Math.round((1 - config.trainRatio) * 100)}% Test
                  </span>
                </div>
                <input
                  type="range"
                  min="0.6"
                  max="0.85"
                  step="0.05"
                  value={config.trainRatio}
                  onChange={(e) =>
                    setConfig((prev) => ({ ...prev, trainRatio: parseFloat(e.target.value) }))
                  }
                  className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-slate-500 font-mono">
                  <span>Train size: {Math.round(dataset.length * config.trainRatio)} samples</span>
                  <span>Test size: {Math.round(dataset.length * (1 - config.trainRatio))} samples</span>
                </div>
              </div>

              {/* Toggles: Standardization & Log Transformation */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 flex items-start justify-between">
                  <div>
                    <h4 className="text-xs font-semibold text-slate-900 dark:text-white">
                      Z-Score Feature Standardization
                    </h4>
                    <p className="text-[11px] text-slate-500 mt-0.5">
                      Scales features to zero mean and unit variance for stable OLS & Ridge regression.
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={config.standardizeFeatures}
                    onChange={(e) =>
                      setConfig((prev) => ({ ...prev, standardizeFeatures: e.target.checked }))
                    }
                    className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500 mt-0.5 ml-2"
                  />
                </div>

                <div className="p-4 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 flex items-start justify-between">
                  <div>
                    <h4 className="text-xs font-semibold text-slate-900 dark:text-white">
                      Log-Transform Target (np.log1p)
                    </h4>
                    <p className="text-[11px] text-slate-500 mt-0.5">
                      Normalizes right-skewed price distribution to stabilize variance.
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={config.logTransformTarget}
                    onChange={(e) =>
                      setConfig((prev) => ({ ...prev, logTransformTarget: e.target.checked }))
                    }
                    className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500 mt-0.5 ml-2"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Outlier Inspector sidebar */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm space-y-4">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
              Detected Outliers ({outlierInfo.outliers.length})
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              These properties fell beyond the IQR confidence boundary and are {config.filterOutliers ? 'trimmed from training' : 'currently included'}.
            </p>

            <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
              {outlierInfo.outliers.length === 0 ? (
                <div className="text-xs text-slate-500 italic p-4 text-center">
                  No extreme outliers detected at current threshold.
                </div>
              ) : (
                outlierInfo.outliers.map((o) => (
                  <div
                    key={o.id}
                    className="p-2.5 rounded bg-amber-50/60 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800/50 text-xs space-y-1"
                  >
                    <div className="flex justify-between font-mono font-medium text-slate-800 dark:text-slate-200">
                      <span>{o.id}</span>
                      <span className="text-amber-700 dark:text-amber-400 font-bold">
                        ${o.salePrice.toLocaleString()}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-500 flex justify-between">
                      <span>{o.neighborhood}</span>
                      <span>{o.sqftLiving} sqft · {o.overallQuality}/10 qual</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: DATASET VIEWER */}
      {activeTab === 'data' && (
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="relative w-full sm:w-72">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                placeholder="Search by ID, neighborhood, sqft..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-9 pr-3 py-1.5 text-xs bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>

            <div className="flex items-center gap-3 w-full sm:w-auto justify-end">
              <select
                value={selectedNeighborhood}
                onChange={(e) => setSelectedNeighborhood(e.target.value)}
                className="text-xs bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-slate-800 dark:text-slate-200 focus:outline-none"
              >
                <option value="all">All Neighborhoods</option>
                {NEIGHBORHOODS.map((n) => (
                  <option key={n.name} value={n.name}>
                    {n.name}
                  </option>
                ))}
              </select>

              <span className="text-xs text-slate-500 font-mono">
                Showing {filteredRecords.length} of {dataset.length}
              </span>
            </div>
          </div>

          <div className="overflow-x-auto max-h-[500px]">
            <table className="w-full text-xs text-left">
              <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-300 font-medium border-b border-slate-200 dark:border-slate-700 z-10">
                <tr>
                  <th
                    onClick={() => {
                      if (sortField === 'id') setSortAsc(!sortAsc);
                      else {
                        setSortField('id');
                        setSortAsc(true);
                      }
                    }}
                    className="py-2.5 px-3 cursor-pointer hover:text-slate-900 dark:hover:text-white"
                  >
                    Property ID
                  </th>
                  <th
                    onClick={() => {
                      if (sortField === 'neighborhood') setSortAsc(!sortAsc);
                      else {
                        setSortField('neighborhood');
                        setSortAsc(true);
                      }
                    }}
                    className="py-2.5 px-3 cursor-pointer hover:text-slate-900 dark:hover:text-white"
                  >
                    Neighborhood
                  </th>
                  <th
                    onClick={() => {
                      if (sortField === 'sqftLiving') setSortAsc(!sortAsc);
                      else {
                        setSortField('sqftLiving');
                        setSortAsc(false);
                      }
                    }}
                    className="py-2.5 px-3 text-right cursor-pointer hover:text-slate-900 dark:hover:text-white"
                  >
                    SqFt Living
                  </th>
                  <th className="py-2.5 px-3 text-right">Beds / Baths</th>
                  <th
                    onClick={() => {
                      if (sortField === 'overallQuality') setSortAsc(!sortAsc);
                      else {
                        setSortField('overallQuality');
                        setSortAsc(false);
                      }
                    }}
                    className="py-2.5 px-3 text-right cursor-pointer hover:text-slate-900 dark:hover:text-white"
                  >
                    Quality (1-10)
                  </th>
                  <th className="py-2.5 px-3 text-right">Year Built</th>
                  <th className="py-2.5 px-3 text-right">Features</th>
                  <th
                    onClick={() => {
                      if (sortField === 'salePrice') setSortAsc(!sortAsc);
                      else {
                        setSortField('salePrice');
                        setSortAsc(false);
                      }
                    }}
                    className="py-2.5 px-3 text-right cursor-pointer hover:text-slate-900 dark:hover:text-white font-bold"
                  >
                    Sale Price ($)
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800 font-mono tabular-nums">
                {filteredRecords.slice(0, 50).map((r) => (
                  <tr key={r.id} className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors">
                    <td className="py-2 px-3 text-slate-800 dark:text-slate-200">{r.id}</td>
                    <td className="py-2 px-3 font-sans text-slate-600 dark:text-slate-300">{r.neighborhood}</td>
                    <td className="py-2 px-3 text-right text-slate-800 dark:text-slate-200">
                      {r.sqftLiving.toLocaleString()}
                    </td>
                    <td className="py-2 px-3 text-right text-slate-600 dark:text-slate-400">
                      {r.bedrooms}b / {r.bathrooms}ba
                    </td>
                    <td className="py-2 px-3 text-right">
                      <span className="font-semibold text-slate-800 dark:text-slate-200">
                        {r.overallQuality}
                      </span>
                      <span className="text-slate-400 text-[10px]">/10</span>
                    </td>
                    <td className="py-2 px-3 text-right text-slate-600 dark:text-slate-400">{r.yearBuilt}</td>
                    <td className="py-2 px-3 text-right font-sans text-[11px] text-slate-500">
                      {[
                        r.waterfront ? 'Waterfront' : '',
                        r.hasPool ? 'Pool' : '',
                        r.hasBasement ? 'Basement' : '',
                      ]
                        .filter(Boolean)
                        .join(' · ') || 'Standard'}
                    </td>
                    <td className="py-2 px-3 text-right font-semibold text-slate-900 dark:text-white">
                      ${r.salePrice.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {filteredRecords.length > 50 && (
            <div className="text-center text-xs text-slate-500 pt-2 border-t border-slate-100 dark:border-slate-800">
              Showing top 50 records. Download CSV for the full {dataset.length}-row dataset.
            </div>
          )}
        </div>
      )}
    </div>
  );
};

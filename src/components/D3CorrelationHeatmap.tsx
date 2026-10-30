import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { PropertyRecord } from '../ml/types.ts';
import { Info, Sparkles, Filter, Eye, Palette } from 'lucide-react';

interface D3CorrelationHeatmapProps {
  dataset: PropertyRecord[];
}

interface FeatureDef {
  key: keyof PropertyRecord;
  label: string;
  shortLabel: string;
}

const ALL_NUMERICAL_FEATURES: FeatureDef[] = [
  { key: 'salePrice', label: 'Sale Price ($)', shortLabel: 'Price' },
  { key: 'sqftLiving', label: 'Living Area (SqFt)', shortLabel: 'SqFt' },
  { key: 'overallQuality', label: 'Overall Quality', shortLabel: 'Quality' },
  { key: 'yearBuilt', label: 'Year Built', shortLabel: 'Year' },
  { key: 'bathrooms', label: 'Bathrooms', shortLabel: 'Baths' },
  { key: 'bedrooms', label: 'Bedrooms', shortLabel: 'Beds' },
  { key: 'garageCars', label: 'Garage Capacity', shortLabel: 'Garage' },
  { key: 'distanceToCenterKm', label: 'Dist. to Center', shortLabel: 'Distance' },
  { key: 'lotSizeSqft', label: 'Lot Size (SqFt)', shortLabel: 'Lot' },
];

export const D3CorrelationHeatmap: React.FC<D3CorrelationHeatmapProps> = ({ dataset }) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // User options
  const [selectedKeys, setSelectedKeys] = useState<string[]>([
    'salePrice',
    'sqftLiving',
    'overallQuality',
    'yearBuilt',
    'bathrooms',
    'bedrooms',
    'garageCars',
    'distanceToCenterKm',
  ]);
  const [colorPalette, setColorPalette] = useState<'blues' | 'diverging' | 'viridis'>('blues');
  const [showValues, setShowValues] = useState<boolean>(true);
  const [hoveredCell, setHoveredCell] = useState<{
    f1: FeatureDef;
    f2: FeatureDef;
    corr: number;
    r2: number;
  } | null>(null);

  // Active feature definitions
  const activeFeatures = useMemo(() => {
    return ALL_NUMERICAL_FEATURES.filter((f) => selectedKeys.includes(f.key));
  }, [selectedKeys]);

  // Compute correlation matrix
  const correlationData = useMemo(() => {
    const n = dataset.length;
    if (n <= 1) return [];

    const getPearson = (k1: keyof PropertyRecord, k2: keyof PropertyRecord) => {
      const x = dataset.map((d) => Number(d[k1]));
      const y = dataset.map((d) => Number(d[k2]));
      const meanX = d3.mean(x) || 0;
      const meanY = d3.mean(y) || 0;

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

    const cells: {
      f1: FeatureDef;
      f2: FeatureDef;
      corr: number;
      r2: number;
      i: number;
      j: number;
    }[] = [];

    activeFeatures.forEach((f1, i) => {
      activeFeatures.forEach((f2, j) => {
        const corr = getPearson(f1.key, f2.key);
        cells.push({
          f1,
          f2,
          corr,
          r2: corr * corr,
          i,
          j,
        });
      });
    });

    return cells;
  }, [dataset, activeFeatures]);

  // Render D3 heatmap
  useEffect(() => {
    if (!svgRef.current || correlationData.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 75, right: 35, bottom: 25, left: 85 };
    const width = 580;
    const height = 520;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    svg.attr('viewBox', `0 0 ${width} ${height}`);

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const keys = activeFeatures.map((f) => f.shortLabel);

    // X Scale
    const x = d3.scaleBand().range([0, innerWidth]).domain(keys).padding(0.04);

    // Y Scale
    const y = d3.scaleBand().range([0, innerHeight]).domain(keys).padding(0.04);

    // Color Scale based on selection
    let colorScale: (val: number) => string;
    if (colorPalette === 'blues') {
      // Linear scale from soft slate to deep royal blue
      colorScale = (val: number) => {
        // Map -1..1 to 0..1 for absolute correlation or directional
        const norm = (val + 1) / 2;
        return d3.interpolateBlues(0.1 + norm * 0.9);
      };
    } else if (colorPalette === 'diverging') {
      // Diverging Red (negative) to Neutral to Blue (positive)
      colorScale = (val: number) => {
        return d3.interpolateRdBu((val + 1) / 2);
      };
    } else {
      // Viridis
      colorScale = (val: number) => {
        return d3.interpolateViridis((val + 1) / 2);
      };
    }

    // Top X Axis with tilted text
    const xAxis = g.append('g').call(d3.axisTop(x).tickSize(0));
    xAxis.select('.domain').remove();
    xAxis
      .selectAll('text')
      .style('text-anchor', 'start')
      .attr('dx', '8px')
      .attr('dy', '-6px')
      .attr('transform', 'rotate(-38)')
      .attr('class', 'text-[11px] font-medium fill-slate-700 dark:fill-slate-300 font-sans');

    // Left Y Axis
    const yAxis = g.append('g').call(d3.axisLeft(y).tickSize(0));
    yAxis.select('.domain').remove();
    yAxis
      .selectAll('text')
      .attr('class', 'text-[11px] font-medium fill-slate-700 dark:fill-slate-300 font-sans')
      .attr('dx', '-6px');

    // Draw Heatmap Cells
    const cellGroups = g
      .selectAll('.cell-group')
      .data(correlationData)
      .enter()
      .append('g')
      .attr('class', 'cell-group')
      .style('cursor', 'pointer');

    // Cell rectangles
    cellGroups
      .append('rect')
      .attr('x', (d) => x(d.f2.shortLabel) || 0)
      .attr('y', (d) => y(d.f1.shortLabel) || 0)
      .attr('rx', 4)
      .attr('ry', 4)
      .attr('width', x.bandwidth())
      .attr('height', y.bandwidth())
      .style('fill', (d) => colorScale(d.corr))
      .style('stroke', '#ffffff')
      .style('stroke-width', 1.5)
      .style('opacity', 0.9)
      .on('mouseover', function (event, d) {
        d3.select(this)
          .style('stroke', '#2563eb')
          .style('stroke-width', 2.5)
          .style('opacity', 1);
        setHoveredCell({
          f1: d.f1,
          f2: d.f2,
          corr: d.corr,
          r2: d.r2,
        });
      })
      .on('mouseout', function () {
        d3.select(this)
          .style('stroke', '#ffffff')
          .style('stroke-width', 1.5)
          .style('opacity', 0.9);
      });

    // Numerical text labels inside cells
    if (showValues) {
      cellGroups
        .append('text')
        .attr('x', (d) => (x(d.f2.shortLabel) || 0) + x.bandwidth() / 2)
        .attr('y', (d) => (y(d.f1.shortLabel) || 0) + y.bandwidth() / 2)
        .attr('dy', '.35em')
        .attr('text-anchor', 'middle')
        .attr('class', 'text-[10px] font-mono font-semibold pointer-events-none select-none')
        .style('fill', (d) => {
          if (colorPalette === 'diverging') {
            return Math.abs(d.corr) > 0.45 ? '#ffffff' : '#0f172a';
          }
          return d.corr > 0.4 ? '#ffffff' : '#1e293b';
        })
        .text((d) => d.corr.toFixed(2));
    }
  }, [correlationData, activeFeatures, colorPalette, showValues]);

  // Toggle feature inclusion
  const toggleFeature = (key: string) => {
    if (selectedKeys.includes(key)) {
      if (selectedKeys.length <= 3) return; // Keep at least 3 features
      setSelectedKeys(selectedKeys.filter((k) => k !== key));
    } else {
      setSelectedKeys([...selectedKeys, key]);
    }
  };

  return (
    <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm space-y-6">
      {/* Title & Controls Strip */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-slate-900 dark:text-white">
              D3.js Pearson Correlation Heatmap
            </h3>
            <span className="text-xs px-2 py-0.5 rounded-full bg-blue-50 dark:bg-blue-950 text-blue-600 dark:text-blue-400 font-mono font-medium">
              D3 SVG
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Visualize linear relationships and co-dependence between square footage, construction year, quality, and sale price.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {/* Color Palette Selector */}
          <div className="flex items-center gap-1.5 bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-1">
            <Palette className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={colorPalette}
              onChange={(e) => setColorPalette(e.target.value as any)}
              className="bg-transparent text-slate-800 dark:text-slate-200 text-xs focus:outline-none cursor-pointer"
            >
              <option value="blues">Sequential Blues</option>
              <option value="diverging">Red-Blue Diverging</option>
              <option value="viridis">Viridis Spectral</option>
            </select>
          </div>

          {/* Show Values Toggle */}
          <button
            onClick={() => setShowValues(!showValues)}
            className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg border text-xs font-medium transition-colors ${
              showValues
                ? 'bg-blue-50 dark:bg-blue-950 text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800'
                : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-slate-700'
            }`}
          >
            <Eye className="w-3 h-3" />
            <span>Labels</span>
          </button>
        </div>
      </div>

      {/* Feature Selector Chips */}
      <div className="space-y-1.5">
        <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider block">
          Toggle Features in Correlation Matrix:
        </span>
        <div className="flex flex-wrap gap-1.5">
          {ALL_NUMERICAL_FEATURES.map((f) => {
            const isSelected = selectedKeys.includes(f.key);
            return (
              <button
                key={f.key}
                onClick={() => toggleFeature(f.key)}
                className={`px-2.5 py-1 text-xs rounded-md transition-colors whitespace-nowrap shrink-0 border ${
                  isSelected
                    ? 'bg-blue-600 text-white border-blue-600 font-medium shadow-xs'
                    : 'bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-slate-700 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                {f.shortLabel}
              </button>
            );
          })}
        </div>
      </div>

      {/* Heatmap Graphic & Live Inspector Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-center">
        {/* D3 Heatmap SVG Canvas */}
        <div ref={containerRef} className="lg:col-span-2 flex justify-center w-full overflow-x-auto">
          <svg ref={svgRef} className="w-full max-w-[580px] h-auto select-none" />
        </div>

        {/* Live Pairwise Correlation Inspector */}
        <div className="space-y-4 bg-slate-50 dark:bg-slate-800/50 p-5 rounded-xl border border-slate-200 dark:border-slate-700/60 text-xs">
          <div className="flex items-center gap-1.5 text-slate-800 dark:text-slate-200 font-semibold">
            <Sparkles className="w-4 h-4 text-blue-500" />
            <span>Pairwise Correlation Insight</span>
          </div>

          {hoveredCell ? (
            <div className="space-y-3">
              <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700/80 space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-900 dark:text-white">
                    {hoveredCell.f1.label}
                  </span>
                  <span className="text-slate-400">vs</span>
                  <span className="font-semibold text-slate-900 dark:text-white">
                    {hoveredCell.f2.label}
                  </span>
                </div>

                <div className="flex items-baseline justify-between pt-2 border-t border-slate-100 dark:border-slate-800">
                  <span className="text-slate-500">Pearson r:</span>
                  <span
                    className={`font-mono text-lg font-bold tabular-nums ${
                      hoveredCell.corr > 0.6
                        ? 'text-blue-600 dark:text-blue-400'
                        : hoveredCell.corr < -0.3
                        ? 'text-amber-600 dark:text-amber-400'
                        : 'text-slate-700 dark:text-slate-300'
                    }`}
                  >
                    {hoveredCell.corr.toFixed(3)}
                  </span>
                </div>

                <div className="flex justify-between text-[11px] text-slate-500 font-mono">
                  <span>Variance Explained (r²):</span>
                  <span className="text-slate-800 dark:text-slate-200 font-medium">
                    {(hoveredCell.r2 * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              <div className="text-slate-600 dark:text-slate-400 leading-relaxed text-[11px]">
                {hoveredCell.corr > 0.7 && (
                  <p>
                    <strong className="text-blue-600 dark:text-blue-400">Strong positive relationship:</strong> An increase in{' '}
                    {hoveredCell.f1.shortLabel} is strongly associated with an increase in {hoveredCell.f2.shortLabel}.
                  </p>
                )}
                {hoveredCell.corr >= 0.4 && hoveredCell.corr <= 0.7 && (
                  <p>
                    <strong className="text-blue-500">Moderate positive relationship:</strong> Notable upward alignment with moderate variance.
                  </p>
                )}
                {hoveredCell.corr > -0.4 && hoveredCell.corr < 0.4 && hoveredCell.corr !== 1 && (
                  <p>
                    <strong className="text-slate-500">Weak/Independent:</strong> Minimal direct linear co-dependence observed.
                  </p>
                )}
                {hoveredCell.corr <= -0.4 && (
                  <p>
                    <strong className="text-amber-600">Negative relationship:</strong> As {hoveredCell.f1.shortLabel} rises, {hoveredCell.f2.shortLabel} exhibits an inverse trend (e.g. urban distance discounting).
                  </p>
                )}
                {hoveredCell.corr === 1 && (
                  <p className="italic text-slate-400">Identity diagonal: Perfectly correlated with itself ($r = 1.00$).</p>
                )}
              </div>
            </div>
          ) : (
            <div className="p-4 text-center text-slate-400 italic bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800">
              Hover over any cell in the D3 heatmap to inspect the bivariate correlation coefficient and variance explained ($r^2$).
            </div>
          )}

          {/* Quick Key Takeaways */}
          <div className="pt-3 border-t border-slate-200 dark:border-slate-700/60 space-y-2">
            <span className="font-semibold text-slate-800 dark:text-slate-200 block text-[11px]">
              Core Empirical Observations:
            </span>
            <ul className="space-y-1.5 text-[11px] text-slate-500">
              <li className="flex items-start gap-1.5">
                <span className="text-blue-500 font-bold">·</span>
                <span>
                  <strong>SqFt & Quality</strong> share the highest positive correlation with Sale Price ($r &gt; 0.78$), serving as primary linear drivers.
                </span>
              </li>
              <li className="flex items-start gap-1.5">
                <span className="text-blue-500 font-bold">·</span>
                <span>
                  <strong>Year Built</strong> shows moderate positive correlation with price, capturing structural freshness and modern standards.
                </span>
              </li>
              <li className="flex items-start gap-1.5">
                <span className="text-blue-500 font-bold">·</span>
                <span>
                  <strong>Distance to Center</strong> exhibits a negative correlation ($r \approx -0.32$), reflecting location accessibility premiums.
                </span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

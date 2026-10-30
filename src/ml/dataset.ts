import { PropertyRecord, Neighborhood } from './types.ts';

export const NEIGHBORHOODS: { name: Neighborhood; multiplier: number; avgDistanceKm: number }[] = [
  { name: 'Westside Heights', multiplier: 1.35, avgDistanceKm: 4.2 },
  { name: 'Downtown Urban', multiplier: 1.45, avgDistanceKm: 1.5 },
  { name: 'Oakridge Suburbs', multiplier: 1.05, avgDistanceKm: 12.0 },
  { name: 'Riverside Vista', multiplier: 1.25, avgDistanceKm: 7.8 },
  { name: 'Pine Crest', multiplier: 0.95, avgDistanceKm: 16.5 },
  { name: 'Sunset Valley', multiplier: 1.15, avgDistanceKm: 9.3 },
  { name: 'Lakeview Terrace', multiplier: 1.50, avgDistanceKm: 6.0 },
];

// Seeded pseudorandom generator for deterministic reproducible data
function pseudoRandom(seed: number) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

export function generateRealisticHousingDataset(count = 1000, seed = 42): PropertyRecord[] {
  const rng = pseudoRandom(seed);
  const records: PropertyRecord[] = [];

  for (let i = 1; i <= count; i++) {
    // Select neighborhood
    const nIndex = Math.floor(rng() * NEIGHBORHOODS.length);
    const nInfo = NEIGHBORHOODS[nIndex];

    // Square footage: log-normal skew around 2100 sq ft (range ~750 to 5500)
    const u1 = rng();
    const u2 = rng();
    const z0 = Math.sqrt(-2.0 * Math.log(u1 || 0.0001)) * Math.cos(2.0 * Math.PI * u2);
    let sqftLiving = Math.round(2150 + z0 * 680);
    sqftLiving = Math.max(750, Math.min(5600, sqftLiving));

    // Bedrooms correlated with square footage
    let bedrooms = Math.round(1 + (sqftLiving / 700) + (rng() * 1.2 - 0.6));
    bedrooms = Math.max(1, Math.min(6, bedrooms));

    // Bathrooms correlated with bedrooms and sqft
    let bathrooms = Math.round((1 + bedrooms * 0.5 + (sqftLiving / 1400) * 0.5) * 2) / 2;
    bathrooms = Math.max(1, Math.min(5.5, bathrooms));

    // Lot size in sqft
    const lotBase = sqftLiving * (2.5 + rng() * 4.5);
    const lotSizeSqft = Math.round(Math.max(2500, Math.min(35000, lotBase)));

    // Year built (1940 to 2024)
    const yearBuilt = Math.round(1950 + Math.pow(rng(), 1.4) * 74);
    const hasRemodel = rng() > 0.65 && yearBuilt < 2012;
    const yearRemodeled = hasRemodel ? Math.round(yearBuilt + rng() * (2024 - yearBuilt)) : yearBuilt;

    // Quality: 1 to 10 rating
    let quality = Math.round(4 + (sqftLiving / 1200) + (yearRemodeled > 2005 ? 1.5 : 0) + (rng() * 2 - 1));
    quality = Math.max(3, Math.min(10, quality));

    // Condition: 1 to 10 rating
    let condition = Math.round(5 + (yearRemodeled > 2010 ? 1.8 : 0) - ((2024 - yearRemodeled) / 30) + (rng() * 2 - 1));
    condition = Math.max(2, Math.min(9, condition));

    // Garage cars: 0 to 4
    let garageCars = Math.round((sqftLiving > 1600 ? 1 : 0) + (sqftLiving > 2400 ? 1 : 0) + (rng() > 0.6 ? 1 : 0));
    garageCars = Math.max(0, Math.min(4, garageCars));

    // Distance to city center (km) with some variance from neighborhood average
    const distanceToCenterKm = parseFloat(Math.max(0.8, nInfo.avgDistanceKm + (rng() * 3.0 - 1.5)).toFixed(1));

    // Special features
    const waterfront = (nInfo.name === 'Lakeview Terrace' || nInfo.name === 'Riverside Vista') && rng() < 0.22;
    const hasBasement = rng() < 0.55;
    const hasPool = (quality >= 7 || sqftLiving > 3000) && rng() < 0.35;

    // Real estate hedonic price calculation
    const basePrice = 85000;
    const sqftComponent = sqftLiving * 142;
    const qualityMultiplier = Math.pow(quality / 5.5, 1.45);
    const conditionMultiplier = 0.85 + (condition / 10) * 0.25;
    const ageComponent = -Math.max(0, (2024 - yearRemodeled)) * 450;
    const bedBathComponent = bedrooms * 12000 + bathrooms * 18000;
    const garageComponent = garageCars * 16000;
    const distanceDiscount = -distanceToCenterKm * 2800;
    const waterfrontBonus = waterfront ? 180000 : 0;
    const poolBonus = hasPool ? 35000 : 0;
    const basementBonus = hasBasement ? 25000 : 0;
    const lotBonus = (lotSizeSqft / 1000) * 1200;

    // Combined economic baseline
    const unadjustedPrice =
      (basePrice + sqftComponent + bedBathComponent + garageComponent + lotBonus + poolBonus + basementBonus + ageComponent + distanceDiscount) *
      nInfo.multiplier *
      qualityMultiplier *
      conditionMultiplier +
      waterfrontBonus;

    // Realistic market noise (normal distribution ~6% standard deviation)
    const noiseZ = (rng() + rng() + rng() - 1.5) * 0.08;
    let salePrice = Math.round(unadjustedPrice * (1 + noiseZ));

    // Add rare outliers (e.g. distressed sale or luxury overbidding, ~1.5% probability)
    if (rng() < 0.012) {
      salePrice = Math.round(salePrice * 1.55); // high outlier
    } else if (rng() < 0.008) {
      salePrice = Math.round(salePrice * 0.62); // distressed sale outlier
    }

    // Floor price at minimum sensible value
    salePrice = Math.max(95000, Math.round(salePrice / 500) * 500);

    records.push({
      id: `PROP-${1000 + i}`,
      sqftLiving,
      bedrooms,
      bathrooms,
      lotSizeSqft,
      yearBuilt,
      yearRemodeled,
      overallQuality: quality,
      overallCondition: condition,
      neighborhood: nInfo.name,
      garageCars,
      distanceToCenterKm,
      waterfront,
      hasBasement,
      hasPool,
      salePrice,
    });
  }

  return records;
}

export function exportDatasetToCSV(records: PropertyRecord[]): string {
  if (records.length === 0) return '';
  const headers = Object.keys(records[0]).join(',');
  const rows = records.map((r) =>
    Object.values(r)
      .map((val) => (typeof val === 'string' ? `"${val}"` : val))
      .join(',')
  );
  return [headers, ...rows].join('\n');
}

export function parseCSVToDataset(csvText: string): PropertyRecord[] {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 2) return [];

  const headers = lines[0].split(',').map((h) => h.trim().replace(/^["']|["']$/g, ''));
  const records: PropertyRecord[] = [];

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map((v) => v.trim().replace(/^["']|["']$/g, ''));
    if (values.length < headers.length) continue;

    const rowObj: any = {};
    headers.forEach((header, idx) => {
      rowObj[header] = values[idx];
    });

    records.push({
      id: rowObj.id || `UPLOAD-${i}`,
      sqftLiving: Number(rowObj.sqftLiving || 1800),
      bedrooms: Number(rowObj.bedrooms || 3),
      bathrooms: Number(rowObj.bathrooms || 2),
      lotSizeSqft: Number(rowObj.lotSizeSqft || 7500),
      yearBuilt: Number(rowObj.yearBuilt || 1990),
      yearRemodeled: Number(rowObj.yearRemodeled || rowObj.yearBuilt || 1990),
      overallQuality: Number(rowObj.overallQuality || 6),
      overallCondition: Number(rowObj.overallCondition || 5),
      neighborhood: rowObj.neighborhood || 'Oakridge Suburbs',
      garageCars: Number(rowObj.garageCars || 2),
      distanceToCenterKm: Number(rowObj.distanceToCenterKm || 10),
      waterfront: String(rowObj.waterfront).toLowerCase() === 'true' || rowObj.waterfront === '1',
      hasBasement: String(rowObj.hasBasement).toLowerCase() === 'true' || rowObj.hasBasement === '1',
      hasPool: String(rowObj.hasPool).toLowerCase() === 'true' || rowObj.hasPool === '1',
      salePrice: Number(rowObj.salePrice || 350000),
    });
  }

  return records;
}

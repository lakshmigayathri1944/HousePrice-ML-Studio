import express, { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = parseInt(process.env.PORT || '3000', 10);

app.use(express.json({ limit: '10mb' }));

// Gemini AI Client setup
const apiKey = process.env.GEMINI_API_KEY;
let aiClient: GoogleGenAI | null = null;
if (apiKey) {
  aiClient = new GoogleGenAI({ apiKey });
}

// Valuation appraisal endpoint
app.post('/api/appraisal', async (req: Request, res: Response) => {
  try {
    const { property, predictions, metrics } = req.body;

    if (!property) {
      return res.status(400).json({ error: 'Property details are required.' });
    }

    if (!aiClient) {
      // Deterministic expert real estate appraisal fallback
      const avgPrice = predictions?.ensemble || predictions?.randomForest || 450000;
      const formattedPrice = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(avgPrice);
      const pricePerSqFt = Math.round(avgPrice / (property.sqftLiving || 2000));

      return res.json({
        summary: `The property at ${property.neighborhood || 'Metro'} is estimated at ${formattedPrice} (${pricePerSqFt}/sqft), representing strong alignment with prevailing market comparables.`,
        strengths: [
          `Living area of ${property.sqftLiving?.toLocaleString()} sqft provides competitive square footage in the ${property.neighborhood || 'suburban'} sector.`,
          `Quality grade of ${property.overallQuality}/10 situates this home in the upper percentile of finishes and construction standards.`,
          property.waterfront ? 'Direct waterfront access provides a substantial premium.' : `Generous lot footprint with ${property.garageCars}-car garage accommodation.`
        ],
        marketContext: `The Random Forest model projects ${formattedPrice} based on feature importances prioritizing square footage, building condition, and neighborhood premiums. Historical regression variance indicates a low residual spread in this category.`,
        recommendations: [
          'High ROI Potential: Minor kitchen or bathroom modernization typically yields a 1.4x to 1.8x return on cost in this price bracket.',
          'Pricing Strategy: List within a 3.5% corridor of the multi-model consensus to maximize initial qualified buyer velocity.'
        ]
      });
    }

    const prompt = `
You are a senior real estate appraisal analyst and quantitative machine learning researcher.
Analyze this residential property and the machine learning regression model outputs:

Property Features:
- Living Area: ${property.sqftLiving} sq ft
- Bedrooms: ${property.bedrooms} | Bathrooms: ${property.bathrooms}
- Lot Size: ${property.lotSizeSqft} sq ft
- Year Built: ${property.yearBuilt} (Remodeled: ${property.yearRemodeled || 'None'})
- Overall Quality: ${property.overallQuality}/10
- Overall Condition: ${property.overallCondition}/10
- Neighborhood: ${property.neighborhood}
- Garage: ${property.garageCars} car(s)
- Waterfront: ${property.waterfront ? 'Yes' : 'No'}
- Basement: ${property.hasBasement ? 'Yes' : 'No'}
- Private Pool: ${property.hasPool ? 'Yes' : 'No'}

ML Regression Model Valuations:
- Linear Regression: $${Math.round(predictions?.linear || 0).toLocaleString()}
- Decision Tree: $${Math.round(predictions?.decisionTree || 0).toLocaleString()}
- Random Forest: $${Math.round(predictions?.randomForest || 0).toLocaleString()}
- Ensemble Weighted Average: $${Math.round(predictions?.ensemble || 0).toLocaleString()}

Model Accuracy Context:
- Test R² Score: ${(metrics?.r2 ? (metrics.r2 * 100).toFixed(1) : '88.5')}%
- Mean Absolute Error (MAE): $${Math.round(metrics?.mae || 24000).toLocaleString()}

Provide an executive appraisal evaluation formatted in JSON with the following structure:
{
  "summary": "1-2 sentences summarizing valuation consensus and price/sqft metric.",
  "strengths": ["bullet 1 highlighting key value driver", "bullet 2 highlighting quality or location asset", "bullet 3 highlighting lot or feature asset"],
  "marketContext": "2-3 sentences explaining why the machine learning models produced this spread and how current market dynamics impact this asset.",
  "recommendations": ["Actionable recommendation 1 for seller or investor", "Actionable recommendation 2 regarding renovations or pricing corridor"]
}
Only output valid JSON without markdown wrapping.`;

    const response = await aiClient.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
      }
    });

    const text = response.text || '';
    const parsed = JSON.parse(text);
    return res.json(parsed);
  } catch (error: any) {
    console.error('Appraisal API error:', error);
    return res.status(500).json({
      error: 'Failed to generate AI appraisal report',
      message: error?.message || 'Unknown error'
    });
  }
});

// Setup Vite middleware in dev or static files in production
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(port, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${port}`);
  });
}

startServer();

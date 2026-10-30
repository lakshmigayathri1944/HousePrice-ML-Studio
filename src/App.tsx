import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header.tsx';
import { OverviewHero } from './components/OverviewHero.tsx';
import { DataCleaningEDA } from './components/DataCleaningEDA.tsx';
import { ModelComparison } from './components/ModelComparison.tsx';
import { FeatureImportanceView } from './components/FeatureImportance.tsx';
import { PricePredictor } from './components/PricePredictor.tsx';
import { CrossValidationTab } from './components/CrossValidationTab.tsx';
import { PythonExportModal } from './components/PythonExportModal.tsx';
import { PropertyRecord, TrainingConfig, Hyperparameters, ModelResult } from './ml/types.ts';
import { generateRealisticHousingDataset } from './ml/dataset.ts';
import { preprocessDataset } from './ml/preprocessing.ts';
import { LinearRegressionModel } from './ml/linearRegression.ts';
import { DecisionTreeRegressor } from './ml/decisionTree.ts';
import { RandomForestRegressor } from './ml/randomForest.ts';
import { calculateMetrics, buildPredictionPoints } from './ml/evaluation.ts';

export default function App() {
  const [isDark, setIsDark] = useState<boolean>(() => {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  const [activeTab, setActiveTab] = useState<string>('overview');
  const [isPythonModalOpen, setIsPythonModalOpen] = useState(false);
  const [dataset, setDataset] = useState<PropertyRecord[]>(() => generateRealisticHousingDataset(1000));

  const [config, setConfig] = useState<TrainingConfig>({
    trainRatio: 0.8,
    randomSeed: 42,
    filterOutliers: true,
    outlierThresholdIQR: 1.5,
    standardizeFeatures: true,
    missingValueStrategy: 'median',
    logTransformTarget: false,
  });

  const [hyperparams, setHyperparams] = useState<Hyperparameters>({
    linearRidgeLambda: 1.0,
    treeMaxDepth: 6,
    treeMinSamplesSplit: 10,
    forestNumTrees: 18,
    forestMaxDepth: 6,
    forestFeatureSubsampleRatio: 0.7,
  });

  const [models, setModels] = useState<ModelResult[]>([]);
  const [means, setMeans] = useState<number[]>([]);
  const [stds, setStds] = useState<number[]>([]);
  const [isTraining, setIsTraining] = useState<boolean>(false);

  // Sync dark class on document element
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDark]);

  // Train regression models
  const trainAllModels = useCallback(() => {
    setIsTraining(true);

    // Timeout allows UI to render spinner smoothly
    setTimeout(() => {
      try {
        const prep = preprocessDataset(dataset, config);
        setMeans(prep.means);
        setStds(prep.stds);

        const results: ModelResult[] = [];

        // 1. Linear Regression (Ridge)
        const t0Linear = performance.now();
        const lin = new LinearRegressionModel();
        lin.train(prep.X_train, prep.y_train, prep.featureNames, hyperparams.linearRidgeLambda);
        const t1Linear = performance.now();

        const trainPredsLinear = lin.predictBatch(prep.X_train);
        const testPredsLinear = lin.predictBatch(prep.X_test);

        const trainMetricsLinear = calculateMetrics(prep.y_train, trainPredsLinear);
        const testMetricsLinear = calculateMetrics(prep.y_test, testPredsLinear);

        results.push({
          name: 'Linear Regression (Ridge)',
          type: 'linear',
          trainingTimeMs: t1Linear - t0Linear,
          metrics: {
            ...testMetricsLinear,
            trainMae: trainMetricsLinear.mae,
            trainRmse: trainMetricsLinear.rmse,
            trainR2: trainMetricsLinear.r2,
          },
          predictions: buildPredictionPoints(
            prep.rawTest,
            prep.y_test,
            testPredsLinear,
            config.logTransformTarget
          ),
          featureImportances: lin.getFeatureImportances(),
        });

        // 2. Decision Tree Regressor (CART)
        const t0Tree = performance.now();
        const tree = new DecisionTreeRegressor(
          hyperparams.treeMaxDepth,
          hyperparams.treeMinSamplesSplit
        );
        tree.train(prep.X_train, prep.y_train, prep.featureNames);
        const t1Tree = performance.now();

        const trainPredsTree = tree.predictBatch(prep.X_train);
        const testPredsTree = tree.predictBatch(prep.X_test);

        const trainMetricsTree = calculateMetrics(prep.y_train, trainPredsTree);
        const testMetricsTree = calculateMetrics(prep.y_test, testPredsTree);

        results.push({
          name: 'Decision Tree Regressor',
          type: 'decisionTree',
          trainingTimeMs: t1Tree - t0Tree,
          metrics: {
            ...testMetricsTree,
            trainMae: trainMetricsTree.mae,
            trainRmse: trainMetricsTree.rmse,
            trainR2: trainMetricsTree.r2,
          },
          predictions: buildPredictionPoints(
            prep.rawTest,
            prep.y_test,
            testPredsTree,
            config.logTransformTarget
          ),
          featureImportances: tree.getFeatureImportances(),
        });

        // 3. Random Forest Regressor
        const t0Forest = performance.now();
        const forest = new RandomForestRegressor(
          hyperparams.forestNumTrees,
          hyperparams.forestMaxDepth,
          hyperparams.forestFeatureSubsampleRatio
        );
        forest.train(prep.X_train, prep.y_train, prep.featureNames);
        const t1Forest = performance.now();

        const trainPredsForest = forest.predictBatch(prep.X_train);
        const testPredsForest = forest.predictBatch(prep.X_test);

        const trainMetricsForest = calculateMetrics(prep.y_train, trainPredsForest);
        const testMetricsForest = calculateMetrics(prep.y_test, testPredsForest);

        results.push({
          name: 'Random Forest Regressor',
          type: 'randomForest',
          trainingTimeMs: t1Forest - t0Forest,
          metrics: {
            ...testMetricsForest,
            trainMae: trainMetricsForest.mae,
            trainRmse: trainMetricsForest.rmse,
            trainR2: trainMetricsForest.r2,
          },
          predictions: buildPredictionPoints(
            prep.rawTest,
            prep.y_test,
            testPredsForest,
            config.logTransformTarget
          ),
          featureImportances: forest.getFeatureImportances(),
        });

        setModels(results);
      } catch (err) {
        console.error('Error during model training:', err);
      } finally {
        setIsTraining(false);
      }
    }, 40);
  }, [dataset, config, hyperparams]);

  // Initial training
  useEffect(() => {
    trainAllModels();
  }, [trainAllModels]);

  const handleResetDefaultData = () => {
    setDataset(generateRealisticHousingDataset(1000));
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors font-sans flex flex-col">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenPythonModal={() => setIsPythonModalOpen(true)}
        onRetrain={trainAllModels}
        isTraining={isTraining}
        isDark={isDark}
        setIsDark={setIsDark}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {activeTab === 'overview' && (
          <OverviewHero
            onNavigate={(tab) => setActiveTab(tab)}
            datasetSize={dataset.length}
            models={models}
          />
        )}

        {activeTab === 'eda' && (
          <DataCleaningEDA
            dataset={dataset}
            setDataset={setDataset}
            config={config}
            setConfig={setConfig}
            onResetDefaultData={handleResetDefaultData}
          />
        )}

        {activeTab === 'models' && (
          <ModelComparison
            models={models}
            hyperparams={hyperparams}
            setHyperparams={setHyperparams}
            onRetrain={trainAllModels}
            isTraining={isTraining}
          />
        )}

        {activeTab === 'crossValidation' && (
          <CrossValidationTab
            dataset={dataset}
            config={config}
            hyperparams={hyperparams}
          />
        )}

        {activeTab === 'features' && (
          <FeatureImportanceView models={models} dataset={dataset} />
        )}

        {activeTab === 'predictor' && (
          <PricePredictor
            models={models}
            dataset={dataset}
            means={means}
            stds={stds}
            isStandardized={config.standardizeFeatures}
            isLogScale={config.logTransformTarget}
            onOpenPythonModal={() => setIsPythonModalOpen(true)}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 py-6 text-xs text-slate-500 transition-colors mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-800 dark:text-slate-200">
              HousePrice ML Studio
            </span>
            <span>·</span>
            <span>Machine Learning Regression Pipeline</span>
            <span>·</span>
            <span>Scikit-learn, Pandas, NumPy, Matplotlib</span>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsPythonModalOpen(true)}
              className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
            >
              Export Python Code
            </button>
            <button
              onClick={trainAllModels}
              className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
            >
              Retrain Pipeline
            </button>
          </div>
        </div>
      </footer>

      {/* Python Code Modal */}
      <PythonExportModal
        isOpen={isPythonModalOpen}
        onClose={() => setIsPythonModalOpen(false)}
        config={config}
        hyperparams={hyperparams}
      />
    </div>
  );
}

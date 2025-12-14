from agents.model_training.agent_hyperparameter_tuning import HyperparameterTuningAgent
from agents.model_training.model_evaluation import ModelEvaluationAgent
from transformers import pipeline
from sklearn.ensemble import RandomForestRegressor
import json
import os



class ModelTrainingAgent:
    def __init__(self, experiments,orchestrator=None):
        self.experiments = experiments
        self.results = []
        self.llm_pipeline = None
        self.llm_enabled = any(exp.get('use_llm', False) for exp in experiments)
        self.fallback_used = False

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        os.makedirs("results", exist_ok=True)
        evaluator = ModelEvaluationAgent()

        for config in self.experiments:
            model = config['model']
            name = config['name']
            use_llm = config.get('use_llm', False)

            if use_llm:
                print(f"🤖 Applying LLM-guided tuning for {name}...")
                model = self._apply_llm_method(model, X_train, y_train)
            else:
                model.set_params(**config['params'])
                model.fit(X_train, y_train)

            metrics = evaluator.evaluate_model(model, X_test, y_test)
            metrics["name"] = name
            metrics["model"] = model
            self.results.append(metrics)

        return self._select_best_model()

    def _select_best_model(self):
        return max(self.results, key=lambda x: x["R2 Score"])

    def _apply_llm_method(self, model, X, y):
        dataset_info = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "target_type": y.dtype.name
        }

        suggestion = self._query_llm(dataset_info)

        self.fallback_used = False
        try:
            tuner = HyperparameterTuningAgent(model=model)
            best_model, metrics = tuner.bayesian_optimization(
                X, y,
                search_spaces=suggestion["search_space"],
                n_iter=suggestion.get("n_iter", 15)
            )
            print(f"🤖 LLM-guided tuning metrics: {metrics}")

            if not hasattr(best_model, "predict"):
                raise ValueError("Returned model is not fitted.")

        except Exception as e:
            print(f"⚠️ Tuning failed: {e}")
            print("🔁 Using fallback RandomForest instead.")
            best_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            best_model.fit(X, y)
            self.fallback_used = True

        return best_model

    def _query_llm(self, dataset_info):
        generator = pipeline("text-generation", model="distilgpt2", device=-1)

        prompt = f"""
        Suggest hyperparameter ranges for a regression model.
        Dataset info:
        - Samples: {dataset_info['n_samples']}
        - Features: {dataset_info['n_features']}
        - Target type: {dataset_info['target_type']}
        """

        output = generator(prompt, max_length=200, do_sample=False)[0]["generated_text"]
        print("🤖 LLM Raw Output:\n", output)

        return self._get_fallback_params(dataset_info)

    def _get_fallback_params(self, dataset_info):
        """Връща стандартни хиперпараметри, ако LLM не може да генерира валидни."""
        return {
            "search_space": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            },
            "n_iter": 10
        }
from model_training.agent_hyperparameter_tuning import HyperparameterTuningAgent
from model_training.model_evaluation import ModelEvaluationAgent
from transformers import pipeline
import json

class ModelTrainingAgent:
    def __init__(self, experiments):
        self.experiments = experiments
        self.results = []

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        evaluator = ModelEvaluationAgent()  # ✅ instantiate evaluator

        for config in self.experiments:
            model = config['model']
            name = config['name']
            use_llm = config.get('use_llm', False)

            if use_llm:
                model = self._apply_llm_method(model, X_train, y_train)

            model.set_params(**config['params'])
            model.fit(X_train, y_train)

            # ✅ use evaluation agent
            metrics = evaluator.evaluate_model(model, X_test, y_test)
            metrics["name"] = name
            metrics["model"] = model
            self.results.append(metrics)

        return self._select_best_model()

    def _select_best_model(self):
        # ✅ match key name from ModelEvaluationAgent
        return max(self.results, key=lambda x: x["R2 Score"])

    def _apply_llm_method(self, model, X, y):
        dataset_info = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "target_type": y.dtype.name
        }
        suggestion = self._query_llm(dataset_info)

        tuner = HyperparameterTuningAgent(model=model)
        best_model, metrics = tuner.bayesian_optimization(
            X, y,
            search_spaces=suggestion["search_space"],
            n_iter=suggestion.get("n_iter", 15)
        )
        print(f"🤖 LLM-guided tuning metrics: {metrics}")
        return best_model

    def _query_llm(self, dataset_info):
        generator = pipeline("text-generation", model="your-local-model")

        prompt = f"""
        Suggest hyperparameter ranges for a regression model.
        Dataset info:
        - Samples: {dataset_info['n_samples']}
        - Features: {dataset_info['n_features']}
        - Target type: {dataset_info['target_type']}
        
        Output JSON with keys 'search_space' and 'n_iter'.
        """

        output = generator(prompt, max_length=200, do_sample=False)[0]["generated_text"]

        # Parse JSON from model output
        suggestion = json.loads(output.strip())
        return suggestion
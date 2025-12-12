from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

class ModelSelectionAgent:
    def __init__(self, cv=5, scoring='r2', overfit_threshold=0.2, perfect_threshold=0.999):
        self.cv = cv
        self.scoring = scoring
        self.overfit_threshold = overfit_threshold
        self.perfect_threshold = perfect_threshold  # new: threshold for suspiciously perfect fits

    def select_best_model(self, X_train, y_train):
        # Candidate models
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "RandomForest": RandomForestRegressor(),
            "SVR": SVR()
        }

        results = {}
        valid_models = {}

        for name, model in models.items():
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv, scoring=self.scoring)
            cv_mean = cv_scores.mean()

            # Train score
            model.fit(X_train, y_train)
            train_score = r2_score(y_train, model.predict(X_train))

            print(f"✅ {name}: Train R²={train_score:.4f}, CV R²={cv_mean:.4f}")

            # Detect suspiciously perfect fits
            if cv_mean >= self.perfect_threshold or train_score >= 0.9999:
                print(f"⚠️ {name} skipped due to suspiciously perfect fit (CV={cv_mean:.4f}, Train={train_score:.4f})")
                results[name] = {"train": train_score, "cv": cv_mean, "note": "Skipped due to perfect fit"}
                continue

            # Detect overfitting
            if train_score - cv_mean > self.overfit_threshold:
                print(f"⚠️ {name} flagged as OVERFITTING (gap={train_score - cv_mean:.2f})")
            else:
                valid_models[name] = cv_mean

            results[name] = {"train": train_score, "cv": cv_mean}

        # Choose best model
        if not valid_models:
            print("⚠️ No valid models found! Falling back to best non-skipped CV score.")
            best_name = max(results, key=lambda k: results[k]["cv"])
        else:
            best_name = max(valid_models, key=valid_models.get)

        best_model = models[best_name]
        print(f"🏆 Best Model: {best_name} (CV Score: {results[best_name]['cv']:.4f})")
        return best_name, best_model, results

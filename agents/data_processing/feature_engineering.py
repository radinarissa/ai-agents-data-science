import featuretools as ft

class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def generate_features(self, max_depth=1):
        es = ft.EntitySet(id="data")
        es = es.add_dataframe(
            dataframe_name="main",
            dataframe=self.df,
            index=self.df.index.name or "index"
        )
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name="main",
            max_depth=max_depth
        )
        print(f"🧠 Generated {len(feature_defs)} new features.")
        return feature_matrix

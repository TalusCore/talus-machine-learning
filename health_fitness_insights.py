"""
Health and Fitness Insight ML Algorithm - MEMORY OPTIMIZED VERSION
Analyzes user data against reference dataset to identify strengths and struggles
"""

import sys
import io

# Fix encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.filterwarnings('ignore')
import gc


class HealthFitnessInsights:

    NON_ACTIONABLE_COLS = {'age', 'height_cm', 'participant_id', 'bmi', 'gender'}

    METRICS_HIGHER_IS_BETTER = {
        'steps_per_day', 'exercise_minutes_per_day',
        'workout_frequency_per_week', 'fitness_level'
    }

    METRICS_LOWER_IS_BETTER = {
        'weight_kg', 'bmi', 'avg_heart_rate'
    }


    MAX_ROWS = 50_000

    # fitness_level in the dataset is a raw float (e.g. 0.04 → ~100).
    # We store the dataset max so we can convert user input (0–100 scale)
    # back to the raw dataset scale before analysis.
    FITNESS_LEVEL_USER_MAX = 100.0   # what the user sends us
    _fitness_level_dataset_max = None  # learned from the CSV at load time

    def __init__(self, reference_dataset_path=None):
        import os

        if reference_dataset_path is None:
            reference_dataset_path = os.getenv('DATASET_PATH')
            if not reference_dataset_path:
                raise ValueError(
                    "DATASET_PATH not provided as argument or environment variable. "
                    "Pass it directly to __init__ or set the DATASET_PATH env var."
                )

        self.reference_data = None
        self.scaler = StandardScaler()
        self.numeric_cols = []
        self.categorical_cols = []
        self.n_clusters = 5
        self.anomaly_detector = None

        self.load_reference_data(reference_dataset_path)

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_csv_path(self, path):
        import os

        path = os.path.expanduser(str(path))
        path = os.path.abspath(path)

        if os.path.isfile(path):
            if path.endswith('.csv'):
                print(f"   CSV file found directly: {path}")
                return path
            else:
                raise ValueError(f"Path is a file but not a CSV: {path}")

        if os.path.isdir(path):
            TARGET_FILENAME = 'health_fitness_dataset.csv'
            for root, dirs, files in os.walk(path):
                if TARGET_FILENAME in files:
                    found = os.path.join(root, TARGET_FILENAME)
                    print(f"   Found target CSV: {found}")
                    return found
            all_csvs = []
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith('.csv'):
                        all_csvs.append(os.path.join(root, f))
            if all_csvs:
                chosen = all_csvs[0]
                print(f"   Using first CSV found: {chosen}")
                return chosen
            raise ValueError(f"No CSV files found anywhere under directory: {path}")

        raise ValueError(
            f"Path does not exist: {path}\n"
            f"Place health_fitness_dataset.csv in the project root."
        )

    # ------------------------------------------------------------------
    # Fitness level scaling helpers
    #
    # The dataset stores fitness_level as a raw accumulating float
    # (e.g. 0.04, 0.07 … up to the participant's max ~100+).
    # Users submit a 0–100 score, so we learn the dataset max at load
    # time and use it to convert both directions.
    # ------------------------------------------------------------------

    def _learn_fitness_level_scale(self):
        """Store the dataset's max fitness_level so we can rescale user input."""
        if 'fitness_level' in self.reference_data.columns:
            self._fitness_level_dataset_max = float(
                self.reference_data['fitness_level'].max()
            )
            print(f"   fitness_level dataset range: "
                  f"0 – {self._fitness_level_dataset_max:.2f}  "
                  f"(user sends 0–100, we rescale)")
        else:
            self._fitness_level_dataset_max = None

    def _user_fitness_to_dataset_scale(self, user_value: float) -> float:
        """Convert a user-supplied 0–100 score to the raw dataset scale."""
        if self._fitness_level_dataset_max is None:
            return user_value
        return (user_value / self.FITNESS_LEVEL_USER_MAX) * self._fitness_level_dataset_max

    def _dataset_fitness_to_user_scale(self, raw_value: float) -> float:
        """Convert a raw dataset fitness_level back to 0–100 for display."""
        if self._fitness_level_dataset_max is None or self._fitness_level_dataset_max == 0:
            return raw_value
        return (raw_value / self._fitness_level_dataset_max) * self.FITNESS_LEVEL_USER_MAX

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_reference_data(self, path):
        try:
            csv_path = self._resolve_csv_path(path)
            print(f"Loading CSV: {csv_path}")

            self.reference_data = pd.read_csv(csv_path, nrows=self.MAX_ROWS)
            print(f"Loaded {len(self.reference_data):,} rows (cap: {self.MAX_ROWS:,})")

            EXCLUDE_COLS = [
                'participant_id', 'date', 'sleep_hours',
                'water_intake_liters', 'calories_burned'
            ]
            self.reference_data.drop(
                columns=[c for c in EXCLUDE_COLS if c in self.reference_data.columns],
                inplace=True
            )

            # Learn fitness_level scale BEFORE any dtype conversion
            self._learn_fitness_level_scale()

            # Downcast to save RAM
            for col in self.reference_data.select_dtypes(include='float64').columns:
                self.reference_data[col] = self.reference_data[col].astype(np.float32)
            for col in self.reference_data.select_dtypes(include='int64').columns:
                self.reference_data[col] = self.reference_data[col].astype(np.int32)
            for col in self.reference_data.select_dtypes(include='object').columns:
                self.reference_data[col] = self.reference_data[col].astype('category')

            self.numeric_cols = self.reference_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            self.categorical_cols = self.reference_data.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()

            self.reference_medians = self.reference_data[self.numeric_cols].median()
            self.reference_modes = {
                col: self.reference_data[col].mode()[0]
                for col in self.categorical_cols
            }

            print(f"Reference dataset shape: {self.reference_data.shape}")
            print(f"Numeric columns:     {self.numeric_cols}")
            print(f"Categorical columns: {self.categorical_cols}")

            gc.collect()

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_data(self, df, fit=False):
        df_processed = df.copy()

        for col in self.numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        for col in self.numeric_cols:
            if col in df_processed.columns:
                fill_val = self.reference_medians.get(col, 0)
                df_processed[col] = df_processed[col].fillna(fill_val)

        for col in self.categorical_cols:
            if col in df_processed.columns:
                fill_val = self.reference_modes.get(col, "unknown")
                df_processed[col] = df_processed[col].fillna(fill_val)

        for col in self.categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))

        numeric_present = [col for col in self.numeric_cols if col in df_processed.columns]
        if fit:
            self.scaler.fit(df_processed[numeric_present])

        df_processed[numeric_present] = self.scaler.transform(
            df_processed[numeric_present]
        ).astype(np.float32)

        df_processed = df_processed.fillna(0)
        return df_processed

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _get_age_bracket(self, age):
        age = float(age)
        if age < 31:
            return '18-30'
        elif age < 46:
            return '31-45'
        elif age < 61:
            return '46-60'
        else:
            return '60+'

    def perform_clustering(self, max_clusters=6):
        print("\n=== CLUSTERING ANALYSIS (memory-optimised) ===")

        if 'participant_id' in self.reference_data.columns:
            ref_deduped = self.reference_data.groupby('participant_id').mean().reset_index()
        else:
            ref_deduped = self.reference_data.copy()

        self.reference_data['age_bracket'] = self.reference_data['age'].apply(self._get_age_bracket)
        ref_deduped['age_bracket'] = ref_deduped['age'].apply(self._get_age_bracket)

        genders = ref_deduped['gender'].unique() if 'gender' in ref_deduped.columns else ['Unknown']

        self.age_bracket_clusters = {}
        self.age_bracket_scalers = {}
        self.cluster_profiles = {}
        self.training_columns = None

        brackets = ['18-30', '31-45', '46-60', '60+']

        for bracket in brackets:
            for gender in genders:
                if 'gender' in ref_deduped.columns:
                    bracket_data = ref_deduped[
                        (ref_deduped['age_bracket'] == bracket) &
                        (ref_deduped['gender'] == gender)
                    ].copy()
                    bracket_label = f"{bracket} - {gender}"
                else:
                    bracket_data = ref_deduped[ref_deduped['age_bracket'] == bracket].copy()
                    bracket_label = bracket

                print(f"\n--- {bracket_label}: {len(bracket_data)} participants ---")

                if len(bracket_data) < 10:
                    print(f"  Skipping — not enough data")
                    continue

                if len(bracket_data) > 10_000:
                    bracket_data = bracket_data.sample(10_000, random_state=42)

                df_processed = self.preprocess_data(bracket_data, fit=True)
                drop_cols = [c for c in ['age_bracket', 'gender', 'cluster'] if c in df_processed.columns]
                df_processed.drop(columns=drop_cols, inplace=True)

                if self.training_columns is None:
                    self.training_columns = df_processed.columns.tolist()

                optimal_k = min(4, len(bracket_data) // 10)
                optimal_k = max(optimal_k, 2)

                km_final = MiniBatchKMeans(
                    n_clusters=optimal_k,
                    random_state=42,
                    batch_size=1024,
                    n_init=3,
                    max_iter=100
                )
                labels = km_final.fit_predict(df_processed)

                self.age_bracket_clusters[bracket_label] = {
                    'model': km_final,
                    'scaler': self.scaler,
                    'n_clusters': optimal_k
                }

                bracket_ref = self.reference_data[
                    self.reference_data['age_bracket'] == bracket
                ].copy()
                if 'gender' in self.reference_data.columns:
                    bracket_ref = bracket_ref[bracket_ref['gender'] == gender]

                participant_clusters = dict(zip(bracket_data.index, labels))
                self.cluster_profiles[bracket_label] = {}

                for i in range(optimal_k):
                    cluster_indices = [idx for idx, lbl in participant_clusters.items() if lbl == i]
                    cluster_data = bracket_data.loc[bracket_data.index.intersection(cluster_indices)]
                    profile = {}
                    for col in self.numeric_cols:
                        if col in cluster_data.columns:
                            profile[col] = {
                                'mean': float(cluster_data[col].mean()),
                                'std': float(cluster_data[col].std()),
                                'median': float(cluster_data[col].median())
                            }
                    self.cluster_profiles[bracket_label][i] = profile
                    print(f"  Cluster {i}: {len(cluster_indices)} participants")

                del df_processed, bracket_data
                gc.collect()

        del ref_deduped
        gc.collect()

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def train_anomaly_detector(self):
        print("\n=== TRAINING ANOMALY DETECTORS ===")
        self.anomaly_detectors = {}

        for bracket_label in self.age_bracket_clusters:
            age_bracket = bracket_label.split(' - ')[0] if ' - ' in bracket_label else bracket_label

            bracket_data = self.reference_data[
                self.reference_data['age_bracket'] == age_bracket
            ].copy()

            if ' - ' in bracket_label:
                gender = bracket_label.split(' - ')[1]
                if 'gender' in self.reference_data.columns:
                    bracket_data = bracket_data[bracket_data['gender'] == gender]

            if len(bracket_data) == 0:
                print(f"  Skipping {bracket_label} — no data")
                continue

            if len(bracket_data) > 5_000:
                bracket_data = bracket_data.sample(5_000, random_state=42)

            df_processed = self.preprocess_data(bracket_data, fit=False)
            drop_cols = [c for c in ['age_bracket', 'gender', 'cluster'] if c in df_processed.columns]
            df_processed.drop(columns=drop_cols, inplace=True)
            df_processed = df_processed.reindex(columns=self.training_columns, fill_value=0)

            detector = IsolationForest(
                contamination=0.05,
                random_state=42,
                n_estimators=25,
                max_samples=128,
                n_jobs=1
            )
            detector.fit(df_processed)
            self.anomaly_detectors[bracket_label] = detector
            print(f"  Trained detector for {bracket_label}")

            del df_processed, bracket_data
            gc.collect()

    # ------------------------------------------------------------------
    # User analysis
    # ------------------------------------------------------------------

    def analyze_user(self, user_data_dict):
        """
        Analyze a user's fitness data.

        Expects fitness_level on a 0–100 scale (user-facing).
        Internally converts to the raw dataset scale before ML processing,
        then converts percentile results back to 0–100 for the response.
        """
        print("\n" + "="*60)
        print("ANALYZING USER DATA")
        print("="*60)

        # --- Sanitize numeric inputs ---
        sanitized = {}
        for k, v in user_data_dict.items():
            if k in self.numeric_cols:
                try:
                    sanitized[k] = float(v)
                except (ValueError, TypeError):
                    sanitized[k] = float(self.reference_medians.get(k, 0))
            else:
                sanitized[k] = v

        # --- Convert fitness_level from 0–100 → raw dataset scale ---
        if 'fitness_level' in sanitized:
            user_fitness_0_100 = sanitized['fitness_level']
            sanitized['fitness_level'] = self._user_fitness_to_dataset_scale(user_fitness_0_100)
            print(f"   fitness_level: {user_fitness_0_100:.1f}/100 "
                  f"→ dataset scale {sanitized['fitness_level']:.3f}")

        user_data_dict = sanitized

        user_age = user_data_dict.get('age', self.reference_medians.get('age', 35))
        user_bracket = self._get_age_bracket(user_age)
        user_gender = user_data_dict.get('gender', 'Unknown')

        print(f"\nAGE BRACKET: {user_bracket}")
        if user_gender != 'Unknown':
            print(f"   GENDER: {user_gender}")

        bracket_label = (
            f"{user_bracket} - {user_gender}"
            if 'gender' in self.reference_data.columns
            else user_bracket
        )

        if bracket_label not in self.age_bracket_clusters:
            matching = [b for b in self.age_bracket_clusters if b.startswith(user_bracket)]
            bracket_label = matching[0] if matching else list(self.age_bracket_clusters.keys())[0]
            print(f"   (Using {bracket_label} as closest fit)")

        bracket_model = self.age_bracket_clusters[bracket_label]['model']

        user_df = pd.DataFrame([user_data_dict])
        for col in self.reference_data.columns:
            if col not in user_df.columns and col != 'cluster':
                user_df[col] = (
                    self.reference_medians.get(col)
                    if col in self.numeric_cols
                    else self.reference_modes.get(col, 'unknown')
                )

        user_processed = self.preprocess_data(user_df, fit=False)
        drop_cols = [c for c in ['age_bracket', 'gender', 'cluster'] if c in user_processed.columns]
        user_processed.drop(columns=drop_cols, inplace=True)
        user_processed = user_processed.reindex(columns=self.training_columns, fill_value=0)

        user_cluster = bracket_model.predict(user_processed)[0]

        bracket_ref = self.reference_data[self.reference_data['age_bracket'] == user_bracket].copy()
        if 'gender' in self.reference_data.columns:
            bracket_ref = bracket_ref[bracket_ref['gender'] == user_gender]
        cluster_size = len(bracket_ref)

        print(f"\nFITNESS PROFILE: Cluster {user_cluster} (within {bracket_label})")
        print(f"   Compared against {cluster_size} people in your demographic")

        detector = self.anomaly_detectors.get(
            bracket_label, list(self.anomaly_detectors.values())[0]
        )
        anomaly_score = detector.score_samples(user_processed)[0]
        is_anomaly = detector.predict(user_processed)[0] == -1

        print(f"\nANOMALY SCORE: {anomaly_score:.3f}")
        print("   WARNING: Unusual metrics for your age group" if is_anomaly else "   OK: Within normal range")

        print("\nPERCENTILE RANKINGS (vs your age group):")
        percentiles = {}
        struggles = []
        strengths = []

        for col in self.numeric_cols:
            if col in user_data_dict and col not in self.NON_ACTIONABLE_COLS:
                user_value = user_data_dict[col]
                ref_values = bracket_ref[col].dropna()
                percentile = (ref_values < user_value).sum() / len(ref_values) * 100
                percentiles[col] = percentile

                if col in self.METRICS_HIGHER_IS_BETTER:
                    if percentile < 25:
                        struggles.append((col, percentile))
                        print(f"   {col}: {percentile:.1f}th percentile [NEEDS ATTENTION]")
                    elif percentile > 75:
                        strengths.append((col, percentile))
                        print(f"   {col}: {percentile:.1f}th percentile [STRENGTH]")
                elif col in self.METRICS_LOWER_IS_BETTER:
                    if percentile > 75:
                        struggles.append((col, percentile))
                        print(f"   {col}: {percentile:.1f}th percentile [NEEDS ATTENTION]")
                    elif percentile < 25:
                        strengths.append((col, percentile))
                        print(f"   {col}: {percentile:.1f}th percentile [STRENGTH]")

        print(f"\nCOMPARISON TO YOUR FITNESS GROUP (Cluster {user_cluster}, {bracket_label}):")
        cluster_profile = self.cluster_profiles.get(bracket_label, {}).get(user_cluster, {})
        deviations = []

        for col in self.numeric_cols:
            if col in user_data_dict and col in cluster_profile and col not in self.NON_ACTIONABLE_COLS:
                user_value = user_data_dict[col]
                col_mean = cluster_profile[col]['mean']
                col_std = cluster_profile[col]['std']
                if col_std and col_std > 0:
                    z_score = (user_value - col_mean) / col_std
                    deviations.append((col, z_score))
                    if abs(z_score) > 1:
                        direction = "above" if z_score > 0 else "below"
                        print(f"   {col}: {abs(z_score):.2f} std devs {direction} group average")

        print("\nKEY INSIGHTS:")
        if struggles:
            print("\n   Areas that may need attention:")
            for metric, pct in sorted(struggles, key=lambda x: x[1])[:3]:
                print(f"   - {metric}: {pct:.1f}th percentile in your age group")
        if strengths:
            print("\n   Your strengths:")
            for metric, pct in sorted(strengths, key=lambda x: x[1], reverse=True)[:3]:
                print(f"   - {metric}: {pct:.1f}th percentile in your age group")
        if deviations:
            print("\n   Biggest differences from your fitness group:")
            for metric, z_score in sorted(deviations, key=lambda x: abs(x[1]), reverse=True)[:3]:
                direction = "higher" if z_score > 0 else "lower"
                print(f"   - Your {metric} is significantly {direction} than similar people in your demographic")

        recommendations = self._get_recommendations(struggles)

        # --- Convert fitness_level percentile output back to 0–100 context ---
        # The percentile itself is already 0–100, but we include the user's
        # original 0–100 score in the response for clarity.
        response_fitness_level_score = (
            user_fitness_0_100 if 'fitness_level' in sanitized else None
        )

        result = {
            'age_bracket': user_bracket,
            'cluster': int(user_cluster),
            'cluster_size': int(cluster_size),
            'cluster_percentage': float(cluster_size / len(self.reference_data) * 100),
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly),
            'percentiles': {k: float(v) for k, v in percentiles.items()},
            'struggles': [(k, float(v)) for k, v in struggles],
            'strengths': [(k, float(v)) for k, v in strengths],
            'cluster_deviations': [(k, float(v)) for k, v in deviations],
            'recommendations': recommendations,
        }

        if response_fitness_level_score is not None:
            result['fitness_level_score'] = float(response_fitness_level_score)

        return result

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _get_recommendations(self, struggles):
        recommendations_map = {
            'steps_per_day': [
                "Try to increase daily steps by 1000-2000",
                "Take short walking breaks throughout the day",
                "Use stairs instead of elevators when possible",
                "Park farther away to add extra walking distance"
            ],
            'workout_frequency_per_week': [
                "Start with small, achievable workout goals",
                "Try to add 1-2 more workout sessions per week",
                "Schedule workouts like regular appointments",
                "Find a workout buddy for accountability"
            ],
            'exercise_minutes_per_day': [
                "Gradually increase exercise duration by 5-10 minutes per week",
                "Break workouts into shorter sessions if needed",
                "Mix cardio, strength, and flexibility training",
                "Start with low-impact activities if you're just beginning"
            ],
            'avg_heart_rate': [
                "Work on cardiovascular fitness through aerobic activity",
                "Include 30+ minutes of moderate cardio 4-5 times per week",
                "Consult with a healthcare provider for personalized advice",
                "Track your average heart rate to monitor improvement"
            ],
            'weight_kg': [
                "Focus on balanced nutrition and portion control",
                "Combine cardio with strength training",
                "Aim for gradual weight loss of 0.5-1 kg per week",
                "Consult with a nutritionist for a personalized plan"
            ],
            'bmi': [
                "Work on diet and exercise habits simultaneously",
                "Increase protein intake and reduce processed foods",
                "Aim for 150+ minutes of moderate activity per week",
                "Track your BMI progress over time"
            ],
            'fitness_level': [
                "Consistently track your workouts to build fitness over time",
                "Aim to increase workout duration or intensity each week",
                "Mix different activity types to improve overall fitness",
                "Set a goal and monitor your progress monthly"
            ]
        }
        return [
            {'metric': metric, 'tips': recommendations_map[metric]}
            for metric, _ in struggles
            if metric in recommendations_map
        ]

    # ------------------------------------------------------------------
    # Batch analysis & cluster summary
    # ------------------------------------------------------------------

    def batch_analyze(self, user_dataset_path):
        print("\n=== Batch Analysis Mode ===")
        user_data = pd.read_csv(user_dataset_path)
        print(f"Loaded {len(user_data)} users for analysis")
        all_insights = []
        for idx, row in user_data.iterrows():
            print(f"\nAnalyzing user {idx + 1}/{len(user_data)}...")
            insights = self.analyze_user(row.to_dict())
            insights['user_id'] = idx
            all_insights.append(insights)
        return pd.DataFrame(all_insights)

    def get_cluster_summary(self):
        print("\n=== CLUSTER SUMMARY (by Age Bracket & Gender) ===")
        for bracket_label, clusters in self.cluster_profiles.items():
            print(f"\n{'='*50}")
            print(f"GROUP: {bracket_label}")
            age_bracket = bracket_label.split(' - ')[0] if ' - ' in bracket_label else bracket_label
            bracket_ref = self.reference_data[self.reference_data['age_bracket'] == age_bracket]
            if ' - ' in bracket_label:
                gender = bracket_label.split(' - ')[1]
                if 'gender' in self.reference_data.columns:
                    bracket_ref = bracket_ref[bracket_ref['gender'] == gender]
            print(f"Total participants: {len(bracket_ref)}")
            print(f"{'='*50}")
            for cluster_id, profile in clusters.items():
                print(f"\n  Cluster {cluster_id}:")
                for col in self.numeric_cols[:5]:
                    if col in profile:
                        print(f"    {col}: {profile[col]['mean']:.2f} avg (±{profile[col]['std']:.2f})")
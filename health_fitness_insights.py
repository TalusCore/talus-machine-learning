"""
Health and Fitness Insight ML Algorithm - OPTIMIZED VERSION
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
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class HealthFitnessInsights:

    NON_ACTIONABLE_COLS = {'age', 'height_cm', 'participant_id', 'bmi', 'gender'}
    
    # Metrics where HIGHER percentile = better performance
    METRICS_HIGHER_IS_BETTER = {
        'steps_per_day', 'exercise_minutes_per_day', 
        'workout_frequency_per_week', 'fitness_level'
    }
    
    # Metrics where LOWER percentile = better performance
    METRICS_LOWER_IS_BETTER = {
        'weight_kg', 'bmi', 'heart_rate_resting'
    }

    """
    ML system to analyze user health/fitness data and provide insights
    on where they struggle compared to a reference population
    """
    
    def __init__(self, reference_dataset_path=None):
        """
        Initialize with path to the Kaggle health and fitness dataset
        
        Args:
            reference_dataset_path: Path to the downloaded dataset
                                   If None, will read from DATASET_PATH environment variable
        """
        import os
        from dotenv import load_dotenv
        
        # Load environment variables from .env file
        load_dotenv()
        
        # If no path provided, get from environment variable
        if reference_dataset_path is None:
            reference_dataset_path = os.getenv('DATASET_PATH')
            if not reference_dataset_path:
                raise ValueError(
                    "DATASET_PATH not provided as argument or environment variable. "
                    "Set it in .env file or pass it to __init__"
                )
        
        self.reference_data = None
        self.user_data = None
        self.scaler = StandardScaler()
        self.numeric_cols = []
        self.categorical_cols = []
        self.clusters = None
        self.n_clusters = 5
        self.anomaly_detector = None
        
        # Load reference dataset
        self.load_reference_data(reference_dataset_path)
        
    def load_reference_data(self, path):
        """Load and preprocess the reference dataset"""
        try:
            # Try common file formats
            import os
            from pathlib import Path
            
            # Expand user home directory if path starts with ~
            path = os.path.expanduser(path)
            
            if os.path.isfile(path):
                if path.endswith('.csv'):
                    self.reference_data = pd.read_csv(path)
                elif path.endswith('.xlsx'):
                    self.reference_data = pd.read_excel(path)
            else:
                # Try to find CSV files in directory
                if os.path.isdir(path):
                    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
                    if csv_files:
                        self.reference_data = pd.read_csv(os.path.join(path, csv_files[0]))
                        print(f"Loaded: {csv_files[0]}")
                    else:
                        raise ValueError(f"No CSV files found in directory: {path}")
                else:
                    raise ValueError(f"Path does not exist: {path}")
            
            print(f"Reference dataset loaded: {self.reference_data.shape}")
            print(f"Columns: {list(self.reference_data.columns)}")
            
            # Drop columns that should never be used as features
            EXCLUDE_COLS = ['participant_id', 'date', 'sleep_hours', 'water_intake_liters', 'calories_burned']
            self.reference_data = self.reference_data.drop(
                columns=[c for c in EXCLUDE_COLS if c in self.reference_data.columns]
            )

            # NOW identify column types (after dropping noise columns)
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

            print(f"\nNumeric columns: {self.numeric_cols}")
            print(f"Categorical columns: {self.categorical_cols}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, df, fit=False):
        """Preprocess data for ML analysis"""
        df_processed = df.copy()

        # Coerce all expected numeric columns to numeric dtype first
        for col in self.numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # Fill missing numeric values using REFERENCE medians (not current df median)
        # This is critical for single-row user data where df.median() returns NaN
        for col in self.numeric_cols:
            if col in df_processed.columns:
                fill_val = self.reference_medians.get(col, 0)
                df_processed[col] = df_processed[col].fillna(fill_val)

        # Fill missing categorical values using reference modes
        for col in self.categorical_cols:
            if col in df_processed.columns:
                fill_val = self.reference_modes.get(col, "unknown")
                df_processed[col] = df_processed[col].fillna(fill_val)

        # Encode categorical variables
        for col in self.categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))

        # Scale numeric features
        numeric_present = [col for col in self.numeric_cols if col in df_processed.columns]
        if fit:
            self.scaler.fit(df_processed[numeric_present])

        df_processed[numeric_present] = self.scaler.transform(df_processed[numeric_present])

        # Final safety net: replace any remaining NaNs with 0 (post-scaling mean)
        df_processed = df_processed.fillna(0)

        return df_processed
    
    def _get_recommendations(self, struggles):
        """Generate actionable recommendations based on identified struggles"""
        recommendations = []
        
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
            'heart_rate_resting': [
                "Work on cardiovascular fitness through aerobic activity",
                "Include 30+ minutes of moderate cardio 4-5 times per week",
                "Consult with a healthcare provider for personalized advice",
                "Track your resting heart rate to monitor improvement"
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
            ]
        }
        
        for metric, _ in struggles:
            if metric in recommendations_map:
                recommendations.append({
                    'metric': metric,
                    'tips': recommendations_map[metric]
                })
        
        return recommendations
    
    def _get_age_bracket(self, age):
        """Assign an age bracket label based on age value"""
        age = float(age)
        if age < 31:
            return '18-30'
        elif age < 46:
            return '31-45'
        elif age < 61:
            return '46-60'
        else:
            return '60+'

    def perform_clustering(self, max_clusters=8):
        """Cluster reference population within age brackets and gender"""
        print("\n=== PERFORMING AGE & GENDER-BRACKETED CLUSTERING ANALYSIS ===")

        # Deduplicate to one row per participant using mean of their records
        if 'participant_id' in self.reference_data.columns:
            ref_deduped = self.reference_data.groupby('participant_id').mean().reset_index()
        else:
            ref_deduped = self.reference_data.copy()

        # Assign age brackets
        self.reference_data['age_bracket'] = self.reference_data['age'].apply(self._get_age_bracket)
        ref_deduped['age_bracket'] = ref_deduped['age'].apply(self._get_age_bracket)

        # Get unique genders (handle if gender column doesn't exist)
        if 'gender' in ref_deduped.columns:
            genders = ref_deduped['gender'].unique()
        else:
            genders = ['Unknown']

        self.age_bracket_clusters = {}   # stores KMeans model per bracket+gender
        self.age_bracket_scalers = {}    # stores scaler per bracket+gender
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
                
                print(f"\n--- Processing {bracket_label} ---")
                print(f"  {len(bracket_data)} participants in this group")

                if len(bracket_data) < 10:
                    print(f"  Skipping {bracket_label} — not enough data")
                    continue

                # Preprocess (fit scaler per bracket)
                df_processed = self.preprocess_data(bracket_data, fit=True)

                # Drop non-feature columns
                drop_cols = [c for c in ['age_bracket', 'gender', 'cluster'] if c in df_processed.columns]
                df_processed = df_processed.drop(columns=drop_cols)

                if self.training_columns is None:
                    self.training_columns = df_processed.columns.tolist()

                # Find optimal k via elbow method
                max_k = min(max_clusters + 1, len(bracket_data) // 10)
                k_range = range(3, max(4, max_k))
                inertias = []

                for k in k_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
                    km.fit(df_processed)
                    inertias.append(km.inertia_)

                # Elbow detection
                if len(inertias) > 2:
                    deltas = np.diff(inertias)
                    delta_deltas = np.diff(deltas)
                    optimal_idx = np.argmax(delta_deltas) + 2
                    optimal_k = list(k_range)[min(optimal_idx, len(k_range) - 1)]
                else:
                    optimal_k = list(k_range)[0]

                print(f"  Optimal clusters for {bracket_label}: {optimal_k}")

                # Fit final model for this bracket
                km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=100)
                labels = km_final.fit_predict(df_processed)

                self.age_bracket_clusters[bracket_label] = {
                    'model': km_final,
                    'scaler': self.scaler,  # already fit above via preprocess_data
                    'n_clusters': optimal_k
                }

                # Store cluster profiles for this bracket
                bracket_ref = self.reference_data[self.reference_data['age_bracket'] == bracket].copy()
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
                                'mean': cluster_data[col].mean(),
                                'std': cluster_data[col].std(),
                                'median': cluster_data[col].median()
                            }
                    self.cluster_profiles[bracket_label][i] = profile
                    print(f"  Cluster {i}: {len(cluster_indices)} participants")
    
    def train_anomaly_detector(self):
        """Train one anomaly detector per age bracket and gender combination"""
        print("\n=== TRAINING ANOMALY DETECTORS (PER AGE BRACKET & GENDER) ===")

        self.anomaly_detectors = {}

        for bracket_label, bracket_info in self.age_bracket_clusters.items():
            # Extract age bracket from label (e.g., "31-45" from "31-45 - Male")
            age_bracket = bracket_label.split(' - ')[0] if ' - ' in bracket_label else bracket_label
            
            # Start with age bracket data
            bracket_data = self.reference_data[self.reference_data['age_bracket'] == age_bracket].copy()
            
            # Further filter by gender if applicable
            if ' - ' in bracket_label:
                gender = bracket_label.split(' - ')[1]
                if 'gender' in self.reference_data.columns:
                    bracket_data = bracket_data[bracket_data['gender'] == gender]
            
            if len(bracket_data) == 0:
                print(f"  Skipping {bracket_label} — no data found")
                continue
            
            df_processed = self.preprocess_data(bracket_data, fit=False)
            drop_cols = [c for c in ['age_bracket', 'gender', 'cluster'] if c in df_processed.columns]
            df_processed = df_processed.drop(columns=drop_cols)
            df_processed = df_processed.reindex(columns=self.training_columns, fill_value=0)

            detector = IsolationForest(
                contamination=0.05,  # Only flag top 5% as anomalies instead of 10%
                random_state=42,
                n_estimators=50,
                max_samples=min(256, len(df_processed)),
                n_jobs=-1
            )
            detector.fit(df_processed)
            self.anomaly_detectors[bracket_label] = detector
            print(f"  Trained detector for {bracket_label}")
    
    def analyze_user(self, user_data_dict):
        print("\n" + "="*60)
        print("ANALYZING USER DATA")
        print("="*60)

        # Sanitize inputs
        sanitized = {}
        for k, v in user_data_dict.items():
            if k in self.numeric_cols:
                try:
                    sanitized[k] = float(v)
                except (ValueError, TypeError):
                    sanitized[k] = float(self.reference_medians.get(k, 0))
            else:
                sanitized[k] = v
        user_data_dict = sanitized

        # Determine user's age bracket and gender
        user_age = user_data_dict.get('age', self.reference_medians.get('age', 35))
        user_bracket = self._get_age_bracket(user_age)
        user_gender = user_data_dict.get('gender', 'Unknown')
        
        print(f"\nAGE BRACKET: {user_bracket}")
        if user_gender != 'Unknown':
            print(f"   GENDER: {user_gender}")

        # Create bracket+gender label
        if 'gender' in self.reference_data.columns:
            bracket_label = f"{user_bracket} - {user_gender}"
        else:
            bracket_label = user_bracket

        # Fall back to nearest bracket if user's bracket has no model
        if bracket_label not in self.age_bracket_clusters:
            # Try to find a matching gender-age bracket
            matching_brackets = [b for b in self.age_bracket_clusters.keys() if b.startswith(user_bracket)]
            if matching_brackets:
                bracket_label = matching_brackets[0]
                print(f"   (No exact match, using {bracket_label} as closest fit)")
            else:
                bracket_label = list(self.age_bracket_clusters.keys())[0]
                print(f"   (No model for your group, using {bracket_label} as fallback)")

        bracket_model = self.age_bracket_clusters[bracket_label]['model']

        # Build user DataFrame and preprocess
        user_df = pd.DataFrame([user_data_dict])
        for col in self.reference_data.columns:
            if col not in user_df.columns and col != 'cluster':
                user_df[col] = self.reference_medians.get(col) if col in self.numeric_cols else self.reference_modes.get(col, 'unknown')

        user_processed = self.preprocess_data(user_df, fit=False)
        drop_cols = [c for c in ['age_bracket', 'gender', 'cluster'] if c in user_processed.columns]
        user_processed = user_processed.drop(columns=drop_cols)
        user_processed = user_processed.reindex(columns=self.training_columns, fill_value=0)

        # Cluster assignment within age+gender bracket
        user_cluster = bracket_model.predict(user_processed)[0]
        bracket_ref = self.reference_data[self.reference_data['age_bracket'] == user_bracket].copy()
        if 'gender' in self.reference_data.columns:
            bracket_ref = bracket_ref[bracket_ref['gender'] == user_gender]
        cluster_size = len(bracket_ref)  # size of group as context

        print(f"\nFITNESS PROFILE: Cluster {user_cluster} (within {bracket_label})")
        print(f"   Compared against {cluster_size} people in your demographic")

        # Anomaly detection within bracket
        detector = self.anomaly_detectors.get(bracket_label)
        if detector is None:
            # Fallback to any available detector
            detector = list(self.anomaly_detectors.values())[0]
        anomaly_score = detector.score_samples(user_processed)[0]
        is_anomaly = detector.predict(user_processed)[0] == -1

        print(f"\nANOMALY SCORE: {anomaly_score:.3f}")
        if is_anomaly:
            print("   WARNING: Your metrics are unusual compared to your age group")
        else:
            print("   OK: Your metrics are within normal range for your age group")

        # --- Percentile rankings vs age bracket peers ---
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

                # Determine if this is a struggle or strength based on metric type
                if col in self.METRICS_HIGHER_IS_BETTER:
                    # For these metrics, HIGH percentile is good
                    if percentile < 25:
                        struggles.append((col, percentile))
                        print(f"   {col}: {percentile:.1f}th percentile [NEEDS ATTENTION]")
                    elif percentile > 75:
                        strengths.append((col, percentile))
                        print(f"   {col}: {percentile:.1f}th percentile [STRENGTH]")
                        
                elif col in self.METRICS_LOWER_IS_BETTER:
                    # For these metrics, LOW percentile is good
                    if percentile > 75:
                        struggles.append((col, percentile))
                        print(f"   {col}: {percentile:.1f}th percentile [NEEDS ATTENTION]")
                    elif percentile < 25:
                        strengths.append((col, percentile))
                        print(f"   {col}: {percentile:.1f}th percentile [STRENGTH]")

        # Cluster deviation vs age+gender bracket cluster profile
        print(f"\nCOMPARISON TO YOUR FITNESS GROUP (Cluster {user_cluster}, {bracket_label}):")
        cluster_profile = self.cluster_profiles.get(bracket_label, {}).get(user_cluster, {})
        deviations = []

        for col in self.numeric_cols:
            if col in user_data_dict and col in cluster_profile:
                # Skip non-actionable columns (age, height, bmi, gender, etc.)
                if col in self.NON_ACTIONABLE_COLS:
                    continue
                    
                user_value = user_data_dict[col]
                col_mean = cluster_profile[col]['mean']
                col_std = cluster_profile[col]['std']

                if col_std and col_std > 0:
                    z_score = (user_value - col_mean) / col_std
                    deviations.append((col, z_score))
                    if abs(z_score) > 1:
                        direction = "above" if z_score > 0 else "below"
                        print(f"   {col}: {abs(z_score):.2f} std devs {direction} group average")

        # Key insights
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

        # Generate recommendations
        recommendations = self._get_recommendations(struggles)

        insights = {
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

        return insights
    
    def batch_analyze(self, user_dataset_path):
        """
        Analyze multiple users from a dataset
        
        Args:
            user_dataset_path: Path to CSV file with user data
            
        Returns:
            DataFrame with insights for all users
        """
        print("\n=== Batch Analysis Mode ===")
        
        user_data = pd.read_csv(user_dataset_path)
        print(f"Loaded {len(user_data)} users for analysis")
        
        all_insights = []
        
        for idx, row in user_data.iterrows():
            print(f"\nAnalyzing user {idx + 1}/{len(user_data)}...")
            user_dict = row.to_dict()
            insights = self.analyze_user(user_dict)
            insights['user_id'] = idx
            all_insights.append(insights)
        
        return pd.DataFrame(all_insights)
    
    def get_cluster_summary(self):
        """Get summary statistics for all clusters, organized by age bracket and gender"""
        print("\n=== CLUSTER SUMMARY (by Age Bracket & Gender) ===")

        for bracket_label, clusters in self.cluster_profiles.items():
            print(f"\n{'='*50}")
            print(f"GROUP: {bracket_label}")
            
            # Extract age bracket from label
            age_bracket = bracket_label.split(' - ')[0] if ' - ' in bracket_label else bracket_label
            bracket_ref = self.reference_data[self.reference_data['age_bracket'] == age_bracket]
            
            # Further filter by gender if applicable
            if ' - ' in bracket_label:
                gender = bracket_label.split(' - ')[1]
                if 'gender' in self.reference_data.columns:
                    bracket_ref = bracket_ref[bracket_ref['gender'] == gender]
            
            print(f"Total participants: {len(bracket_ref)}")
            print(f"{'='*50}")

            for cluster_id, profile in clusters.items():
                print(f"\n  Cluster {cluster_id}:")

                for col in self.numeric_cols[:5]:  # Show first 5 numeric columns
                    if col in profile:
                        mean_val = profile[col]['mean']
                        std_val = profile[col]['std']
                        print(f"    {col}: {mean_val:.2f} avg (±{std_val:.2f})")

# Example usage function
def main():
    """Example of how to use the HealthFitnessInsights system"""
    
    # Path to the downloaded Kaggle dataset
    dataset_path = "/path/to/downloaded/dataset.csv"  # Update this path
    
    # Initialize the system
    print("Initializing Health & Fitness Insights System...")
    hfi = HealthFitnessInsights(dataset_path)
    
    # Perform clustering to identify fitness profiles
    hfi.perform_clustering()
    
    # Train anomaly detector
    hfi.train_anomaly_detector()
    
    # Show cluster summary
    hfi.get_cluster_summary()
    
    # Example: Analyze a single user
    example_user = {
        # Add your user's data here matching the dataset columns
    }
    
    # Analyze the user
    insights = hfi.analyze_user(example_user)
    
    return insights


if __name__ == "__main__":
    print("Health & Fitness Insights ML Algorithm - OPTIMIZED VERSION")
    print("=" * 60)
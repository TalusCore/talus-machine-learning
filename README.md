# Health & Fitness Insights ML System - User Guide

## Overview
This ML system analyzes user health and fitness data against a reference population to identify:
- Where users struggle compared to peers
- Their strengths and weaknesses
- Which fitness profile group they belong to
- Unusual or concerning metrics

## Installation

```bash
pip install kagglehub pandas numpy scikit-learn
```

## Step 1: Download the Dataset

```python
import kagglehub

# Download the health and fitness dataset
path = kagglehub.dataset_download("evan65549/health-and-fitness-dataset")
print("Path to dataset files:", path)
```

## Step 2: Initialize the System

```python
from health_fitness_insights import HealthFitnessInsights

# Initialize with the downloaded dataset path
hfi = HealthFitnessInsights(path)

# Perform clustering to identify fitness profiles
hfi.perform_clustering()

# Train anomaly detector
hfi.train_anomaly_detector()
```

## Step 3: Analyze User Data

### Single User Analysis

```python
# Example user data (adjust based on actual dataset columns)
user_data = {
    'age': 28,
    'weight': 70,
    'height': 170,
    'bmi': 24.2,
    'steps_per_day': 6500,
    'calories_burned': 2100,
    'sleep_hours': 6.5,
    'water_intake': 1.8,
    'workout_frequency': 3,
    'heart_rate': 72
}

# Get insights
insights = hfi.analyze_user(user_data)
```

### Batch Analysis

```python
# Analyze multiple users from a CSV file
insights_df = hfi.batch_analyze('user_data.csv')

# Save results
insights_df.to_csv('user_insights.csv', index=False)
```

## Understanding the Output

The system provides several types of insights:

### 1. **Fitness Profile (Cluster)**
- Assigns user to a group with similar characteristics
- Shows what percentage of the population has a similar profile

### 2. **Anomaly Score**
- Indicates if user's metrics are unusual
- Negative scores closer to 0 are more normal
- Highly negative scores indicate outliers

### 3. **Percentile Rankings**
- Shows where user stands for each metric
- Lower percentiles indicate areas that may need attention
- Higher percentiles indicate strengths

### 4. **Cluster Comparison**
- Compares user to others in their fitness profile group
- Shows which metrics deviate significantly

### 5. **Struggles and Strengths**
- Automatically identifies bottom 25% (struggles)
- Automatically identifies top 25% (strengths)

## Example Output

```
============================================================
ANALYZING USER DATA
============================================================

📊 FITNESS PROFILE: Cluster 2
   Similar to 234 people (23.4% of population)

🎯 ANOMALY SCORE: -0.142
   ✓ Your metrics are within normal range

📈 PERCENTILE RANKINGS:
   steps_per_day: 35.2th percentile
   sleep_hours: 28.7th percentile
   calories_burned: 42.1th percentile
   workout_frequency: 65.3th percentile
   water_intake: 78.9th percentile

🔍 COMPARISON TO YOUR FITNESS PROFILE GROUP (Cluster 2):
   steps_per_day: 1.2 std devs below group average
   sleep_hours: 1.8 std devs below group average

💡 KEY INSIGHTS:

   Areas that may need attention:
   • sleep_hours: You're in the bottom 25% (percentile: 28.7)
   • steps_per_day: You're in the bottom 25% (percentile: 35.2)

   Your strengths:
   • water_intake: You're in the top 25% (percentile: 78.9)

   Biggest differences from your fitness profile group:
   • Your sleep_hours is significantly lower than similar individuals
   • Your steps_per_day is significantly lower than similar individuals
```

## Advanced Features

### View Cluster Summaries

```python
# Get summary of all fitness profiles
hfi.get_cluster_summary()
```

### Access Raw Insights Data

```python
insights = hfi.analyze_user(user_data)

# Access specific insights
print(f"User cluster: {insights['cluster']}")
print(f"Percentiles: {insights['percentiles']}")
print(f"Struggles: {insights['struggles']}")
print(f"Strengths: {insights['strengths']}")
```

### Customize Analysis

You can modify the source code to:
- Adjust struggle/strength thresholds (default: 25th/75th percentiles)
- Change the number of clusters
- Modify anomaly detection sensitivity
- Add custom metrics or calculations

## Tips for Best Results

1. **Complete Data**: Ensure user data includes all columns from the reference dataset
2. **Consistent Units**: Match the units used in the reference dataset
3. **Regular Updates**: Re-train models periodically as new reference data becomes available
4. **Context Matters**: Some metrics (like BMI) may need domain-specific interpretation
5. **Multiple Analyses**: Track changes over time by running analyses periodically

## Interpreting Struggles

When the system identifies "struggles," consider:
- **Context**: Some low percentiles might be acceptable (e.g., resting heart rate)
- **Trends**: A single data point vs. consistent patterns
- **Individual Goals**: What matters most to the user
- **Medical Advice**: Always consult healthcare professionals for medical concerns

## Common Use Cases

1. **Personal Fitness Tracking**: Monitor progress against population norms
2. **Coaching**: Identify areas where clients need support
3. **Research**: Analyze patterns in health data
4. **App Integration**: Provide automated insights to users
5. **Comparative Studies**: Understand how groups differ

## Troubleshooting

**Issue**: "Column not found" error
- **Solution**: Ensure user data columns match reference dataset columns

**Issue**: Poor clustering results
- **Solution**: May need more data or feature engineering

**Issue**: All users flagged as anomalies
- **Solution**: Adjust contamination parameter in anomaly detector

## Next Steps

1. Customize the struggle/strength identification logic for your use case
2. Add visualization capabilities (plots, charts)
3. Integrate with a web app or mobile app
4. Add temporal analysis to track changes over time
5. Implement recommendation systems based on insights

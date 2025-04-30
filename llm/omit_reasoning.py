import pandas as pd

# Load the original CSV
input_file = "llm_prediction_results.csv"         # Change to your file path if needed
output_file = "llm_prediction_results_without_reasoning.csv"         # Output file without the 'reasoning' column

# Read the CSV as-is
df = pd.read_csv(input_file)

# Drop the 'reasoning' column only
df.drop(columns=['reasoning'], inplace=True)

# Save the cleaned CSV with the exact same format (no index)
df.to_csv(output_file, index=False)

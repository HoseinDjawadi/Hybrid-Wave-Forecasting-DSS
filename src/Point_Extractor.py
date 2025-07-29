import pandas as pd
import matplotlib.pyplot as plt
# Load CSV
csv_path = "/home/artorias/Desktop/RA - Oman - Modeling & Data/Paper3_OLD_Failed/Latest_Modifications/Bias_Correction/catboost_corrected/2020_corrected_spatiotemporal.csv"  # Replace with actual path
df = pd.read_csv(csv_path)

# Drop rows with missing coordinates
df = df.dropna(subset=['lat', 'lon'])

# Get unique coordinate pairs
unique_coords = df[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
print(f"Total unique points: {len(unique_coords)}")
print(unique_coords)

# Plot map of points
plt.figure(figsize=(8, 6))
plt.scatter(unique_coords['lon'], unique_coords['lat'], c='red', s=30, edgecolor='k')
plt.title("Unique Coordinate Points")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Save unique coordinates to CSV
output_path = "/home/artorias/Desktop/Paper3_NEW_RETHINKED/unique_coordinates.csv"  # Change path if needed
unique_coords.to_csv(output_path, index=False)
print(f"Saved unique coordinates to {output_path}")

import pandas as pd

# Read CSV file
df = pd.read_csv("cars.csv")

# Display dataset
print("Original Dataset:")
print(df)

# Display records where Buy Price >= 3000
print("\nCars with Buy Price >= 3000:")
filtered_data = df[df["Buy Price"] >= 3000]
print(filtered_data)

# Sort the car data in ascending order (by Sell Price)
print("\nSorted Data (Ascending by Sell Price):")
sorted_data = df.sort_values(by="Sell Price", ascending=True)
print(sorted_data)

# Group cars according to Model
print("\nCars grouped by Model:")
grouped_data = df.groupby("Model")

for model, data in grouped_data:
    print("\nModel:", model)
    print(data)

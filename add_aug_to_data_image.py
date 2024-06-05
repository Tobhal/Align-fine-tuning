import pandas as pd

# Load the original CSV file
df = pd.read_csv('../data/czsl/DATA_ROOT/BengaliWords/BengaliWords_CroppedVersion_Folds/Fold0_use_50_superaug/train_pairs.csv')

# Create an empty DataFrame to hold the augmented data
augmented_df = pd.DataFrame(columns=df.columns)

# Generate augmented filenames
for index, row in df.iterrows():
    base_name = row['Image'].rsplit('.', 1)[0]
    extension = row['Image'].split('.')[-1]
    
    # Append original filename first
    augmented_df = augmented_df.append(row, ignore_index=True)
    
    # Generate and append augmented filenames
    for i in range(20):
        augmented_row = row.copy()
        augmented_row['Image'] = f"{base_name}_aug{i}.{extension}"
        augmented_df = augmented_df.append(augmented_row, ignore_index=True)

# Write the augmented data to a new CSV file
augmented_df.to_csv('data.csv', index=False)

print("Augmented filenames written to data.csv")

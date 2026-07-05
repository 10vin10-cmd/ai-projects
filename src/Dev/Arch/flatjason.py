import pandas as pd

nested_data = [
    {
        "id": 1,
        "info": {"name": "Alice", "age": 28}
    },
    {
        "id": 2,
        "info": {"name": "Bob", "age": 32}
    }
]

# Flatten the nested JSON structure into a DataFrame
df = pd.json_normalize(nested_data, sep='_')
print(df)
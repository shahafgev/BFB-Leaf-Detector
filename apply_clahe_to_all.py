from src.preprocessing.enhancement import batch_apply_clahe

input_folder = "data"
output_folder = "clahe_leaves"

batch_apply_clahe(input_folder, output_folder)

import os
import pandas as pd

def merge_excel_files(input_folder, output_file):
    all_data = []
    total_files = 0
    processed_files = 0

    for file in os.listdir(input_folder):
        if file.endswith('.xlsx'):
            total_files += 1

    for file in os.listdir(input_folder):
        if file.endswith('.xlsx'):
            file_path = os.path.join(input_folder, file)
            df = pd.read_excel(file_path)
            all_data.append(df)

            processed_files += 1
            progress = processed_files / total_files * 100
            print(f"Processed file {processed_files}/{total_files} - {file} [{progress:.2f}%]")

    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_excel(output_file, index=False)

    print("Excel files merged successfully.")

if __name__ == '__main__':
    input_folder = r"D:\Users\LCY\PycharmProjects\pythonProject1\QL_ZH_03"
    output_file = r"D:\Users\LCY\PycharmProjects\pythonProject1\merged_color_clusters.xlsx"

    merge_excel_files(input_folder, output_file)

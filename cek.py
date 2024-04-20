import pandas as pd

def process_data_and_save(input_file, output_file):
    data = pd.read_csv(input_file)
    new_data = []
    skipped_rows = 0
    for index, row in data.iterrows():
        if len(row['content'].split()) <= 250:
            new_data.append(row['content'])
        else:
            skipped_rows += 1
            print(f"Skipped row {index} due to size mismatch: {len(row['content'].split())}")
    new_df = pd.DataFrame(new_data, columns=['content'])
    new_df.to_csv(output_file, index=False)
    print(f"Data processing complete. Skipped {skipped_rows} rows.")

def main():
    input_file = 'raw_data_baru_content.csv'
    output_file = 'data_baru_all.csv'
    process_data_and_save(input_file, output_file)

if __name__ == "__main__":
    main()

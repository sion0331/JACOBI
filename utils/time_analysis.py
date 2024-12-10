import pandas as pd
import matplotlib.pyplot as plt
import os

directory = "../data"
records = []
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        list_of_dictionaries = df.to_dict('records')
        df.to_csv(filepath)

        df['cumulative_ts'] = df['ts'].cumsum()
        df['cumulative_duplicate'] = df['duplicate'].cumsum()

        tag = filename.split('_')

        if tag[1] == 'opt': continue
        records.append({
            'func': "["+tag[0]+"]" if tag[1] !='opt' else tag[0]+"(Hashed)",
            'df': df,
        })

plt.figure(figsize=(12, 7))
for record in records:
    plt.plot(record['df']['n'], record['df']['cumulative_ts'], label=record['func'])
    # plt.plot(record['df']['n'], record['df']['cumulative_duplicate'],
    #          label=record['func'] + '_Cumulative Duplicate Count')

plt.xlabel('Number of Systems')
plt.ylabel('Cumulative Execution Time (seconds)')
plt.title('Execution Time vs. Number of Systems')
plt.grid(True)
plt.legend()
plt.show()

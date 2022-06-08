from socket import if_nameindex
import pandas as pd
from pathlib import Path


# Adds target column
def date2seconds(date: str):
    total_seconds = 0
    days, _, time = date.split()
    days = int(days)
    total_seconds += 60 * 60 * 24 * days
    hours, minutes, seconds = time.split(':')
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    total_seconds += hours * 60 * 60
    total_seconds += minutes * 60
    total_seconds += seconds
    return total_seconds


def main(csv_in: Path, csv_out: Path):

    csv_folder = Path('confident_learning', 'csv')
    df = pd.read_csv(csv_folder / csv_in, index_col=False)
    # print(df)

    df.drop(labels='auction_duration', axis=1, inplace=True)

    # Removes columns by conditions
    for index, row in df.iterrows():
        if row['place_1'] >= row['place_2']:
            df.drop(index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    target_colunms = []
    for index, row in df.iterrows():
        d1 = date2seconds(date=row['bid_to_end_date_1'])
        d2 = date2seconds(date=row['bid_to_end_date_2'])

        if d1 >= d2:
            target = 'no'
        else:
            target = 'maybe'
        target_colunms.append(target)
        
    df.insert(loc=6, column='target', value=target_colunms)
    
    df['bid_to_end_date_1'] = df['bid_to_end_date_1'].apply(lambda date: date2seconds(date=date))
    df['bid_to_end_date_2'] = df['bid_to_end_date_2'].apply(lambda date: date2seconds(date=date))
    # df['target'] = df['target'].astype('object')
    # df['Auction_duration'] = df['Auction_duration'].astype('float')
    

    df.to_csv(csv_folder / csv_out, index=False)
    print(df)


if __name__ == '__main__':

    main(csv_in='new_all_in_2.csv',
         csv_out='promejutok_2.csv')

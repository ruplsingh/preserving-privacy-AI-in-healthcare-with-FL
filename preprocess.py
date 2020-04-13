import pandas as pd

age_ranges = {
    '10-14': 1,
    '15-29': 2,
    '20-24': 3,
    '25-29': 4,
    '30-34': 5,
    '35-39': 6,
    '40-44': 7,
    '45-49': 8,
    '50-54': 9,
    '55-59': 10,
    '60-64': 11,
    '65-69': 12,
    '70-74': 13,
    '75-79': 14,
    '80-84': 15,
    '85-99': 16,
}


def create_data_set_for(is_patient, file_name, gender, age):
    df = pd.read_csv("data/" + is_patient + "/" + file_name + ".csv")
    results = []
    for key, value in df['date'].value_counts().items():
        current_day = df.copy()
        current_day.where(current_day['date'] == key, inplace=True)
        current_day = current_day.dropna()

        data_sum = current_day['activity'].sum()
        data_mean = current_day['activity'].mean()
        data_std = current_day['activity'].std()

        zero_count = current_day['activity'].value_counts()[0] if 0.0 in current_day[
            'activity'].value_counts() else 0.0
        zero_percentage = (zero_count / value) * 100

        if zero_percentage == 100.0:
            continue

        results.append(
            [data_mean, data_std, data_sum, zero_percentage, gender, age_ranges[age],
             1 if is_patient == "condition" else 0])

    return results


def pre_process_score():
    scores = pd.read_csv('data/scores.csv')
    all_entries = []
    for index, row in scores.iterrows():
        file_name = row[0]
        is_patient = "condition" if "condition" in row[0] else "control"
        gender = row[2]
        age = row[3]

        for got_row in create_data_set_for(is_patient, file_name, gender, age):
            all_entries.append(got_row)

    csv_pd = pd.DataFrame(all_entries,
                          columns=['Mean', 'STD', 'Sum', 'Null%', 'Gender', 'Age', 'Label'])
    csv_pd.to_csv('to_train_data_new.csv', index=False)


def main():
    pre_process_score()


if __name__ == "__main__":
    main()

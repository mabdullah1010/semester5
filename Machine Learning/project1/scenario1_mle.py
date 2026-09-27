import ast


files = ["dataset_20rolls.txt", "dataset_200rolls.txt", "dataset_2000rolls.txt"]


def read_data(lst):

    data = {}

    for item in lst:

        with open(item, "r", encoding="utf-8") as file:
            content_str = file.read()
            content = ast.literal_eval(content_str)

        data[item] = content

    return data


rolls_data = read_data(files)


for key,value in rolls_data.items():
    n = len(value)
    print()
    print(f"MLE for dataset: {key}")
    for state in range(1, 7):
        k = value.count(state)
        mle_probability = k / n
        print(f"Probability of rolling a {state}: {mle_probability:.2f}")
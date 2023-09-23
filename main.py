import pandas as pd
import optiver2023

env = optiver2023.make_env()
iter_test = env.iter_test()

df = pd.read_csv('train.csv')
const_pred = df.target.dropna().mean()
print(f'Constant pred: {const_pred}')

counter = 0
for (test, revealed_targets, sample_prediction) in iter_test:
    if counter == 0:
        print(test.head(3))
        print(revealed_targets.head(3))
        print(sample_prediction.head(3))
    sample_prediction['target'] = const_pred
    env.predict(sample_prediction)
    counter += 1

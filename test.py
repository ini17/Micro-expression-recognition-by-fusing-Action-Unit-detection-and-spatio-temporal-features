import pandas as pd


files = pd.read_csv(r"B:\0_0NewLife\0_Papers\SMC\Best_accuracy.csv",
                    dtype={
                        "LOSO Round": float,
                        "Best Accuracy": float,
                        "Best F1": float
                    })
temp = files.loc[(files['Dataset'] == "CASME") & (files['LOSO Round'] == 3), "Best Accuracy"]
# if temp.item() < 0:
#     print(temp.values)
# print(temp.item())
# print(type(temp.values))
files.loc[(files['Dataset'] == "CASME") & (files['LOSO Round'] == 3), "Best Accuracy"] = 2.0
print(temp)

files.to_csv(r"B:\0_0NewLife\0_Papers\SMC\Best_accuracy.csv", index=False)

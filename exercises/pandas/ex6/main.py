import pandas as pd

df = pd.DataFrame({
    "name": ["Ali", "Sara", "Umer", "Hassan"],
    "score": [85, 95, 75, 88]
})

# TODO:
# 1. Sort by 'score' descending
# 2. Add a new column 'rank' showing position after sorting


desending = df.sort_values('score', ascending=False)
print(desending)

desending['rank'] = range(1, len(desending) + 1)
print(desending)
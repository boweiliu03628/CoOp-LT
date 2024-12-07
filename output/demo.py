import pandas as pd

# 1. 读取 Excel 文件
df = pd.read_excel('demo.xlsx')

# 2. 在这里进行修改数据的操作，比如修改某一列的值
df['new_column'] = [1, 2, 3, 4]  # 假设这是你要添加的新列

# 3. 保存修改后的 DataFrame 到原文件（覆盖原文件）
df.to_excel('demo.xlsx', index=False)  # 直接保存，不保存行索引

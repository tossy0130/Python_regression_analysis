import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

# 小数点第 3位まで出力
%precision 3 
%matplotlib inline

# !git clone https://github.com/umacchi/python-regression-tutorial-data datasets

# pandas の read_csv()関数で、ふぁるを読み込み
df = pd.read_csv("/content/datasets/test.csv")
n = len(df)
print('test.csvの長さ:::' + str(n))
# test.csv の最初の 5行を出力する
df.head()

# 説明変数(x) = 月曜の売上(monday_sales)
x = np.array(df['monday_sales'])

# 目的変数(y) = 週間売上(week_sales)
y = np.array(df['week_sales'])

# 説明変数 の数
p = 1

# 単回帰式　作成
poly_fit = np.polyfit(x, y, 1)
# array([ 6.501, 23.794])

poly_1d = np.poly1d(poly_fit)
# poly1d 多項式の生成ができるメソッド

# x = 月曜の売上(monday_sales)
x_min = x.min()
x_max = x.max()

# linespace 等差数列を作成するためのメソッド
# numは省いて初期値である50を使用することで、50個の値を持つ等差数列を作成
# np.linspace(start, stop, [num]) num 初期値 = 50
xs = np.linspace(x_min, x_max)
ys = poly_1d(xs)

#========= 散布図と回帰直線を描画 ==========

# 引数の figsize は、 figsize=(横幅の大きさ, 縦幅の大きさ) をセットすることでキャンバスが表示される大きさを指定で
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, xlabel="monday_sales", ylabel="week_sales")
ax.plot(xs, ys, color="gray")

ax.scatter(x, y)


# *************************** statsmodelsによる回帰分析 *********************

formula = 'week_sales ~ monday_sales'
result = smf.ols(formula, df).fit()
result.summary()



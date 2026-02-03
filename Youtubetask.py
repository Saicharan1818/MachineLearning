#goal :If the  thumbnail gets a 8% CTR, how many views can I expect?"
#step 1 : import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# step 2: create the Dataset
data = {
    'ctr': [3.2, 5.8, 2.1, 7.4, 4.5, 6.2, 1.8, 5.1, 3.9, 8.5,
            4.2, 2.8, 6.8, 3.5, 7.1, 2.4, 5.5, 4.8, 6.5, 3.1,
            7.8, 2.6, 5.3, 4.1, 6.9, 3.7, 8.1, 2.2, 5.9, 4.6],

    'total_views': [12000, 28000, 8500, 42000, 19000, 33000, 7000, 24000, 16000, 51000,
                    18000, 11000, 38000, 14500, 44000, 9500, 26000, 21000, 35000, 13000,
                    47000, 10000, 25000, 17500, 40000, 15000, 49000, 8000, 31000, 20000]
}


df = pd.DataFrame(data)

#
#step 3: Explore the Data
print("First 5 rows of data:")
print(df.head())
print("\nBasic statistics:")#to get basic statistics like mean, std, min, max
print(df.describe()) 

#load the data
x = df[['ctr']]
y = df['total_views']

model = LinearRegression()
model.fit(x, y)

new_ctr = np.array([[8]])
predict_views = model.predict(new_ctr)


print(f"if CTR is {new_ctr} , no of views we expect is")
print(f"the expected view is {predict_views}")




#vizualize b/w ctr and total views
plt.figure(figsize=(10,6))
plt.scatter(df['ctr'],df['total_views'],c=df['ctr'], s=100, cmap = 'plasma', alpha=0.3, linewidths= 1.5, label = 'Actual data')

X_line = np.linspace(df['ctr'].min(), df['ctr'].max(), 100).reshape(-1, 1)

plt.plot(df['ctr'] , model.predict(x), color = 'blue' ,label = 'Regressionline')
plt.scatter(df['ctr'], predict_views[0], c = 'red', alpha=0.3, linewidths=1.5, label = 'Predicted')

plt.colorbar(label= 'ctr intensity')
plt.xlabel("CTR (%)", fontsize = 12, fontweight = 'bold')
plt.ylabel("Total_views",fontsize = 12)
plt.title('CTR VS TOTAL VIEWS')
plt.legend()
plt.grid(alpha = 0.3)
plt.show()






































 











import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'distance_km': [2.5, 6.0, 1.2, 8.5, 3.8, 5.2, 1.8, 7.0, 4.5, 9.2,
                    2.0, 6.5, 3.2, 7.8, 4.0, 5.8, 1.5, 8.0, 3.5, 6.8,
                    2.2, 5.5, 4.2, 9.0, 2.8, 7.2, 3.0, 6.2, 4.8, 8.2],

    'prep_time_min': [10, 20, 8, 25, 12, 18, 7, 22, 15, 28,
                      9, 19, 11, 24, 14, 17, 6, 26, 13, 21,
                      10, 16, 14, 27, 11, 23, 12, 18, 15, 25],

    'delivery_time_min': [18, 38, 12, 52, 24, 34, 14, 45, 29, 58,
                          15, 40, 21, 50, 27, 35, 11, 54, 23, 43,
                          17, 32, 26, 56, 19, 47, 20, 37, 30, 53]
}

df = pd.DataFrame(data)

#Explore the data
print("First 5 rows of the data")
print(df.head())
print("Bastic Statistics")
print(df.describe())

#visualize the data
plt.style.use("seaborn-v0_8-whitegrid")

fig , axes = plt.subplots(1, 2, figsize = (14, 4), dpi =150)

scatter1 = axes[0].scatter(df['distance_km'], df['delivery_time_min'], c = df['delivery_time_min'], cmap= 'viridis', s = 100, alpha = 0.7, edgecolors= 'white' , linewidth = 1.5 )
axes[0].set_xlabel('distanece_km' , fontsize = 12, fontweight = 'bold')
axes[0].set_ylabel('delivery_time_min', fontsize = 12, fontweight = 'bold')
axes[0].set_title('Distance_km VS Delivery_time_min', fontsize = 16, fontweight = 'bold')

plt.colorbar(scatter1, ax= axes[0], label = 'delivery time  min')

scatter2 = axes[1].scatter(df['prep_time_min'], df['delivery_time_min'], c =  df['delivery_time_min'], cmap = 'plasma', s=100, alpha = 0.7, edgecolor= 'white',linewidth = 1.5 )
axes[1].set_xlabel('prep_time_min', fontsize=12, fontweight = 'bold')
axes[1].set_ylabel('delivery_time_min', fontsize = 12, fontweight = 'bold')
axes[1].set_title('Prep Time Min Vs Delivery Time Min', fontsize = 12, fontweight = 'bold')

plt.colorbar(scatter2, ax= axes[1], label = 'delivery time  min')

plt.savefig("Food Delivery time predictions.png",dpi = 150, bbox_inches = 'tight')


plt.tight_layout()
plt.show()

#preparing a model

X = df[['distance_km', 'prep_time_min']]
y= df['delivery_time_min']


#split the train

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


#create a model and train

model = LinearRegression()
model.fit(X_train, y_train)

#pred with actual test values
y_pred = model.predict(X_test)

print("Actual vs predicted values")
final_answer  = pd.DataFrame({'Actual': y_test.values, 'predicted ': y_pred})
print(final_answer)

#model parameters checking

print(f"Coefficients: {model.coef_[0]:.2f}, {model.coef_[1]}")
print(f"Intercepts: {model.intercept_[0]:.0f}")

#answering Vikrams question

new_order = [[7, 15]] # 7km, 15min
predicted_value = model.predict(new_order)

print("Completed prediction")
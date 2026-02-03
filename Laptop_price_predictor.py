import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
data = {
    'ram_gb': [4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16],

    'storage_gb': [256, 512, 128, 512, 256, 512, 256, 1024, 256, 512,
                   128, 512, 256, 1024, 256, 512, 128, 512, 256, 1024,
                   256, 512, 128, 512, 256, 512, 256, 1024, 256, 512],

    'processor_ghz': [2.1, 2.8, 1.8, 3.2, 2.4, 3.0, 2.0, 3.5, 2.6, 3.0,
                      1.6, 2.8, 2.2, 3.4, 2.5, 2.9, 1.9, 3.1, 2.3, 3.6,
                      2.0, 2.7, 1.7, 3.3, 2.4, 3.0, 2.1, 3.5, 2.6, 3.2],

    'price_inr': [28000, 45000, 22000, 72000, 38000, 52000, 26000, 95000, 42000, 68000,
                  20000, 48000, 29000, 88000, 40000, 50000, 23000, 70000, 36000, 98000,
                  25000, 46000, 21000, 75000, 39000, 53000, 27000, 92000, 43000, 73000]
}

df = pd.DataFrame(data)

#explore the data
print("first five rows of the data")
print(df.head())

print("Basic statistics")
print(df.describe())

#preparing the model

X = df[['ram_gb','storage_gb', 'processor_ghz']]
y = df['price_inr']

#train the model
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

#predict the model for y_test values

y_pred = model.predict(X_test)

print({'Actual values':y_test.values, 'predicted':y_pred})

#check model parameters
print(f'coeffients:{model.coef_[0]:.2f},{model.coef_[1]:.2f},{model.coef_[2]:.2f}')
print(f"Intercepts:{model.intercept_:.0f}")


#calculate r2score
from sklearn.metrics import r2_score

score = r2_score(y_test, y_pred)
print(f"The accuracy of the model is :{score:.2f}")

new_features =pd.DataFrame([[16, 512, 3.2]],columns=['ram_gb', 'storage_gb','processor_ghz'])
predicted_price = model.predict(new_features)
print(f"The laptop price predicted price is: {predicted_price[0]}")


meera_laptop = pd.DataFrame([[8, 512, 2.8]],columns=['ram_gb','storage_gb', 'processor_ghz'])
laptop_predicted_price = model.predict(meera_laptop)
print(f"The meeras laptop predicted price is: {laptop_predicted_price}")

actual_price = 55000

if laptop_predicted_price > actual_price:
    print("It is Overpriced")
else:
    print("Not overpriced")

#visualize 3 scatter plots between ram gb vs price inr, storage_gb vs price_inr, processor_ghz vs price_inr
plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(1, 3, figsize = (18, 5))

plt.subplot(1, 3, 1)
scatter1 = axes[0].scatter(df['ram_gb'],df['price_inr'], c = df['price_inr'], cmap = 'viridis', s=100, alpha = 0.7, edgecolors = 'white', linewidths = 1.5)
axes[0].set_xlabel('RAM GB',fontsize = 12, fontweight = 'bold')
axes[0].set_ylabel('PRICE INR',fontsize = 12, fontweight = 'bold')
axes[0].set_title('RAM GB Vs PRICE INR',fontsize = 16, fontweight = 'bold')
plt.colorbar(scatter1, ax=axes[0], label = 'price intensity' )

plt.subplot(1, 3, 2)

scatter2 = axes[1].scatter(df['storage_gb'],df['price_inr'], c=df['price_inr'], cmap= 'plasma', s=100, alpha = 0.7,edgecolors = 'white', linewidths = 1.5)
axes[1].set_xlabel('STORAGE GB',fontsize = 12, fontweight= 'bold')
axes[1].set_ylabel('PRICE INR', fontsize = 12, fontweight = 'bold')
axes[1].set_title('STORAGE VS PRICE', fontsize = 16, fontweight = 'bold')
plt.colorbar(scatter2, ax=axes[1], label = 'price intensity' )

plt.subplot(1, 3, 3)

scatter3 = axes[2].scatter(df['processor_ghz'],df['price_inr'], c=df['price_inr'], cmap = 'viridis', s=100, alpha = 0.7, edgecolors = 'white', linewidths = 1.5)
axes[2].set_xlabel('PROCESSOR_GHZ', fontsize = 12, fontweight = 'bold')
axes[2].set_ylabel('PRICE INR', fontsize = 12, fontweight = 'bold')
axes[2].set_title('PROCESSOR VS PRICE INR', fontweight = 'bold')
plt.colorbar(scatter3, ax=axes[2], label = 'price intensity' )

plt.suptitle('Laptop price prediction Analysis', fontsize = 16)

plt.savefig('Laptop price predictor.png',dpi = 150, bbox_inches = 'tight')

plt.tight_layout()
plt.show()




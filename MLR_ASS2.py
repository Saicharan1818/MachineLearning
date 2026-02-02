#step1 : Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Create the Dataset
data = {
    'pages': [5, 12, 3, 8, 15, 4, 10, 6, 20, 7,
              9, 2, 14, 5, 11, 3, 8, 18, 6, 13,
              4, 16, 7, 10, 2, 12, 5, 9, 15, 6],

    'deadline_days': [14, 7, 21, 5, 10, 18, 6, 12, 8, 15,
                      4, 25, 9, 11, 7, 20, 6, 5, 14, 8,
                      16, 6, 10, 4, 22, 7, 13, 5, 9, 11],

    'rate_inr': [8000, 22000, 5000, 18000, 28000, 6500, 19000, 10000, 35000, 11000,
                 20000, 3500, 25000, 9000, 21000, 5500, 17000, 38000, 9500, 24000,
                 7000, 32000, 13000, 23000, 4000, 22500, 8500, 19500, 29000, 11500]
}

df = pd.DataFrame(data)

#Step 3 :Explore the data
print("First five rows of data")
print(df.head())
print("\n Basic statistics")
print(df.describe())

#step 4: vizualize the data

plt.style.use('seaborn-v0_8-whitegrid') #to look good

fig, axes = plt.subplots(1, 2, figsize = (14, 6))

scatter1 = axes[0].scatter(df['pages'],df['rate_inr'], c= df['rate_inr'], cmap='viridis', s=100, alpha = 0.7, edgecolors = 'white', linewidth = 1.5)
axes[0].set_xlabel('Number of pages', fontsize = 12, fontweight = 'bold')
axes[0].set_ylabel('Rate (INR)',fontsize = 12, fontweight = 'bold')
axes[0].set_title('Pages vs rate', fontsize=14, fontweight= 'bold',pad=15)

plt.colorbar(scatter1, ax=axes[0], label = 'Rate Intensity')

scatter2 = axes[1].scatter(df['deadline_days'], df['rate_inr'], c=df['rate_inr'], cmap='plasma', s=100, alpha = 0.7, edgecolors = 'white', linewidth=1.5)
axes[1].set_xlabel('Deadline_days', fontsize = 12, fontweight='bold')
axes[1].set_ylabel('Rate (INR)', fontsize= 12, fontweight = 'bold')
axes[1].set_title('Deadline vs rate', fontsize = 15, fontweight = 'bold')

plt.colorbar(scatter2, ax=axes[1], label= 'Rate Intensity')

plt.tight_layout()
plt.show()


#step5 : Prepare data for training
X = df[['pages','deadline_days']]
y = df['rate_inr']

#step6: SPlit into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 42)


#step7: Create and train model

model = LinearRegression()

model.fit(X_train, y_train)


#step 8: prediction on test data
y_pred = model.predict(X_test)

print("\n Predictions vs actual")
results = pd.DataFrame({'Actual': y_test.values, 'predicted': y_pred.round()})
print(results)

#step 9 :Check model parameters

print("\n Model Parameters:")
print(f"Coefficients: Pages = {model.coef_[0]:.2f}, Deadline = {model.coef_[1]:.2f}")
print(f"Intercepts: {model.intercept_:.2f}")


#step 10:Model Accuracy
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print(f"Model Accuracy (R2 score):{score}.2f")

#step 11 : Answer priyas question

new_project = [[10, 5]]
predicted_rate = model.predict(new_project)
print(f"priya's question: 10 pages website, 5 day deadline")
print(f"Recommend rate: {predicted_rate[0]:.2f}")

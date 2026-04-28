import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set random seed for reproducibility 
np.random.seed(13160240)

# Define correct column names
num_cols = ['Average Rating', 'Average Difficulty', 'Number of Ratings', 
            'Received a Pepper', 'Proportion Would Take Again', 
            'Number of Online Ratings', 'Male', 'Female']

qual_cols = ['Major/Field', 'University', 'State']

# Load numerical data
num_df = pd.read_csv('rmpCapstoneNum.csv', header=None, names=num_cols)

# Load qualitative data
qual_df = pd.read_csv('rmpCapstoneQual.csv', header=None, names=qual_cols)

# Merge the two datasets 
df = pd.concat([num_df, qual_df], axis=1)

# Drop entries where 'Number of Ratings' is missing
df = df.dropna(subset=['Number of Ratings'])

# Keep only professors with at least 5 ratings
df = df[df['Number of Ratings'] >= 5]



#Q1
print('Q1 (Mann-Whitney U):')

df_q1 = df[(df['Male'] + df['Female']) > 0]

male_ratings = df_q1[df_q1['Male'] == 1]['Average Rating']
female_ratings = df_q1[df_q1['Female'] == 1]['Average Rating']

median_male = male_ratings.median()
median_female = female_ratings.median()

from scipy.stats import mannwhitneyu
u_stat_q1, p_val_q1 = mannwhitneyu(male_ratings, female_ratings, alternative='two-sided')

print(f"Median (Male): {median_male:.3f}")
print(f"Median (Female): {median_female:.3f}")
print(f"U-statistic: {u_stat_q1:.1f}")
print(f"P-value: {p_val_q1:.5f}")

gender_df = pd.DataFrame({
    'Average Rating': pd.concat([male_ratings, female_ratings]),
    'Gender': ['Male'] * len(male_ratings) + ['Female'] * len(female_ratings)
})

plt.figure(figsize=(8,6))
sns.boxplot(x='Gender', y='Average Rating', data=gender_df)
plt.title('Average Rating by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()



#Q2
corr_coeff, p_val = stats.pearsonr(df['Number of Ratings'], df['Average Rating'])
print('Q2:')
print(f"Correlation coefficient (r): {corr_coeff:.3f}")
print(f"P-value: {p_val:.5f}")

plt.figure(figsize=(8,6))
sns.regplot(x='Number of Ratings', y='Average Rating', data=df, scatter_kws={'alpha':0.3})
plt.title('Experience (Number of Ratings) vs Teaching Quality (Average Rating)')
plt.xlabel('Number of Ratings (Experience Proxy)')
plt.ylabel('Average Rating (Teaching Quality)')
plt.grid(True)
plt.show()



#Q3
corr_coeff, p_val = stats.pearsonr(df['Average Difficulty'], df['Average Rating'])
print('Q3:')
print(f"Correlation coefficient (r): {corr_coeff:.3f}")
print(f"P-value: {p_val:.5f}")

plt.figure(figsize=(8,6))
sns.regplot(x='Average Difficulty', y='Average Rating', data=df, scatter_kws={'alpha':0.3})
plt.title('Average Difficulty vs Average Rating')
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()



#Q4
print('Q4 (Mann-Whitney U):')

df['online_ratio'] = df['Number of Online Ratings'] / df['Number of Ratings']
df_q4 = df.dropna(subset=['online_ratio', 'Average Rating'])

df_q4['Online Group'] = np.where(df_q4['online_ratio'] < 0.5, 'Low Online (<0.5)', 'High Online (≥0.5)')

low = df_q4[df_q4['Online Group'] == 'Low Online (<0.5)']['Average Rating']
high = df_q4[df_q4['Online Group'] == 'High Online (≥0.5)']['Average Rating']

median_low = low.median()
median_high = high.median()


from scipy.stats import mannwhitneyu
u_stat, p_val = mannwhitneyu(low, high, alternative='two-sided')

print(f"Median (Low): {median_low:.2f}")
print(f"Median (High): {median_high:.2f}")
print(f"U-statistic: {u_stat:.1f}")
print(f"P-value: {p_val:.5e}")

plt.figure(figsize=(8,6))
sns.boxplot(x='Online Group', y='Average Rating', data=df_q4)
plt.title('Average Rating by Online Teaching Group (Two Groups)')
plt.xlabel('Online Teaching Group')
plt.ylabel('Average Rating')
plt.grid(True)
plt.tight_layout()
plt.show()



#Q5
subset_df = df[['Proportion Would Take Again', 'Average Rating']].dropna()

corr_coeff_q5, p_val_q5 = stats.pearsonr(subset_df['Proportion Would Take Again'], subset_df['Average Rating'])
print('Q5:')
print(f"Correlation coefficient (r): {corr_coeff_q5:.3f}")
print(f"P-value: {p_val_q5:.5f}")

plt.figure(figsize=(8,6))
sns.regplot(x='Proportion Would Take Again', y='Average Rating', data=df, scatter_kws={'alpha':0.3})
plt.title('Proportion Would Take Again vs Average Rating')
plt.xlabel('Proportion Would Take Again')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()



#Q6
print('Q6 (Mann-Whitney U):')

hot_ratings = df[df['Received a Pepper'] == 1]['Average Rating']
not_hot_ratings = df[df['Received a Pepper'] == 0]['Average Rating']

median_hot = hot_ratings.median()
median_nothot = not_hot_ratings.median()

u_stat_q6, p_val_q6 = mannwhitneyu(hot_ratings, not_hot_ratings, alternative='two-sided')
print(f"Median (Hot): {median_hot:.3f}")
print(f"Median (Not hot): {median_nothot:.3f}")
print(f"U-statistic: {u_stat_q6:.1f}")
print(f"P-value: {p_val_q6:.5f}")

df_pepper = df[df['Received a Pepper'].isin([0,1])].copy()
df_pepper['Hotness'] = np.where(df_pepper['Received a Pepper'] == 1, 'Hot', 'Not Hot')

plt.figure(figsize=(8,6))
sns.boxplot(x='Hotness', y='Average Rating', data=df_pepper)
plt.title('Average Rating by Hotness')
plt.xlabel('Hotness')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()



#Q7
subset_df = df[['Average Difficulty', 'Average Rating']].dropna()

X = sm.add_constant(subset_df['Average Difficulty'])
Y = subset_df['Average Rating']
model = sm.OLS(Y, X).fit()
r_squared = model.rsquared
y_pred = model.predict(X)
rmse = np.sqrt(np.mean((Y - y_pred) ** 2))
print('Q7:')
print(f"R-squared: {r_squared:.4f}")
print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(8,6))
sns.regplot(x='Average Difficulty', y='Average Rating', data=subset_df, scatter_kws={'alpha':0.3})
plt.title('Average Rating vs Average Difficulty')
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()



#Q8
feature_cols = ['Average Difficulty', 'Number of Ratings', 'Received a Pepper',
                'Proportion Would Take Again', 'Number of Online Ratings', 'Male', 'Female']

subset_df = df[['Average Rating'] + feature_cols].dropna()
X = subset_df[feature_cols]
X = sm.add_constant(X)
Y = subset_df['Average Rating']
model = sm.OLS(Y, X).fit()
r_squared = model.rsquared
y_pred = model.predict(X)
rmse = np.sqrt(np.mean((Y - y_pred) ** 2))
print('Q8:')
print(f"R-squared: {r_squared:.4f}")
print(f"RMSE: {rmse:.4f}")

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
y_true = Y
y_pred = model.predict(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred, y=y_true, alpha=0.3)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--')  # 理想参考线
plt.xlabel('Predicted Average Rating')
plt.ylabel('Actual Average Rating')
plt.title('Predicted vs Actual Average Rating')
plt.grid(True)
plt.show()



#Q9
subset_df = df[['Average Rating', 'Received a Pepper']].dropna()
X = subset_df[['Average Rating']] 
y = subset_df['Received a Pepper']
clf = LogisticRegression()
clf.fit(X, y)
y_pred_prob = clf.predict_proba(X)[:,1]
y_pred_class = clf.predict(X)
accuracy = accuracy_score(y, y_pred_class)
auc = roc_auc_score(y, y_pred_prob)
print('Q9:')
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")

fpr, tpr, thresholds = roc_curve(y, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()



#Q10
feature_cols = ['Average Rating', 'Average Difficulty', 'Number of Ratings', 
                'Proportion Would Take Again', 'Number of Online Ratings', 'Male', 'Female']

subset_df = df[['Received a Pepper'] + feature_cols].dropna()

X = subset_df[feature_cols]
y = subset_df['Received a Pepper']

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

y_pred_prob = clf.predict_proba(X)[:,1]
y_pred_class = clf.predict(X)

accuracy = accuracy_score(y, y_pred_class)
auc = roc_auc_score(y, y_pred_prob)
print('Q10:')
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")

fpr, tpr, thresholds = roc_curve(y, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Full Model Predicting Hot Pepper Status')
plt.legend()
plt.grid(True)
plt.show()



#Question for Extra Credit
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
print('Question for Extra Credit:')
df_state = df[['State', 'Received a Pepper']].dropna()

state_pepper_rate = df_state.groupby('State')['Received a Pepper'].mean().sort_values(ascending=False)
top_states = state_pepper_rate.head(10).round(3)  

plt.figure(figsize=(10,6))
ax = sns.barplot(x=top_states.values, y=top_states.index, palette='viridis')

for i, (value, label) in enumerate(zip(top_states.values, top_states.index)):
    ax.text(value + 0.01, i, f"{value:.3f}", va='center', fontsize=10)

plt.xlabel('Proportion of Professors with Hot Pepper')
plt.title('Top 10 States by Hot Pepper Proportion')
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()
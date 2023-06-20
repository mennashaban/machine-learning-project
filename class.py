import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# ======================================================================#

# Feature Encoding Function
def Feature_Encoding(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


# ========================================================================#

# Read mega store data
data = pd.read_csv('megastore-classification-dataset.csv')
# apply encoding before test split
# Apply train test split

X = data.drop('ReturnCategory', axis=1)
Y = data['ReturnCategory']

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
test_data.to_csv('test_data.csv', index=False)
#print(pd.DataFrame(train_data.mean().T).to_dict())
# ======================================================================#

# preprocesssing

# unwanted features
train_data.drop('Customer Name', axis=1, inplace=True)
train_data.drop('Country', axis=1, inplace=True)

# encoding categorical data
# handle category tree --> split it into two new columns [main category - sub category]

train_data[['MainCategory', 'SubCategory']] = train_data['CategoryTree'].apply(lambda x: pd.Series(eval(x)))
train_data.drop('CategoryTree', axis=1, inplace=True)

# split order & ship date

train_data[['order mounth', 'order day', 'order year']] = train_data['Order Date'].str.split('/', expand=True)
train_data[['ship mounth', 'ship day', 'ship year']] = train_data['Ship Date'].str.split('/', expand=True)

# Extract a new feature from date columns --> TimeDuration = ShipDate - OrderDate

train_data["Order Date"] = pd.to_datetime(train_data["Order Date"])
train_data["Ship Date"] = pd.to_datetime(train_data["Ship Date"])
train_data["Time Duration"] = (train_data["Ship Date"] - train_data["Order Date"]).dt.days.astype(int)

train_data["order mounth"] = pd.to_datetime(train_data["Order Date"])
train_data["order day"] = pd.to_datetime(train_data["Order Date"])
train_data["order year"] = pd.to_datetime(train_data["Order Date"])

train_data["ship mounth"] = pd.to_datetime(train_data["Ship Date"])
train_data["ship day"] = pd.to_datetime(train_data["Ship Date"])
train_data["ship year"] = pd.to_datetime(train_data["Ship Date"])

train_data.drop('Order Date', axis=1, inplace=True)
train_data.drop('Ship Date', axis=1, inplace=True)
#########################################################
# Feature selection by chi-squared before encoding categorical input and categorical output

not_encode_X_train_dis = train_data[
    ['City', 'State', 'MainCategory', 'SubCategory', 'Region', 'Ship Mode', 'Segment', 'Order ID', 'Customer ID',
     'Product ID', 'Product Name', 'Postal Code']]
# Create contingency table

chi2_scores = {}
for i in not_encode_X_train_dis:
    contingency_table = pd.crosstab(not_encode_X_train_dis[i], train_data['ReturnCategory'])

    # Print the contingency table
    # print("continegency table of", i, contingency_table)

    # Calculate the chi-squared statistic, p-value, and degrees of freedom
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    chi2_scores[i] = p

    # Print the results
    print("Chi-squared statistic of", i, chi2)
    print("p-value of ", i, p)
    print("Degrees of freedom of", i, dof)
# Select the top K features with the smallest p-values
k = 4
selected_features = sorted(chi2_scores, key=chi2_scores.get)[:k]

print("Selected features:", selected_features)

# print("before",train_data.dtypes)
# encoding
cols = ('State', 'City', 'MainCategory', 'SubCategory', 'Region', 'Ship Mode', 'Segment', 'Order ID', 'Customer ID',
        'Product ID', 'Product Name', 'ReturnCategory')

with open('features for encoding.pkl', 'wb') as f:
    pickle.dump(cols, f)

#train_data = Feature_Encoding(train_data, cols)

for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(train_data[c].values))
    train_data[c] = lbl.transform(list(train_data[c].values))

with open('Feature Encoding.pkl', 'wb') as f:
    pickle.dump(lbl, f)
# encode ReturnCategory (target)
# print("after",train_data.dtypes)
# medium profit =4   low profit=3   low loss=2   high profit = 1   high loss = 0

# Features Scaling
# 1 check if it needed scaling
# Check the range of the feature
'''feature_range = train_data['Discount'].max() - train_data['Discount'].min()
print(feature_range)'''
scaler = MinMaxScaler()
train_data[['Order ID']] = scaler.fit_transform(train_data[['Order ID']])  ## range = 3905
train_data[['Product ID']] = scaler.fit_transform(train_data[['Product ID']])  ## range = 1786
train_data[['Product Name']] = scaler.fit_transform(train_data[['Product Name']])  ## range = 1772
train_data[['Sales']] = scaler.fit_transform(train_data[['Sales']])  ## range = 13999
# print(train_data['Order ID'])
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Feature selection
# Calculate Kendall's correlation coefficient for numerical input and categorical output


X_train_con = train_data[['Sales', 'Quantity', 'Discount', 'Time Duration']]

corr, pval = kendalltau(X_train_con['Sales'], train_data['ReturnCategory'])
print(f"Kendall correlation Sales : {corr:.3f}")
corr, pval1 = kendalltau(X_train_con['Quantity'], train_data['ReturnCategory'])
print(f"Kendall correlation Quantity : {corr:.3f}")
corr, pval2 = kendalltau(X_train_con['Discount'], train_data['ReturnCategory'])
print(f"Kendall correlation Discount : {corr:.3f}")
corr, pval3 = kendalltau(X_train_con['Time Duration'], train_data['ReturnCategory'])
print(f"Kendall correlation Time Duration : {corr:.3f}")

# highest corr => Discount = -0.343

# Handeling the outliers
cols = ['Sales', 'Quantity', 'Discount']

# Loop through each column
for col in cols:
    # Calculate the IQR
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1

    # Define the upper and lower bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers from the column
    train_data = train_data[(train_data[col] >= lower_bound) & (train_data[col] <= upper_bound)]

# Update the columns
train_data['Sales'] = pd.DataFrame(train_data['Sales'])
train_data['Quantity'] = pd.DataFrame(train_data['Quantity'])
train_data['Discount'] = pd.DataFrame(train_data['Discount'])
# Search for OUtlires
columns_of_interest = ['Sales', 'Quantity', 'Discount', 'MainCategory', 'Region', 'State']
z_scores = (train_data[columns_of_interest] - train_data[columns_of_interest].mean()) / train_data[
    columns_of_interest].std()
threshold = 3
outliers = train_data[z_scores > threshold]
print(outliers.count())

# sales ,quantity ,discount
total = train_data[['State', 'MainCategory', 'Product ID', 'Region', 'Sales', 'Quantity', 'Discount']]
with open('selected features.pkl', 'wb') as f:
    pickle.dump(total, f)

#=============================================MODELS==============================================================#
# split the training data into training and validation sets (75/25 split)
X_train_v, X_val, y_train, y_val = train_test_split(total, train_data['ReturnCategory'], test_size=0.25,random_state=42)
# -----------------------decision tree --------------------------------#

param_grid = {'max_depth': [5, 7, 9]}

# Create an instance of the DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Create a GridSearchCV object with the classifier and parameter grid
grid_search = GridSearchCV(clf, param_grid)

# Train the classifier on the training set using GridSearchCV
grid_search.fit(X_train_v, y_train)

clf_best = DecisionTreeClassifier(max_depth=grid_search.best_params_['max_depth'])

# Make predictions on the testing set using the best classifier
clf_best.fit(X_train_v, y_train)

# Make predictions on the validation set
y_val_pred = clf_best.predict(X_val)

# Evaluate the accuracy of the predictions
accuracy = accuracy_score(y_val, y_val_pred)
print("Accuracy of DecisionTree :", accuracy)

# Save the trained model to disk using pickle
with open('decision tree.pkl', 'wb') as f:
    pickle.dump(clf_best, f)

#-----------------------RandomForest--------------------------------#

param_grid = {'n_estimators': [100, 200, 300]}

# Create an instance of the RandomForestClassifier
clf = RandomForestClassifier()

# Create a GridSearchCV object with the classifier and parameter grid
grid_search = GridSearchCV(clf, param_grid)

# Train the classifier on the training set using GridSearchCV
grid_search.fit(X_train_v, y_train)
clf_best = RandomForestClassifier(n_estimators = grid_search.best_params_['n_estimators'])
# Train the classifier on the training set
clf_best.fit(X_train_v, y_train)

# Make predictions on the validation set
y_val_pred = clf_best.predict(X_val)

# Evaluate the accuracy of the predictions
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Accuracy of randon forest :", val_accuracy)

# Save the trained model to disk using pickle
with open('RandomForest.pkl', 'wb') as f:
    pickle.dump(clf_best, f)
# ----------------------svm-------------------------------#
# Create an SVM classifier with a linear kernel
clf = SVC(kernel='linear', C=100)

# Train the classifier on the scaled training set
clf.fit(X_train_v, y_train)

# Evaluate the accuracy of the classifier on the scaled validation set
val_accuracy = clf.score(X_val, y_val)
print("Validation Accuracy svc:", val_accuracy)
# Save the trained model to disk using pickle
with open('SVM.pkl', 'wb') as f:
    pickle.dump(clf, f)
# -----------------------Naive Bayes--------------------------------#

# Create a Naive Bayes classifier
clf = GaussianNB()

# Train the classifier on the training set
clf.fit(X_train_v, y_train)

# Evaluate the accuracy of the classifier on the validation set
val_accuracy = clf.score(X_val, y_val)
print("Validation Accuracy  Naive Bayes:", val_accuracy)

# Save the trained model to disk using pickle
with open('Naive Bayes.pkl', 'wb') as f:
    pickle.dump(clf, f)



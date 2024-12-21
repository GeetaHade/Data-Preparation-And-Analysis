import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings("ignore")

dataf = pd.read_csv('automobile/imports-85.data', header = None)   #file is without headers

#here we need to define the columns from .names file as data set has no columns
cols = ['symboling','normalized-losses','make','fuel-type'             
  ,'aspiration'                   
  ,'num-of-doors'                 
  ,'body-style'                   
  ,'drive-wheels'                 
  ,'engine-location'              
 ,'wheel-base'                  
 ,'length'                       
 ,'width'                        
 ,'height'                       
 ,'curb-weight'                 
 ,'engine-type'                  
 ,'num-of-cylinders'             
 ,'engine-size'                  
 ,'fuel-system'                 
 ,'bore'                         
 ,'stroke'                       
 ,'compression-ratio'            
 ,'horsepower'                   
 ,'peak-rpm'                     
 ,'city-mpg'                     
 ,'highway-mpg'                  
 ,'price']

dataf.columns = cols   #here we assign columns to dataframe
#print(df.head(3))

#shortlist the selected columns from data frame and rename them
dataf_selected = dataf[['wheel-base','compression-ratio','engine-size','length','width','city-mpg']]
dataf_selected.columns = ['X1','X2','X3','X4','X5','Y']

X = dataf_selected[['X1','X2','X3','X4','X5']]
y = dataf_selected['Y']

print(dataf_selected.head())
#print(df.columns)

#here we are spliting dataset into test set, train set , validation set
#first we only split data into training and testing as 60-40

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size = 0.40)   #test_size  means 40% is test set

#here we are splitting validation and test as 50%
X_validatn, X_test, y_validatn, y_test = train_test_split(X_tmp, y_tmp, test_size = 0.5)

# now we have train_set = 60%, test_set = 20%, validation_set = 20%

print("\n")
print(f"Training set: {len(X_train)} ")
print(f"Validation set: {len(X_validatn)}")
print(f"Test set: {len(X_test)}")


#here we initialize the linear regression model and fit it on training data
# Default parameters: fit_intercept=True, normalize=False
Linear = LinearRegression()
Linear.fit(X_train, y_train)

#here we initialize the Ridge model and fit it on training data
# Default parameters: alpha=1.0, solver='auto'
Ridge_model = Ridge()
Ridge_model.fit(X_train, y_train)

#here we initialize the Lasso model and fit it on training data
# Default parameters: alpha=1.0, max_iter=1000
Lasso_model = Lasso()
Lasso_model.fit(X_train, y_train)

#here we use .predict() method to make predictions on validation and test sets to get predicted values

#predictions on validation set
y_validatn_predct_Linear =     Linear.predict(X_validatn)
y_validatn_predct_Ridge =     Ridge_model.predict(X_validatn)
y_validatn_predct_Lasso =     Lasso_model.predict(X_validatn)

#predictions on test set
y_test_predct_Linear =     Linear.predict(X_test)
y_test_predct_Ridge =     Ridge_model.predict(X_test)
y_test_predct_Lasso =     Lasso_model.predict(X_test)

#MSE : it calculates squares of differences between the actual values and values predicted by model
# Lower MSE value is good for model

# MSE for the validation set
MSE_validatn_Linear = mean_squared_error(y_validatn, y_validatn_predct_Linear)
MSE_validatn_Ridge = mean_squared_error(y_validatn, y_validatn_predct_Ridge)
MSE_validatn_Lasso = mean_squared_error(y_validatn, y_validatn_predct_Lasso)


# MSE for the test set
MSE_test_Linear = mean_squared_error(y_test, y_test_predct_Linear)
MSE_test_Ridge = mean_squared_error(y_test, y_test_predct_Ridge)
MSE_test_Lasso = mean_squared_error(y_test, y_test_predct_Lasso)


#PCC measure linear corelation rangs from -1 to 1, 1 means perfect correlation

# PCC for the validation set
PCC_validatn_Linear = np.corrcoef(y_validatn, y_validatn_predct_Linear)[0, 1]
PCC_validatn_Ridge = np.corrcoef(y_validatn, y_validatn_predct_Ridge)[0, 1]
PCC_validatn_Lasso = np.corrcoef(y_validatn, y_validatn_predct_Lasso)[0, 1]

# PCC for the test set
PCC_test_Linear = np.corrcoef(y_test, y_test_predct_Linear)[0, 1]
PCC_test_Ridge = np.corrcoef(y_test, y_test_predct_Ridge)[0, 1]
PCC_test_Lasso = np.corrcoef(y_test, y_test_predct_Lasso)[0, 1]

#r2 will tells us how accurate are the predicted values. it ranges from 0 to 1 where 1 means perfect fit

# R² for the validation set
R2_validatn_Linear = r2_score(y_validatn, y_validatn_predct_Linear)
R2_validatn_Ridge = r2_score(y_validatn, y_validatn_predct_Ridge)
R2_validatn_Lasso = r2_score(y_validatn, y_validatn_predct_Lasso)

# R² for the test set
R2_test_Linear = r2_score(y_test, y_test_predct_Linear)
R2_test_Ridge = r2_score(y_test, y_test_predct_Ridge)
R2_test_Lasso = r2_score(y_test, y_test_predct_Lasso)

# below are the results for the validation set
print("\nValidation Set Performance:")
print(f"Linear Regression - MSE: {MSE_validatn_Linear:.2f}, PCC: {PCC_validatn_Linear:.2f}, R²: {R2_validatn_Linear:.2f}")
print(f"Ridge Regression - MSE: {MSE_validatn_Ridge:.2f}, PCC: {PCC_validatn_Ridge:.2f}, R²: {R2_validatn_Ridge:.2f}")
print(f"Lasso Regression - MSE: {MSE_validatn_Lasso:.2f}, PCC: {PCC_validatn_Lasso:.2f}, R²: {R2_validatn_Lasso:.2f}")

# below are results for the test set
print("\nTest Set Performance:")
print(f"Linear Regression - MSE: {MSE_test_Linear:.2f}, PCC: {PCC_test_Linear:.2f}, R²: {R2_test_Linear:.2f}")
print(f"Ridge Regression - MSE: {MSE_test_Ridge:.2f}, PCC: {PCC_test_Ridge:.2f}, R²: {R2_test_Ridge:.2f}")
print(f"Lasso Regression - MSE: {MSE_test_Lasso:.2f}, PCC: {PCC_test_Lasso:.2f}, R²: {R2_test_Lasso:.2f}")

# Calculating the best alpha value
alphas = [0., 0.25, 0.5, 1., 1000.]
Alpha_Bridge = None
Ridge_B_MSE_validatn = float("inf")

Alpha_Blasso = None
Lasso_B_MSE_validatn = float("inf")


for alpha in alphas:
    # Ridge regression
    Ridge_model = Ridge(alpha=alpha)
    Ridge_model.fit(X_train, y_train)
    y_validatn_predct_Ridge = Ridge_model.predict(X_validatn)

    # Calculate MSE on validation set
    MSE_validatn_Ridge = mean_squared_error(y_validatn, y_validatn_predct_Ridge)
    #R2_validatn_Ridge = r2_score(y_validatn, y_validatn_predct_Ridge)
    #PCC_validatn_Ridge = np.corrcoef(y_validatn, y_validatn_predct_Ridge)[0, 1]
    
    print(f"\nRidge (alpha={alpha:.2f}) - Validation MSE: {MSE_validatn_Ridge:.2f}, R²: {R2_validatn_Ridge:.2f}, PCC: {PCC_validatn_Ridge:.2f}")
    

    if MSE_validatn_Ridge < Ridge_B_MSE_validatn:
        Ridge_B_MSE_validatn = MSE_validatn_Ridge
        Alpha_Bridge = alpha

    # Lasso regression
    Lasso_model = Lasso(alpha=alpha)
    Lasso_model.fit(X_train, y_train)
    y_validatn_predct_Lasso = Lasso_model.predict(X_validatn)
    
    # Calculate MSE on validation set
    MSE_validatn_Lasso = mean_squared_error(y_validatn, y_validatn_predct_Lasso)
    #R2_validatn_Lasso = r2_score(y_validatn, y_validatn_predct_Lasso)
    #PCC_validatn_Lasso = np.corrcoef(y_validatn, y_validatn_predct_Lasso)[0, 1]
    print(f"\nLasso (alpha={alpha:.2f}) - Validation MSE: { MSE_validatn_Lasso :.2f}, R²: {R2_validatn_Lasso:.2f}, PCC: { PCC_validatn_Lasso:.2f}")
    

    if MSE_validatn_Lasso < Lasso_B_MSE_validatn:
        Lasso_B_MSE_validatn = MSE_validatn_Lasso
        Alpha_Blasso = alpha
print("\nBelow are the best values of alpha :")
print(f"Best Ridge (alpha={Alpha_Bridge:.2f})")
print(f"Best Lasso (alpha={Alpha_Blasso:.2f})")

#here we are retraining the ridge model with best alpha

BestRidge_model = Ridge(alpha = Alpha_Bridge)
BestRidge_model.fit(X_train, y_train)
y_test_predct_Ridge =  BestRidge_model.predict(X_test)

#here we are retraining the lasso model with best alpha

BestLasso_model = Lasso(alpha = Alpha_Blasso)
BestLasso_model.fit(X_train, y_train)
y_test_predct_Lasso =  BestLasso_model.predict(X_test)

# Calculate performance metrics on the test set
MSE_test_Ridge = mean_squared_error(y_test, y_test_predct_Ridge)
R2_test_Ridge = r2_score(y_test, y_test_predct_Ridge)
PCC_test_Ridge = np.corrcoef(y_test,y_test_predct_Ridge)[0, 1]

MSE_test_Lasso = mean_squared_error(y_test, y_test_predct_Lasso)
R2_test_Lasso = r2_score(y_test, y_test_predct_Lasso)
PCC_test_Lasso = np.corrcoef(y_test,y_test_predct_Lasso)[0, 1]

print(f"Best Ridge (alpha={Alpha_Bridge:.2f}) - Test MSE: {MSE_test_Ridge:.2f}, R²: {R2_test_Ridge:.2f}, PCC: {PCC_test_Ridge:.2f}")
print(f"Best Lasso (alpha={Alpha_Blasso:.2f}) - Test MSE: {MSE_test_Lasso:.2f}, R²: {R2_test_Lasso:.2f}, PCC: {PCC_test_Lasso:.2f}")


#here we apply PolynomialFeatures to the feature part of the dataset (columns X1- X5), using the degree 5
#polynomial feature transforms feature to higher degree for instan x feature will be converted x2, x3 . This helps in studing non linear relationships between the different features in dataset

polynFtr = PolynomialFeatures(degree = 5)

#fit the transformer to training data set  and transforming it using fit_transform
X_train_polynFtr = polynFtr.fit_transform(X_train)

#now transform validation and test set using fit_transform()

X_validatn_polynFtr = polynFtr.transform(X_validatn)

X_test_polyFtr = polynFtr.transform(X_test)

X_train_polynFtr_df = pd.DataFrame(X_train_polynFtr, columns=polynFtr.get_feature_names_out(X_train.columns))

print(X_train_polynFtr_df.head())

# Fit models on transformed training data
Linear.fit(X_train_polynFtr, y_train)
Ridge_model.fit(X_train_polynFtr, y_train)
Lasso_model.fit(X_train_polynFtr, y_train)

#predictions on transformed validation set
y_validatn_predct_Linear =     Linear.predict(X_validatn_polynFtr)
y_validatn_predct_Ridge =     Ridge_model.predict(X_validatn_polynFtr)
y_validatn_predct_Lasso =     Lasso_model.predict(X_validatn_polynFtr)

#predictions on test set
y_test_predct_Linear =     Linear.predict(X_test_polyFtr)
y_test_predct_Ridge =     Ridge_model.predict(X_test_polyFtr)
y_test_predct_Lasso =     Lasso_model.predict(X_test_polyFtr)


# MSE for the validation set
MSE_validatn_Linear = mean_squared_error(y_validatn, y_validatn_predct_Linear)
MSE_validatn_Ridge = mean_squared_error(y_validatn, y_validatn_predct_Ridge)
MSE_validatn_Lasso = mean_squared_error(y_validatn, y_validatn_predct_Lasso)


# MSE for the test set
MSE_test_Linear = mean_squared_error(y_test, y_test_predct_Linear)
MSE_test_Ridge = mean_squared_error(y_test, y_test_predct_Ridge)
MSE_test_Lasso = mean_squared_error(y_test, y_test_predct_Lasso)


# PCC for the validation set
PCC_validatn_Linear = np.corrcoef(y_validatn, y_validatn_predct_Linear)[0, 1]
PCC_validatn_Ridge = np.corrcoef(y_validatn, y_validatn_predct_Ridge)[0, 1]
PCC_validatn_Lasso = np.corrcoef(y_validatn, y_validatn_predct_Lasso)[0, 1]

# PCC for the test set
PCC_test_Linear = np.corrcoef(y_test, y_test_predct_Linear)[0, 1]
PCC_test_Ridge = np.corrcoef(y_test, y_test_predct_Ridge)[0, 1]
PCC_test_Lasso = np.corrcoef(y_test, y_test_predct_Lasso)[0, 1]

#r2 will tells us how accurate are the predicted values. it ranges from 0 to 1 where 1 means perfect fit

# R² for the validation set
R2_validatn_Linear = r2_score(y_validatn, y_validatn_predct_Linear)
R2_validatn_Ridge = r2_score(y_validatn, y_validatn_predct_Ridge)
R2_validatn_Lasso = r2_score(y_validatn, y_validatn_predct_Lasso)

# R² for the test set
R2_test_Linear = r2_score(y_test, y_test_predct_Linear)
R2_test_Ridge = r2_score(y_test, y_test_predct_Ridge)
R2_test_Lasso = r2_score(y_test, y_test_predct_Lasso)


# below are the results for the validation set
print("\nValidation Set Performance:")
print(f"Linear Regression - MSE: {MSE_validatn_Linear:.2f}, PCC: {PCC_validatn_Linear:.2f}, R²: {R2_validatn_Linear:.2f}")
print(f"Ridge Regression - MSE: {MSE_validatn_Ridge:.2f}, PCC: {PCC_validatn_Ridge:.2f}, R²: {R2_validatn_Ridge:.2f}")
print(f"Lasso Regression - MSE: {MSE_validatn_Lasso:.2f}, PCC: {PCC_validatn_Lasso:.2f}, R²: {R2_validatn_Lasso:.2f}")

# below are results for the test set
print("\nTest Set Performance:")
print(f"Linear Regression - MSE: {MSE_test_Linear:.2f}, PCC: {PCC_test_Linear:.2f}, R²: {R2_test_Linear:.2f}")
print(f"Ridge Regression - MSE: {MSE_test_Ridge:.2f}, PCC: {PCC_test_Ridge:.2f}, R²: {R2_test_Ridge:.2f}")
print(f"Lasso Regression - MSE: {MSE_test_Lasso:.2f}, PCC: {PCC_test_Lasso:.2f}, R²: {R2_test_Lasso:.2f}")

# Calculating the best alpha value on transformed data
alphas = [0., 0.25, 0.5, 1., 1000.]
Alpha_Bridge = None
Ridge_B_MSE_validatn = float("inf")

Alpha_Blasso = None
Lasso_B_MSE_validatn = float("inf")


for alpha in alphas:
    # Ridge regression
    Ridge_model = Ridge(alpha=alpha)
    Ridge_model.fit(X_train_polynFtr, y_train)
    y_validatn_predct_Ridge = Ridge_model.predict(X_validatn_polynFtr)

    # Calculate MSE on validation set
    MSE_validatn_Ridge = mean_squared_error(y_validatn, y_validatn_predct_Ridge)
    #R2_validatn_Ridge = r2_score(y_validatn, y_validatn_predct_Ridge)
    #PCC_validatn_Ridge = np.corrcoef(y_validatn, y_validatn_predct_Ridge)[0, 1]
    
    print(f"\nRidge (alpha={alpha:.2f}) - Validation MSE: {MSE_validatn_Ridge:.2f}, R²: {R2_validatn_Ridge:.2f}, PCC: {PCC_validatn_Ridge:.2f}")
    

    if MSE_validatn_Ridge < Ridge_B_MSE_validatn:
        Ridge_B_MSE_validatn = MSE_validatn_Ridge
        Alpha_Bridge = alpha

    # Lasso regression
    Lasso_model = Lasso(alpha=alpha)
    Lasso_model.fit(X_train_polynFtr, y_train)
    y_validatn_predct_Lasso = Lasso_model.predict(X_validatn_polynFtr)
    
    # Calculate MSE on validation set
    MSE_validatn_Lasso = mean_squared_error(y_validatn, y_validatn_predct_Lasso)
    #R2_validatn_Lasso = r2_score(y_validatn, y_validatn_predct_Lasso)
    #PCC_validatn_Lasso = np.corrcoef(y_validatn, y_validatn_predct_Lasso)[0, 1]
    print(f"\nLasso (alpha={alpha:.2f}) - Validation MSE: { MSE_validatn_Lasso :.2f}, R²: {R2_validatn_Lasso:.2f}, PCC: { PCC_validatn_Lasso:.2f}")
    

    if MSE_validatn_Lasso < Lasso_B_MSE_validatn:
        Lasso_B_MSE_validatn = MSE_validatn_Lasso
        Alpha_Blasso = alpha
print("\nBelow are the best values of alpha after transformation :")
print(f"Best Ridge (alpha={Alpha_Bridge:.2f})")
print(f"Best Lasso (alpha={Alpha_Blasso:.2f})")

#here we are retraining the ridge model with best alpha

BestRidge_model = Ridge(alpha = Alpha_Bridge)
BestRidge_model.fit(X_train_polynFtr, y_train)
y_test_predct_Ridge =  BestRidge_model.predict(X_test_polyFtr)


#here we are retraining the lasso model with best alpha

BestLasso_model = Lasso(alpha = Alpha_Blasso)
BestLasso_model.fit(X_train_polynFtr, y_train)
y_test_predct_Lasso =  BestLasso_model.predict(X_test_polyFtr)

# Calculate performance metrics on the test set
MSE_test_Ridge = mean_squared_error(y_test, y_test_predct_Ridge)
R2_test_Ridge = r2_score(y_test, y_test_predct_Ridge)
PCC_test_Ridge = np.corrcoef(y_test,y_test_predct_Ridge)[0, 1]

MSE_test_Lasso = mean_squared_error(y_test, y_test_predct_Lasso)
R2_test_Lasso = r2_score(y_test, y_test_predct_Lasso)
PCC_test_Lasso = np.corrcoef(y_test,y_test_predct_Lasso)[0, 1]

print(f"Best Ridge after transformation (alpha={Alpha_Bridge:.2f}) - Test MSE: {MSE_test_Ridge:.2f}, R²: {R2_test_Ridge:.2f}, PCC: {PCC_test_Ridge:.2f}")
print(f"Best Lasso after transformation(alpha={Alpha_Blasso:.2f}) - Test MSE: {MSE_test_Lasso:.2f}, R²: {R2_test_Lasso:.2f}, PCC: {PCC_test_Lasso:.2f}")


# Define feature names from polynomial transformation
Fnames = polynFtr.get_feature_names_out(X_train.columns)


Ridge_coef = BestRidge_model.coef_  
Lasso_coef = BestLasso_model.coef_  

# here match the lengths
print("Number of feature names:", len(Fnames))
print("Number of Ridge coefficients:", len(Ridge_coef))
print("Number of Lasso coefficients:", len(Lasso_coef))

if len(Fnames) != len(Ridge_coef) or len(Fnames) != len(Lasso_coef):
    raise ValueError("Length of does not")


Ridge_model_df = pd.DataFrame({
    'Feature': Fnames,
    'Coefficient': Ridge_coef
}).sort_values(by='Coefficient', ascending=False)

Lasso_model_df = pd.DataFrame({
    'Feature': Fnames,
    'Coefficient': Lasso_coef
}).sort_values(by='Coefficient', ascending=False)

print("\nRidge Model Coefficients:")
print(Ridge_model_df)

print("\nLasso Model Coefficients:")
print(Lasso_model_df)

# To analyze the coefficients from the Ridge and Lasso regression models, let’s break down the information provided:

# Ridge Regression Coefficients:
# Most Important Features:

# Ridge regression coefficients are generally small due to the regularization, so the "most important" features are those with the largest absolute values. In your output, the features with the largest coefficients are:

# X1 X3 X4^3: 0.000012
# X2 X4^3: 0.000012
# X3 X4^2 X5^2: 0.000012
# X2^2 X3^2 X5: 0.000011
# These features have the highest coefficients in Ridge regression, indicating they have relatively more influence compared to others in the model.

# Features with Weights Close to Zero:

# Many features have coefficients close to zero in Ridge regression. This indicates that these features have minimal impact on the model's predictions. Some of these features include:

# X2^2 X4^2 X5: -0.000007
# X3^2 X4^2 X5: -0.000009
# X1^2 X3^2 X4: -0.000011
# X1^5: -0.000011
# X1^2 X3 X4^2: -0.000012
# Features with coefficients near zero are less influential in Ridge regression.

# Lasso Regression Coefficients:
# Most Important Features:

# Lasso regression tends to shrink some coefficients to zero, making it easier to identify the most influential features. The feature with the largest coefficient in Lasso regression is:

# X3: -0.450147
# This large magnitude indicates that X3 is the most influential feature in the Lasso model.

# Features with Weights Close to Zero:

# Features with coefficients near zero are excluded or have minimal impact in Lasso regression. Some of the features with smaller coefficients include:

# X4^2: -0.001629
# X4 X5: -0.001722
# X2 X3: -0.002694
# X1^2: -0.006865
# These features have smaller absolute coefficients compared to the most important features and contribute less to the model.

# Summary:
# Ridge Regression: The most important features are those with the highest non-zero coefficients, though they are generally small. Features with coefficients close to zero are less influential.
# Lasso Regression: The most important feature is X3 with a significant coefficient. Features with coefficients close to zero are excluded from the model, showing minimal or no impact on the predictions.








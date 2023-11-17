from sklearn.model_selection import validation_curve
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
    
def plot_validation_curve(model,model_name,X,y,param_range,param_name,scoring="neg_mean_absolute_error",cv=5,
                          verbose=0,xscale="log",lw = 1.5,opacity=0.5,figsize=(10,5)):
    """
    This function cross_validated training data and plots nice graphs for training 
    as well as for validation for each hiperparameter. It shows std of scores in each split.
    
    This function will make sense as long as high score is a good thing. Thats why 
    sklearn has built in score neg_mean_absolute_error to invert the logic and turn error
    that is best low into score that is best high.

    Arguments:
    X,y - data to be fitted on model (training data).
    model - model to use, can have some hiperparameter already set. 
            Only "param_name" parameter will be tested and changed.
    param_range - range of parameters to use (in logarithmic space). Example: np.logspace(-2, 4, 50)
    param_name - very specific name of hiperparameter (same as field in model's class).
    scoring - very specific sklearn predefined scoring function name.
    model_name - just the name of model to show on plot.
    xscale - depending on param_range it may be best to use "log" or "lin".
    cv - number of splits in cross-validation. 
    lw - line width.
    opacity - .
    figsize - change size of the graph, accepts (width,height).
    """

    train_scores, validation_scores = validation_curve(
        model,X,y,
        param_name=param_name,
        param_range=param_range,
        scoring=scoring,
        cv=cv,
        verbose=verbose
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    fig,ax = plt.subplots(1)
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])

    ax.set_title(f"Validation Curve With {model_name}")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Score")
    
    ax.plot(
        param_range, train_scores_mean, 
        label="Training score", color="darkorange", lw=lw
    )
    ax.fill_between(
        param_range, 
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=opacity, color="darkorange", lw=lw
    )
    ax.plot(
        param_range, validation_scores_mean, 
        label="Validation score", color="navy", lw=lw,
    )
    ax.fill_between(
        param_range,
        validation_scores_mean - validation_scores_std,
        validation_scores_mean + validation_scores_std,
        alpha=opacity, color="navy", lw=lw
    )
    
    if(xscale == "log"):
        ax.set_xscale('log')

    ax.legend(loc="best")
    plt.show()

    best_parameter,best_score = sorted(zip(param_range, validation_scores_mean), key=lambda x:x[1])[-1]

    print(f"Best Parameter = {best_parameter} And Best Score = {best_score}")

    return best_parameter, best_score, train_scores_mean, validation_scores_mean

def convert_to_categoricals(y_test,y_pred,n_categories):
    """
    The idea is to convert real and predicted labels to categoricals
    to test the f1 score of a model. 
    n_categories - should reflect number of natural categories of data,
    Example: 3 categories -> cold,warm,hot.
    First and last bin are big to guarantee no NaNs.
    """
    le = LabelEncoder()
    y_pred_cat = le.fit_transform(pd.cut(y_pred,n_categories,retbins=True)[0])
    y_test_cat = le.fit_transform(pd.cut(y_test,n_categories,retbins=True)[0])
    return y_test_cat, y_pred_cat

def test_model(model,X_train,y_train,X_test,y_test,silent=True):
    """
    Very simple helper function with narrow usage.
    ATTENTION! This returns error, not score like previously.
    """
    model.fit(X_train,y_train)

    y_pred = model.predict(X_train)
    train_score = MAPE(y_train,y_pred)

    y_pred = model.predict(X_test)
    score = MAPE(y_test,y_pred)
    if(not silent):
        print(f"Test score = {score} and train score = {train_score}")
    return score,train_score

def MAPE(y_test,y_pred):
    """
    Mean Absolute Percentage Error
    Sklearn does not have this because of division by zero problem.
    Use only with data above zero.
    """
    return sum(abs(-1 * (y_pred/y_test) + 1)) * 100 / len(y_pred)

def save_model(name,parameters,model,error,models_parameters):
    """
    Saves hiperparameters, test score and model to 
    argument lsit "models_parameters".
    Arguments:
    name - name of model
    parameters - ex. {"alpha" : alpha}
    model - sklearn model object
    error - test error ex. mae
    """
    d = {
        "name" : name, 
        "parameters" : parameters, 
        "model" : model, 
        "error" : error
    }
    models_parameters.append(d)

def extend_to_poly_features(X,degree):
    """
    Extends features of a dataframe by adding columns
    with powers of original data. 
    """
    X = X.copy()
    X_out = X.copy()
    if(degree < 2):
        return X
    for i in range(2,degree+1):
        Xpow = X ** i
        Xpow.columns = [f"{c}^{i}" for c in X.columns]
        X_out = pd.concat([X_out.reset_index(),Xpow.reset_index()],axis=1).drop(columns=["index"])
    return X_out
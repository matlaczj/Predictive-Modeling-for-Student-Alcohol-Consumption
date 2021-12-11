from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np
    
def plot_validation_curve(model,model_name,X,y,param_range,param_name,scoring,cv=5,
                          verbose=0,xscale="log",lw = 1.5,opacity=0.5,figsize=(10,5)):
    """
    This function cross_validated training data and plots nice graphs for training 
    as well as for validation for each hiperparameter. It shows std of scores in each split.
    
    This function will make sense as long as high score is a good thing. Thats why 
    sklearn has built in score neg_mean_absolute_error to invert the logic and turn error
    that is best low into score that is best high.

    Arguments:
    X,y - data to be fitted on model (training data).
    model - model to use (without specified hiperparameters).
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
        label="Cross-validation score", color="navy", lw=lw,
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

    return train_scores_mean, validation_scores_mean
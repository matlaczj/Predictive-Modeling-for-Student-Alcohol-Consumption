from sklearn.model_selection import cross_validate
from statistics import mean
import matplotlib.pyplot as plt


def cross_validate_models(models,X_train,y_train,printEveryN,scoring='neg_mean_absolute_error',silent=False):
    """
    models - list of preconfigured models.
    X_train,y_train - models will be fitted on this data.
    printEveryN - print status every n-models to avoid spam.
    scoring - what metric to use on each split.
    """
    scores = []
    for idx,clf in enumerate(models):
        if(idx % printEveryN == 0 and not silent):
            print(f"{idx}/{len(models)}",end=" ")
        scores.append(cross_validate(clf, X_train, y_train, cv=5, return_train_score=True, scoring=scoring))
    train_mean_mae = [mean(abs(s['train_score'])) for s in scores]
    val_mean_mae = [mean(abs(s['test_score'])) for s in scores]
    return train_mean_mae,val_mean_mae
    
def plot_train_vs_validation_scores(train_metrics, val_metrics, hiperparameters, titles, hiperparameterName):
    """
    train_scores,val_scores - metrics to be plotted for easy comparison.
    hiperparameters - list of crossvalidated hiperparameter values to plot on x.
    titles - list of str, names for each of 2 graphs.
    hiperparameterName - name of crossvalidated hiperparameter.
    Does not matter what the metric actually is. 
    """
    fig,axes = plt.subplots(1,2)
    fig.set_figwidth(10)
    axes[0].plot(hiperparameters,train_metrics, c="red")
    axes[1].plot(hiperparameters,val_metrics, c="green")
    axes[0].set_title(titles[0])
    axes[1].set_title(titles[1])
    axes[0].set_xlabel(hiperparameterName)
    axes[1].set_xlabel(hiperparameterName)
    axes[0].set_ylabel("metric")
    axes[1].set_ylabel("metric")
    return fig,axes
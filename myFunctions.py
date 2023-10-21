#!/usr/bin/env python
# coding: utf-8


# Import the main libraries for the functions below
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import f_regression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, BayesianRidge
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from warnings import simplefilter, filterwarnings
from collections import defaultdict
import matplotlib as mpl
import squarify
import re
import shap
import lime
from collections import defaultdict
from collections import OrderedDict



def modelSelection(X, y, n_split = 5, random_state = 42):
    models = []
    
    # Defining the models
    models.append(('LogReg', LogisticRegression(class_weight = 'balanced', max_iter = 200, random_state = random_state)))
    models.append(('Tree', DecisionTreeClassifier(random_state = random_state)))
    models.append(('RandForest', RandomForestClassifier(class_weight = 'balanced', random_state = random_state)))
    models.append(('NeuralNet', MLPClassifier(solver = 'adam', hidden_layer_sizes=(120, 60, 10), max_iter=100, random_state = random_state)))
    models.append(('NaiveBayes', GaussianNB()))
    models.append(('XGB', XGBClassifier(random_state = random_state)))
    #models.append(('KNearN', KNeighborsClassifier()))
    models.append(('SuppVec', SVC(class_weight = 'balanced', random_state = random_state)))
 
    
    # Runing the Cross validation in Recall
    recallResult = []
    recallModelName = []
    
    for name, model in models:
        print(f"Processing Recall for model: {name}")  # Debug print
        kFold = StratifiedKFold(n_splits = n_split, shuffle = True, random_state = random_state)
        cvResults = cross_val_score(model, X, y, cv=kFold, scoring='recall')
        recallResult.append(cvResults)
        recallModelName.append(name)
        
    # Runing the Cross validation in F1
    f1Result = []
    f1ModelName = []
        
    for name, model in models:
        print(f"Processing F1 for model: {name}")  # Debug print
        kFold = StratifiedKFold(n_splits = n_split, shuffle = True, random_state = random_state)
        cvResults = cross_val_score(model, X, y, cv=kFold, scoring='f1')
        f1Result.append(cvResults)
        f1ModelName.append(name)    
     
    # Runing the Cross validation in Precision
    precisionResult = []
    precisionModelName = []
    
    for name, model in models:
        print(f"Processing Precision for model: {name}")  # Debug print
        kFold = StratifiedKFold(n_splits = n_split, shuffle = True, random_state = random_state)
        cvResults = cross_val_score(model, X, y, cv=kFold, scoring='precision')
        precisionResult.append(cvResults)
        precisionModelName.append(name) 
    
    # Setting the font
    mpl.rcParams['font.family'] = 'Times New Roman'
    font_size = 13  # Change this value to adjust the font size
    
    # Plotting the results    
    fig, axes = plt.subplots(3, 1, figsize=(11.8, 10), sharey=True)
    
    axes[0].boxplot(recallResult) 
    axes[0].set_title('Recall of Classification Algorithms', fontsize = font_size+2)
    axes[0].set_ylabel('Recall', fontsize = font_size+1)
    axes[0].set_xticklabels([])
    
    axes[1].boxplot(precisionResult)
    axes[1].set_title('Precision of Classification Algorithms', fontsize = font_size+2)
    axes[1].set_ylabel('Precision', fontsize = font_size+1)
    axes[1].set_xticklabels([])
    
    axes[2].boxplot(f1Result)
    axes[2].set_title('F1 of Classification Algorithms', fontsize = font_size+2)
    axes[2].set_xlabel('Algorithm', fontsize = font_size+1)
    axes[2].set_ylabel('F1', fontsize = font_size+1)
    axes[2].set_xticklabels(f1ModelName, fontsize = font_size+1)
    
    fig.tight_layout(pad=3.0)
    
    
    # Save the figure with 800 dpi
    fig.savefig("model_selection_plot.png", dpi=1200, bbox_inches='tight')
    
    # Show the plots
    plt.show()

# writing a function for Hyperparameter tuning

def parameterTuning (X, y, paramGrid, estimator, split = 0.75, scoring = 'f1', n_split = 3, random_state = 42):
    
    # Split the data set into test and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify = y, 
        train_size = split, 
        random_state = random_state)
    
    # Strafified Cross Validator
    cv = StratifiedKFold(
        n_splits = n_split,
        shuffle = True,
        random_state = random_state
    )
    
    # Creating Grid Search object
    gridSearch = GridSearchCV(
        estimator = estimator,
        param_grid = paramGrid,
        cv = cv,
        scoring = scoring
    )
    
    # Conduct search
    gridResult = gridSearch.fit(X_train, y_train)
    
    # Get the best parameters
    bestParams = gridResult.best_params_
    
    print(bestParams)
    
    return bestParams


# Function for the confusion matrix

def confusionMatrix (X, y, model, split = 0.75, random_state = 42):
    
    # Splitting the data into a Testing and tarinign set
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, train_size = split, random_state = random_state)
    
    #fit the training data to the algorithm
    model.fit(X_train, y_train)
    
    # Run the algorithm on the test data
    y_pred = model.predict(X_test)
    
    # Prepare and plot the confusions matrix
    
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    print('\033[1m'+'Performance Metrics' + '\033[0m' + '\nRecall: {:.2f}% \nPrecision: {:.2f}%\nF1 Score: {:.2f}%'.format(
        recall_score(y_test, y_pred)*100, precision_score(y_test, y_pred)*100, f1_score(y_test, y_pred)*100))
    display.plot()
    plt.show()

# Function for the Shap Value

def shapValue(X_train, X_test, model,  K=25, random_state=42, plotting = False):
    # Sample K data points from the training data
    X_train_sample = shap.sample(X_train, K, random_state=random_state)
    
    # Define a wrapped predict_proba function
    def wrappedPredict(data):
       return model.predict_proba(data)

    # Create the explainer object
    try:
        explainer = shap.KernelExplainer(model.predict_proba, X_train_sample)
    except AttributeError:
        explainer = shap.KernelExplainer(wrappedPredict, X_train_sample)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Create an Explanation object
    explainer = shap.Explanation(values=shap_values[0], 
                                 data=X_test.values, 
                                 feature_names=X_train.columns)
    
    # Calculate the mean absolute SHAP values for each feature
    mean_shap_values = np.abs(shap_values[0]).mean(0)
    
    # Create a DataFrame for feature importance
    feature_importance = pd.DataFrame(list(zip(X_train.columns, mean_shap_values)), columns=['col_name', 'feature_importance_values'])
    feature_importance.sort_values(by=['feature_importance_values'], ascending=False, inplace=True)

    # Create a dictionary to return
    shap_dict = feature_importance.set_index('col_name')['feature_importance_values'].to_dict()
    
    # Plotting
    if plotting:
        shap.summary_plot(shap_values[0], X_test, X_train.columns.tolist())
        shap.plots.bar(explainer, max_display=12)
    
    return shap_dict
    

def localExplanation(X, y, model, explainInstance=0, plotting = False):
    numberFeatures = X.shape[1]
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values,
        training_labels=y,
        feature_names=X.columns.tolist(),
        class_names=['0', '1'],
        mode='classification'
    )

    feature_names = X.columns.tolist()
    feature_effects = {name: [] for name in feature_names}
    
    indices = [explainInstance]  # Just for the specific row
   
    for i in indices:
        exp = explainer.explain_instance(
            X.iloc[i].values,
            model.predict_proba,
            num_features=numberFeatures
        )
        
        for feature_condition, effect in exp.as_list():
            # Use regular expressions to find the feature name
            for feature_name in feature_names:
                if re.search(rf'\b{feature_name}\b', feature_condition):
                    feature_effects[feature_name].append(effect)
                    break  # No need to check the other elements once a match is found
                
    mean_lime_values = {k: np.mean(v) if v else 0 for k, v in feature_effects.items()}
    
    if plotting:
        # Plotting
        exp.show_in_notebook()
    
    return mean_lime_values

def plot_mean_shap_from_dict(shap_values_dict):
    sorted_items = sorted(shap_values_dict.items(), key=lambda x: x[1], reverse=True)
    features, values = zip(*sorted_items)
    
    plt.figure(figsize=(10, len(features)*0.4))
    plt.barh(features, values, color='royalblue')
    plt.xlabel("Mean SHAP Value")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()  # to display the most important feature at the top
    plt.savefig('shap_mean_plot.png', dpi=800, bbox_inches='tight')
    plt.show()
    



# Function to calculate NHHI
def calc_NHHI(shap_dict):
    total = sum(shap_dict.values())
    squared_proportions = [(x / total) ** 2 for x in shap_dict.values()]
    NHHI = sum(squared_proportions)
    return NHHI

# Function to calculate composite metrics for multiple models and store SHAP values
def calc_composite_and_store_shap(X_train, X_test, y_test, models, metric="f1", alpha=0.5, K=25, random_state=42, T=None):
    NHHIs = []
    NHHIs_dict = {}  # Dictionary to store NHHI values for each model
    composite_metrics = {}
    shap_values_storage = {}
    
    # Calculate the NHHI for each model and store SHAP values
    for model in models:
        shap_dict = shapValue(X_train, X_test, model, K=K, random_state=random_state, plotting=False)
        NHHI = calc_NHHI(shap_dict)
        NHHIs.append(NHHI)
        
        model_name = str(model).split("(")[0]
        shap_values_storage[model_name] = shap_dict
        
        NHHIs_dict[model_name] = NHHI  # Store the NHHI value for the current model
    
    # If T is not provided, calculate the mean NHHI
    if T is None:
        T = np.mean(NHHIs)
    
    # Calculate the composite metric for each model
    for i, model in enumerate(models):
        NHHI = NHHIs[i]
        
        # Get model predictions
        y_pred = model.predict(X_test)
        
        # Get chosen metric (Recall, Precision, F1)
        if metric == "f1":
            chosen_metric = f1_score(y_test, y_pred)
        elif metric == "precision":
            chosen_metric = precision_score(y_test, y_pred)
        elif metric == "recall":
            chosen_metric = recall_score(y_test, y_pred)
        else:
            raise ValueError("Invalid metric choice. Please choose 'f1', 'precision', or 'recall'.")
        
        # Calculate penalty term based on deviation from ideal NHHI (T)
        P = abs(NHHI - T)
        
        # Calculate composite metric
        composite_metric = chosen_metric * (1 - alpha * P)
        model_name = str(model).split("(")[0]
        
        composite_metrics[model_name] = composite_metric
        
        # Sort the composite metrics
        composite_metrics = OrderedDict(sorted(composite_metrics.items(), key=lambda x: x[1], reverse=True))
        
        # Sort the NHHI
        NHHIs_dict =  OrderedDict(sorted(NHHIs_dict.items(), key=lambda x: x[1], reverse=False))
    
    return composite_metrics, shap_values_storage, NHHIs_dict  


def generateTreeMap(shap_values_storage,X, y, model, modelName, split = 0.75, K = 25, random_state = 42, explainInstance=None, sortBy = 'SHAP', shapPercentageCutoff = 0, plottingSHAP = False, plottingLIME = False, legend = False, plotName = "treemapPlot:png"):
    
    plt.rcParams['font.size'] = 16  
    plt.rcParams['font.family'] = 'Times New Roman'  
    
    # Splitting the data into a Testing and tarinign set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify = y, 
        train_size = split, 
        random_state = random_state)
    
    #Get the Shap values from the NHHI function
    if not isinstance(modelName, str):
        modelName = modelName.__name__
    
    if modelName not in shap_values_storage:
        raise ValueError(f"No SHAP values found for model '{modelName}' in the storage.")
    
    shap_values_dict = shap_values_storage[modelName]
    
    if plottingSHAP:
        # Plot the mean SHAP values
        plot_mean_shap_from_dict(shap_values_dict)
    
    #shap_values_dict = shapValue(X_train, X_test, model, K, random_state, plotting = plottingSHAP)  # Assuming shapValue returns absolute mean SHAP values
    
    # Get the feature names from the dataset
    featureNames = X_train.columns.tolist()
    
    # Get the LIME values
    lime_values_dict = localExplanation(X, y, model, explainInstance=explainInstance, plotting = plottingLIME)
    
    # Make sure shap_values and lime_values are ordered the same way
    shap_values = [shap_values_dict[feature] for feature in featureNames]
    lime_values = [lime_values_dict[feature] for feature in featureNames]
    
    # Normalize LIME values to fit into a color map
    lime_max_abs = max(max(lime_values), abs(min(lime_values)))
    norm = mcolors.Normalize(-lime_max_abs, lime_max_abs)  # I've set the range from -lime_max_abs to lime_max_abs
    colors = [plt.cm.RdYlGn(norm(value)) for value in lime_values]
    
    # Calculate SHAP value percentages
    total_shap = sum(shap_values)
    shap_percentages = [(shap / total_shap) * 100 for shap in shap_values]
    features_with_percentages = [f'{feature} ({percentage:.2f}%)' for feature, percentage in zip(featureNames, shap_percentages)]
    
    
    # Zip all lists together
    zipped_lists = zip(shap_values, lime_values, featureNames, shap_percentages)

    # Sort by SHAP values (you can also sort by absolute LIME values by changing the key)
    if sortBy == 'SHAP':
        sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[0], reverse=True)
    elif sortBy == 'LIME':
        sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[1], reverse=True)
    else:
        raise ValueError("Invalid value for sortBy. Choose 'SHAP' or 'LIME'.")
    
    # Unzip sorted lists
    sorted_shap_values, sorted_lime_values, sorted_feature_names, sorted_shap_percentages = zip(*sorted_zipped_lists)

    # Filter features with SHAP percentage higher than the cutoff
    filtered_indices = [i for i, perc in enumerate(sorted_shap_percentages) if perc >= shapPercentageCutoff]
    filtered_shap_values = [sorted_shap_values[i] for i in filtered_indices]
    filtered_lime_values = [sorted_lime_values[i] for i in filtered_indices]
    filtered_feature_names = [sorted_feature_names[i] for i in filtered_indices]
    filtered_shap_percentages = [sorted_shap_percentages[i] for i in filtered_indices]

    features_with_percentages = [f'{name} ({perc:.2f}%)' for name, perc in zip(filtered_feature_names, filtered_shap_percentages)]

    # Normalize LIME values for the color map
    lime_max_abs = max(max(filtered_lime_values), abs(min(filtered_lime_values)))
    norm = mcolors.Normalize(-lime_max_abs, lime_max_abs)
    filtered_colors = [plt.cm.RdYlGn(norm(value)) for value in filtered_lime_values]
    
    if legend:
        # Create a mapping of original feature names to numbers
        mapping = {name: i for i, name in enumerate(X_train.columns)}
        
        # Create a legend DataFrame
        legend_df = pd.DataFrame(list(mapping.items()), columns=['Original_Name', 'Numeric_Value'])
        
        # Save legend to CSV
        legend_df.to_csv("legend.csv", index=False)
        
        # For visualization: replace original feature names with numbers in the treemap
        filtered_feature_numbers = [mapping[name] for name in filtered_feature_names]
        features_with_numbers_and_percentages = [f'{num} ({perc:.2f}%)' for num, perc in zip(filtered_feature_numbers, filtered_shap_percentages)]
        
        # Replace original feature names with numbers for the treemap
        filtered_feature_numbers = [mapping[name] for name in filtered_feature_names]
        features_with_numbers_and_percentages = [f'{num} ({perc:.2f}%)' for num, perc in zip(filtered_feature_numbers, filtered_shap_percentages)]
        
        # Create a treemap
        plt.figure(figsize=(11.8, 11.8))
        squarify.plot(sizes=filtered_shap_values, label=features_with_numbers_and_percentages, color=filtered_colors, alpha=0.7, edgecolor="k", linewidth=1, pad = 1)
        plt.title("Feature Importance and Effect", fontname ='Times New Roman', fontsize=16)
        plt.axis('off')

        # Add color bar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=np.linspace(-lime_max_abs, lime_max_abs, 5))
        cbar.set_label('Normalized LIME value', fontname = 'Times New Roman', fontsize = 14)
        cbar.ax.tick_params(labelsize=14)

       #save the plot
        plt.savefig(plotName, dpi=1200, bbox_inches='tight')
        
        plt.show()
        
    else:
        # Create a treemap
        plt.figure(figsize=(11.8, 11.8))
        squarify.plot(sizes=filtered_shap_values, label=features_with_percentages, color=filtered_colors, alpha=0.7, edgecolor="k", linewidth=1, pad = 1)
        plt.title("Feature Importance and Effect", fontname ='Times New Roman', fontsize=16)
        plt.axis('off')

        # Add color bar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=np.linspace(-lime_max_abs, lime_max_abs, 5))
        cbar.set_label('Normalized LIME value', fontsize = 14)
        cbar.ax.tick_params(labelsize=14)
       
        #save the plot
        plt.savefig(plotName, dpi=1200, bbox_inches='tight')
        
        #show the plot
        plt.show()
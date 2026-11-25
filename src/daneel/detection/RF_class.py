#Importing all useful packages
import warnings
warnings.filterwarnings('ignore')

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#try:
 #   import IPython
  #  working_directory = "/".join(
   #         IPython.extract_module_locals()[1]["__vsc_ipynb_file__"].split("/")[:-1]
    #    )
    #print("Setting working directory to: ", working_directory)
    #print(os.chdir(working_directory))
#except Exception as e:
 #   print("It was impossible to set your directory as the current one because of the following message")
  #  print(e)
   # print("The working directory is: ", os.getcwd())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
)

import joblib


class randomforest:
    def __init__(self, params_dict):
        #with open(configuration, 'r') as file:
         #   config = yaml.safe_load(file)

        #I try to modify the reading of the input file as in the transit_method class
        #Model Parameters
        #self.model_path = config["RandomForest"]["model"]
        #self.dataset_path = config["RandomForest"]["dataset"]
        self.model_path = params_dict.get("model")
        self.dataset_path = params_dict.get("dataset")

        #Reading the dataset
        #self.df = pd.read_csv(self.dataset_path)

        #Reading the model
        self.rf_model = joblib.load(self.model_path)
        #self.n_bins = config["RandomForest"]["nbins"] #number of bins= number of bins from training
        self.n_bins = params_dict.get("nbins")
        #self.threshold = float(np.load(config["RandomForest"]["threshold"]))
        self.threshold = float(np.load(params_dict.get("threshold")))

        #outputs
        #self.output_prediction = config["RandomForest"]["output_csv"]
        #self.classification = config["RandomForest"]["statistics"]
        self.output_prediction = params_dict.get("output_csv")   
        self.classification = params_dict.get("statistics")
        
    def load_data(self):
        print("Loading dataset...")

        #loading the CSV
        self.df = pd.read_csv(self.dataset_path)

        #print(f"Dataset: {self.df.shape[0]} samples")

        # Extract flux columns
        flux_cols = [
            f'flux_{i:04d}' for i in range(self.n_bins)
        ]
        flux_err_cols = [
            f'flux_err_{i:04d}' for i in range(self.n_bins)
        ]
        # Extract features
        self.X = self.df[flux_cols].values
        self.X_err = self.df[flux_err_cols].values

        #Extract labels 
        #it may be the dataset does not have the column label which says if it is a exoplamnet or not (1 or 0), like this it should
        #gave it back only if present
        if "label" in self.df.columns:
            self.y = self.df["label"].values
        else:
            self.y = None

        # Extract metadata if available
        metadata_cols = [
            'toi_name',
            'tic',
            'disp',
            'period_d',
            't0_bjd',
            'dur_hr',
            'sector'
        ]

        available_metadata = [
            col for col in metadata_cols
            if col in self.df.columns
        ]

        self.metadata = self.df[available_metadata].copy()

        print("...loading complteted.")

        return self.X #, metadata

    def prediction(self): 
        print("Predicting...")
         # Calculate prediction probabilities
        self.prob = self.rf_model.predict_proba(self.X)[:, 1]
        
        #Apply threshold
        self.pred = (self.prob >= self.threshold).astype(int)

        print("...prediction done.")
        return self.pred, self.prob

    def save_results_prediction(self):

        
        self.results = self.metadata[["toi_name", "tic"]].copy()
        self.results["probability"], self.results["prediction"] = self.prob, self.pred
        self.results.to_csv(self.output_prediction, index=False) #this generates a csv files with the name of the planet, the probability
                                                                #and the predicition
        print("Saving prediction results")
        return self.results
 
    def classification_statistics(self):

        # Check that labels are available
        if self.y is None:
            raise ValueError("True labels are not available.")

        # Accuracy
        accuracy = accuracy_score(self.y, self.pred)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            self.y, self.pred, labels=[0, 1]
        ).ravel()

        # Classification report
        report = classification_report(
            self.y, self.pred, zero_division=0
        )

        # ROC-AUC
        auc = roc_auc_score(self.y, self.prob)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(self.y, self.prob)

        # Write statistics to file
        with open(self.classification, "w") as file:

            file.write("CLASSIFICATION STATISTICS\n")
            file.write("========================\n\n")

            file.write(f"Threshold: {self.threshold:.4f}\n\n")

            file.write(f"Accuracy: {accuracy:.4f}\n")
            file.write(f"ROC-AUC: {auc:.4f}\n\n")

            file.write("Confusion Matrix\n")
            file.write("----------------\n")
            file.write(f"TN: {tn}\n")
            file.write(f"FP: {fp}\n")
            file.write(f"FN: {fn}\n")
            file.write(f"TP: {tp}\n\n")
            
            file.write("Classification Report\n")
            file.write("--------------------\n")
            file.write(report)

            file.write("\nROC Curve data\n")
            file.write("--------------\n")
            file.write("FPR: " + str(fpr) + "\n")
            file.write("TPR: " + str(tpr) + "\n")
            file.write("Thresholds: " + str(thresholds) + "\n")

        print("Classification statistics saved.")
        
    def run_RandomForest(self):
        print("\nRandom Forest classifier")
        print("----------------------------")

        #Load data
        self.load_data()

        #Compute predictions
        self.prediction()

        #Generate and save predictions
        self.save_results_prediction()

        #Compute classification statistics if labels are available
        if self.y is not None:
            print("Computing classification statistics")
            self.classification_statistics()
            print(f"Done, saved to: {self.classification}")
        else:
            print("Labels not available, statistics skipped.")
            
        print("----------------------------------")
        print("Random Forest classification completed.\n")

        return self.results
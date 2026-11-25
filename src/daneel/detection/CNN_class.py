#Importing usefule packages
import warnings
warnings.filterwarnings('ignore')
import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#try:
 #   import IPython
  #  working_directory = "/".join(
   #         IPython.extract_module_locals()[1]["__vsc_ipynb_file__"].split("/")[:-1]
    #    )
   # print("Setting working directory to: ", working_directory)
   # print(os.chdir(working_directory))
#except Exception as e:
 #   print("It was impossible to set your directory as the current one because of the following message")
  #  print(e)
   # print("The working directory is: ", os.getcwd())

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K

class CNNclassifier_class:

    #Initialization function
    def __init__(self, params_dict):
        #with open(configuration, 'r') as file:
         #   config = yaml.safe_load(file)

            
        #I try to modify the parameters reading part to make it similar to what is done for the tarnsit
        #Model Parameters, change with CNN paths
        #self.model_path = config["CNN"]["model"]
        #self.dataset_path = config["CNN"]["dataset"]
        self.model_path = params_dict.get("model")
        self.dataset_path = params_dict.get("daset")

        #Reading the dataset
        #self.df = pd.read_csv(self.dataset_path)

        #Reading the model
        self.CNN_model = tf.keras.models.load_model(self.model_path, compile=False)
        #self.n_bins = config["CNN"]["nbins"] #number of bins= number of bins from training
        self.n_bins = params_dict.get("nbins")
        #self.threshold = float(np.load(config["CNN"]["threshold"]))
        self.threshold = float(np.load(params_dict.get("threshold")))
        #self.scaler = joblib.load(config["CNN"]["scaler"])
        self.scaler = joblib.load(params_dict.get("scaler)")
        #A quanto pare va aggiunto lo scaler da salvere insieme al modello. Basta far re-runnare la pipeline fino a load_data

        #outputs
        #self.output_prediction = config["CNN"]["output_csv"]
        #self.classification = config["CNN"]["statistics"]
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
        self.X_noscaled = self.df[flux_cols].values
        self.X_err = self.df[flux_err_cols].values

        #Scaling the data and reshaping
        self.X_scaled = self.scaler.transform(self.X_noscaled)
        self.X = self.X_scaled.reshape(-1, self.n_bins, 1)

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
         #Calculate prediction probabilities
        self.prob = self.CNN_model.predict(self.X, verbose=0).flatten()
        
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

         #print(f"Statistics saved to {self.statistics_path}")

    def run_CNN(self):
        print("\nCNN classifier")
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
        print("CNN classification completed.\n")

        return self.results
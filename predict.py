import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler

# Clone the repository if not already present
if not os.path.exists("/usr/local/Sublimation_enthalpy_model"):
    os.chdir("/usr/local")
    os.system("git clone -q https://github.com/yifan950/Sublimation_enthalpy_model")

# Paths to the required files
scaler_path = "/usr/local/Sublimation_enthalpy_model/scaler.save"
model_path = "/usr/local/Sublimation_enthalpy_model/model.pkl"

# Load the scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def compute_descriptors(smiles):
    """
    Compute RDKit molecular descriptors for a given SMILES string.

    Parameters:
        smiles (str): The SMILES string of the molecule.

    Returns:
        np.array: A NumPy array of molecular descriptors.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Extract descriptors
    descriptor_values = [func(mol) for _, func in Descriptors.descList]
    return np.array(descriptor_values)

def get_smiles_input():
    """Request user input for a SMILES string."""
    return input("Enter a SMILES string: ")

def predict_sublimation_enthalpy(smiles):
    """Predict the sublimation enthalpy for a given SMILES string."""
    try:
        # Compute molecular descriptors
        descriptors = compute_descriptors(smiles)

        # Reshape and normalize the descriptors using the scaler
        descriptors_normalized = scaler.transform([descriptors])

        # Use the model to predict the sublimation enthalpy
        prediction = model.predict(descriptors_normalized)

        return prediction[0]
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Get SMILES input from the user
    smiles = get_smiles_input()

    # Predict the sublimation enthalpy
    enthalpy = predict_sublimation_enthalpy(smiles)

    # Display the result
    if enthalpy is not None:
        print(f"Predicted Sublimation Enthalpy: {enthalpy:.2f} kJ/mol")
    else:
        print("Failed to predict sublimation enthalpy.")

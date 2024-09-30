# data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_and_split_data(self):
        # Load the full dataset
        data = pd.read_csv(self.filepath)
        
        # Remove duplicates
        data = data.drop_duplicates(subset=['question'], keep='first')
        
        # Split the data into training (80%) and testing (20%) sets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        
        # Save the train and test datasets
        train_data.to_csv('dataset/train_data.csv', index=False)
        test_data.to_csv('dataset/test_data.csv', index=False)
        
        print("Data has been split and saved successfully.")
        return train_data, test_data

if __name__ == "__main__":
    # Provide the path to your main dataset
    data_loader = DataLoader(filepath='/Users/Kermi/Desktop/financial_chatbot/dataset/financial_chatbot_dataset.csv')
    train_data, test_data = data_loader.load_and_split_data()

# train.py
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertTokenizer
from chatbot_model import FinancialChatbot
from preprocess import DataPreprocessor
from data_loader import DataLoader as DataLoaderUtil
from sklearn.preprocessing import LabelEncoder

class FinancialDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, label_encoder):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.encoded_answers = self.label_encoder.transform(self.answers)  # Encode answers to numeric labels

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.encoded_answers[idx]  # Get encoded answer
        
        inputs = self.tokenizer(question, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'answer': torch.tensor(answer, dtype=torch.long)  # Use long for classification
        }

# train.py
def train_model():
    # Load and preprocess data
    data_loader = DataLoaderUtil(filepath='dataset/financial_chatbot_dataset.csv')
    train_data, _ = data_loader.load_and_split_data()

    preprocessor = DataPreprocessor(filepath='dataset/train_data.csv')
    questions, answers = preprocessor.load_data()

    # Encode answers using LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(answers)  # Fit encoder on all answers
    num_classes = len(label_encoder.classes_)  # Get the total number of classes

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = FinancialDataset(questions, answers, tokenizer, label_encoder)
    train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize the model with num_classes
    model = FinancialChatbot(num_classes=num_classes)
    model.train()

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['answer']

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)  # outputs are raw logits, labels are class indices
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}")

    # Save the trained model and the label encoder
    torch.save(model.state_dict(), 'src/chatbot_model.pth')
    torch.save(label_encoder, 'src/label_encoder.pth')
    print("Training complete! Model and label encoder saved.")

if __name__ == "__main__":
    train_model()

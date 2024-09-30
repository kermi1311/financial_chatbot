# main.py
import torch
from transformers import BertTokenizer
from chatbot_model import FinancialChatbot
from sklearn.preprocessing import LabelEncoder

def load_model_and_encoder():
    # Load the trained model
    num_classes = len(torch.load('/Users/Kermi/Desktop/financial_chatbot/src/label_encoder.pth').classes_)  
    model = FinancialChatbot(num_classes=num_classes)
    model.load_state_dict(torch.load('/Users/Kermi/Desktop/financial_chatbot/src/chatbot_model.pth'))
    model.eval()  
    
    # Load the saved label encoder
    label_encoder = torch.load('/Users/Kermi/Desktop/financial_chatbot/src/label_encoder.pth')
    return model, label_encoder

def predict(model, label_encoder, question):
    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Set `clean_up_tokenization_spaces` parameter in the tokenizer configuration
    tokenizer.clean_up_tokenization_spaces = True
    
    # Preprocess the input question
    inputs = tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predicted_class = torch.argmax(outputs, dim=1).item()  # Get the class with the highest score
    
    # Decode the predicted class back to the original answer
    response = label_encoder.inverse_transform([predicted_class])[0]
    return response

if __name__ == "__main__":
    # Load the trained model and label encoder
    model, label_encoder = load_model_and_encoder()
    
    print("Welcome to the Financial Advisory Chatbot. Type 'exit' to end the session.")
    
    # Chat loop
    while True:
        question = input("You: ")
        if question.lower() == 'exit':
            break
        
        response = predict(model, label_encoder, question)
        print(f"Chatbot: {response}")

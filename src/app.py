
import torch
from transformers import BertTokenizer
from chatbot_model import FinancialChatbot

# Load the trained model
model = FinancialChatbot()
model.load_state_dict(torch.load('/Users/Kermi/Desktop/financial_chatbot/src/chatbot_model.pth'))
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_response(user_input):
    inputs = tokenizer(user_input, return_tensors='pt', truncation=True, padding=True, max_length=128)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        # For simplicity, assuming a dummy response logic here
        response = "Sorry, I'm still learning. Here's what I understood: " + user_input  # Replace with actual logic if needed
    return response

def main():
    print("Welcome to the Financial Advisory Chatbot. Type 'exit' to end the session.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        response = get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()

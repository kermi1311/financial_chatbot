# chatbot_model.py
import torch
import torch.nn as nn
from transformers import BertModel

class FinancialChatbot(nn.Module):
    def __init__(self, num_classes):
        super(FinancialChatbot, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)  

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  
        return self.fc(cls_output)  

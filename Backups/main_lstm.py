import torch
import torch.nn as nn
import torch.optim as optim

import os
import pandas as pd
import numpy as np

from utils import *


'''=================================================================MAIN CODE==========================================================='''
if __name__ == "__main__":
    study_city = 'Shanghai'
    TREATMENT = [study_city]
    # data = pd.read_csv(os.path.join(os.getcwd(), 'df_before_scale.csv'))
    data = pd.read_csv(os.path.join(os.getcwd(), 'df_scaled_train.csv'))
    
    ids = data.index
    controls = [col for col in data.columns if col not in TREATMENT]
    
    Y_pre_c = data.loc[:, controls]
    Y_pre_t = data.loc[:, study_city]
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    class BiLSTMNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers, lstm_input_size=None):
            super(BiLSTMNN, self).__init__()
            
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.num_layers = num_layers
            
            if not lstm_input_size==None:
                self.lstm_input_size = lstm_input_size
                self.fc_in = nn.Linear(input_size, lstm_input_size)
                self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            else:
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3, batch_first=True, bidirectional=True)
                
            # for name, param in self.rnn.named_parameters():
            #     if name.startswith("weight"):
            #         nn.init.xavier_normal_(param)
            #     else:
            #         nn.init.zeros_(param)
            self.fc = nn.Linear(hidden_size*2, output_size)
    
        def forward(self, x, device, input_trans=False):
            # Initialize the hidden and cell states
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
            h0 = h0.to(device)
            c0 = c0.to(device)
            
            if input_trans==True:
                x = self.fc_in(x)
            out, _ = self.lstm(x, (h0, c0))
            out = out[:, -1, :]
            out = self.fc(out)
            return out
    

    input_size = Y_pre_c.shape[1]
    lstm_input_size = None
    num_layers = 1
    hidden_size = 2048
    output_size = 1
    sequence_length = 1
    batch_size = 1500
    epoch_num = 500
    
    # origin_data_y = Y_pre_t.values[ sequence_length-1 : ]
    origin_data_y = Y_pre_t.values
    origin_data_x = []
    
    for idx_, row in Y_pre_c.iterrows():
        pd_ = pd.DataFrame(columns=Y_pre_c.columns)
        pd_.loc[pd_.shape[0]] = row.values
        # if add_feature.shape[0]>0:
        #     pd_ = pd.concat([pd_, add_feature], ignore_index=True)
        origin_data_x.append(pd_.to_numpy())
    
    
    train_x = []
    train_y = []
    for i, step_y in enumerate(origin_data_y):
        train_y.append(step_y)
        
        x_slice = origin_data_x[ i : i+sequence_length ]
        train_x.append(np.array(x_slice))
    
    valid_size = int(np.ceil(len(train_x)*0.1))
    valid_x = train_x[-valid_size: ]
    valid_y = train_y[-valid_size: ]
    train_x = train_x[:-valid_size]
    train_y = train_y[:-valid_size]
    
    train_x = torch.tensor(np.array(train_x), dtype=torch.float32)
    train_y = torch.tensor(np.array(train_y), dtype=torch.float32)
    # valid_x = torch.tensor(np.array(valid_x), dtype=torch.float32)
    # valid_y = torch.tensor(np.array(valid_y), dtype=torch.float32)    
    
    model = BiLSTMNN(input_size, hidden_size, output_size, num_layers, lstm_input_size=None)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    model = model.cuda()
    
    for epoch in range(epoch_num):
        pd_lstm = pd.DataFrame(columns=['pred', 'ytrue'])
        for j in range(0, len(train_x), batch_size):
            
            if j+batch_size>=len(train_x):
                batch_size = len(train_x)-j
            
            inputs = train_x[ j:j+batch_size ].view(batch_size, sequence_length, -1) # direct reshaping and concatenating
            y = train_y[ j:j+batch_size ]
            
            inputs = inputs.cuda()
            y = y.cuda()
            
            model.train()
            outputs = model(inputs, device)
            
            loss = loss_function(outputs.squeeze(), y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            preds_ = outputs.cpu().detach().numpy().squeeze()
            ys = y.cpu().detach().numpy().squeeze()
            
            #output last batch
            pd_lstm = pd.concat([ pd_lstm, pd.DataFrame({ 'pred':preds_, 'ytrue' :ys }) ])
            print('epoch {}, loss {} '.format(epoch, loss))

    
    pd_lstm.to_csv(os.path.join(os.getcwd(), 'lstm.csv'))
    
    model.eval()
    # inputs = train_x.view(train_x.shape[0], sequence_length, -1)
    
    
    # outputs = model(inputs)
    #
    # outputs = outputs.detach().numpy().squeeze()
    # train_y = train_y.detach().numpy().squeeze()
    
    
    
    
    
    













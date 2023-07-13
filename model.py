import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class LinearQNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        self.linear2=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        return x

    def save(self,filename='model.pth'):
        model_path="./model"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        fn=os.path.join(model_path,filename)
        torch.save(self.state_dict,fn)

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.model=model
        self.LR=lr
        self.gamma=gamma
        self.optimizer=optim.Adam(model.parameters(),lr=self.LR)
        self.critereon=nn.MSELoss()

    def train_step(self, state,action,reward,new_state,gg):
        state=torch.tensor(state,dtype=torch.float)
        action=torch.tensor(action,dtype=torch.long)
        new_state=torch.tensor(new_state,dtype=torch.float)
        reward=torch.tensor(reward,dtype=torch.float)


        if (len(state.shape)==1):
            state=torch.unsqueeze(state,0)
            action=torch.unsqueeze(action,0)
            new_state=torch.unsqueeze(new_state,0)
            reward=torch.unsqueeze(reward,0)
            gg=(gg, )


            q=self.model(state)


            target=q.clone()
            for i in range(len(gg)):
                qNew=reward[i]
                if not gg[i]:
                    qNew=reward[i]+self.gamma*torch.max(self.model(new_state[i]))
                target[i][torch.argmax(action).item()]=qNew

            self.optimizer.zero_grad()
            loss=self.critereon(target,q)
            loss.backward()
            self.optimizer.step()



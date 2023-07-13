import torch
import random
import numpy as np
from collections import deque
from game import SnakeAI,NSWE
from model import LinearQNet,QTrainer
import matplotlib.pyplot as plt
from IPython import display



Block=20
MAX_MEM=100_000
Batch_Size=1000
Learning_Rate=0.001

class Agent:
    def __init__(self):
        self.n_games=0
        self.epsilon=0 #makes it random
        self.gamma=0.9 # rate of discount
        self.memory=deque(maxlen=MAX_MEM)
        self.model=LinearQNet(11,256,3)
        self.trainer=QTrainer(self.model,lr=Learning_Rate,gamma=self.gamma)

    def _calcDanger(self,game):
        head = game.body[0]
        point_l = (head[0] - Block, head[1])
        point_r = (head[0] +Block, head[1])
        point_u = (head[0], head[1] - Block)
        point_d = (head[0], head[1] + Block)
        Danger=[0,0,0]
        if ((game.direction==NSWE.East and (game.isCollision(point_r))) or (game.direction==NSWE.West and (game.isCollision(point_l))) or (game.direction==NSWE.North and (game.isCollision(point_d))) or (game.direction==NSWE.South and (game.isCollision(point_u)))):
            Danger[0]=1
        if ((game.direction==NSWE.East and (game.isCollision(point_d))) or (game.direction==NSWE.West and (game.isCollision(point_u))) or (game.direction==NSWE.North and (game.isCollision(point_l))) or (game.direction==NSWE.South and (game.isCollision(point_r)))):
            Danger[1]=1
        if ((game.direction==NSWE.East and (game.isCollision(point_u))) or (game.direction==NSWE.West and (game.isCollision(point_d))) or (game.direction==NSWE.North and (game.isCollision(point_r))) or (game.direction==NSWE.South and (game.isCollision(point_l)))):
            Danger[2]=1
        return Danger

    def _calcFood(self,foodpos,snake):
        lr=foodpos[0]-snake[0]
        ud=foodpos[1]-snake[1]
        food=[0,0,0,0]
        if (lr>0):
            food[1]=1
        elif (lr<0):
            food[0]=1
        if (ud>0):
            food[3]=1
        elif (ud<0):
            food[2]=1
        return food

    def _calcDirection(self,game):
        direction=[0,0,0,0]
        if (game.direction==NSWE.West):
            direction[0]=1
        elif (game.direction==NSWE.East):
            direction[1]=1
        elif (game.direction==NSWE.South):
            direction[2]=1
        else:
            direction[3]=1
        return direction
 
    def getState(self,game):
        Danger=self._calcDanger(game)
        Direction=self._calcDirection(game)
        Food=self._calcFood(game.food,game.snake)
        return np.array(Danger+Direction+Food,dtype=int)
    
    def remember(self,state,action,reward,next_state,gg):
        self.memory.append((state,action,reward,next_state,gg))

    def trainLongMem(self):
        if (len(self.memory)>Batch_Size):
            sample=random.sample(self.memory,Batch_Size)
        else:
            sample=self.memory
        states,actions,rewards,next_states,ggs= zip(*sample)
        self.trainer.train_step(states,actions,rewards,next_states,ggs)

    def trainShortMem(self,state,action,reward,next_state,gg):
        self.trainer.train_step(state,action,reward,next_state,gg)

    def getAction(self,state):
        self.epsilon=80 -self.n_games#Start 30%, lower 1% every 2 games, lowest=3%
        move=[0,0,0]
        r=random.randint(0,200)
        if (r<self.epsilon):
            move[random.randint(0,2)]=1
        else:
            stateTorch=torch.tensor(state,dtype=torch.float)
            prediction=self.model(stateTorch)
            x=torch.argmax(prediction).item()
            move[x]=1
        return move

    def train(self):
        plot_scores=[]
        plot_mean_scores=[]
        total_scores=0
        record=0
        game=SnakeAI()
        while True:
            state=self.getState(game)
            action=self.getAction(state)
            reward,gg,score=game.play_step(action)
            new_state=self.getState(game)
            self.trainShortMem(state,action,reward,new_state,gg)
            self.remember(state,action,reward,new_state,gg)

            if gg:# train long memory
                game.reset()
                self.n_games+=1
                self.trainLongMem()
                if (score>record):
                    record=score
                    self.model.save("bestModel.pth")
                print ("Game: ",self.n_games,"Score: ",score,"Record: ",record,"Reward: ",10*(score-1))
                plot_scores.append(score)
                total_scores+=score
                mean_scores=total_scores/self.n_games
                plot_mean_scores.append(mean_scores)
                plot(plot_scores,plot_mean_scores)
        

def plot(scores,mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Model progress")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1,scores[-1],str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1],str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
if __name__=="__main__":
    plt.ion()
    Agent().train()

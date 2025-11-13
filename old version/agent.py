import torch
import random
import numpy as np
from collections import deque
from game import SnakeAI,NSWE
from model import LinearQNet,QTrainer
import matplotlib.pyplot as plt
from IPython import display
import itertools



Block=20
MAX_MEM=10000000

class Agent:
    def __init__(self,gamma=0.9,starting_epislon_percentage=30,hidden_layers=256,lr=0.001,batch_size=1000, sampler="argmax", early_stopping_threshold=700, add_obsticles_to_state=False):
        self.n_games=0
        self.starting_epislon_percentage=starting_epislon_percentage
        self.epsilon=0 #makes it random
        self.gamma=gamma # rate of discount
        self.hidden_layers=hidden_layers
        self.lr=lr
        self.memory=deque(maxlen=MAX_MEM)
        if (add_obsticles_to_state):
            inputs=15
        else:
            inputs=11
        self.model=LinearQNet(inputs,hidden_layers,3)
        self.batch_size=batch_size
        self.trainer=QTrainer(self.model,lr=lr,gamma=self.gamma)
        self.sampler=sampler
        self.last_n_score=[]
        self.early_stopping_counter=0
        self.early_stopping_threshold=early_stopping_threshold
        self.add_obsticles_to_state=add_obsticles_to_state

    def hyperparams_to_string(self):
        return f"gamma_{self.gamma}#starting-epislon-percentage_{self.starting_epislon_percentage}#hidden-layers_{self.hidden_layers}#lr_{self.lr}#batch-size_{self.batch_size}#sampler_{self.sampler}#early-stopping-threshold_{self.early_stopping_threshold}#add-obsticles-to-state_{self.add_obsticles_to_state}"

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

    def _calcObsticles(self, obsticles,snake):
        obsticles_at_each_direction=[0,0,0,0]

        for obsticle in obsticles:
            lr=obsticle[0]-snake[0]
            ud=obsticle[1]-snake[1]
            if (lr>0):
                obsticles_at_each_direction[1]+=1
            elif (lr<0):
                obsticles_at_each_direction[0]+=1
            if (ud>0):
                obsticles_at_each_direction[3]+=1
            elif (ud<0):
                obsticles_at_each_direction[2]+=1
        return obsticles_at_each_direction

 
    def getState(self,game):
        Danger=self._calcDanger(game)
        Direction=self._calcDirection(game)
        Food=self._calcFood(game.food,game.snake)
        
        if (self.add_obsticles_to_state):
            Obsticles=self._calcObsticles(game.obsticles,game.snake)
            return np.array(Danger+Direction+Food+Obsticles,dtype=int)

        return np.array(Danger+Direction+Food,dtype=int)
    
    def remember(self,state,action,reward,next_state,gg): # Save state, action and reward for the action, along with the next_state and if it is game over
        self.memory.append((state,action,reward,next_state,gg))

    def trainLongMem(self):
        
        last_batch=0
        while (last_batch<len(self.memory)):
            next_batch=min(last_batch+self.batch_size,len(self.memory))
            #sample=self.memory[last_batch:next_batch]
            sample=list(itertools.islice(self.memory, last_batch, next_batch))
            last_batch=next_batch
            states,actions,rewards,next_states,ggs= zip(*sample)
            self.trainer.train_step(states,actions,rewards,next_states,ggs)
            
        #if (len(self.memory)>self.batch_size):
        #    sample=random.sample(self.memory,self.batch_size)
        #else:
        #    sample=self.memory
        #states,actions,rewards,next_states,ggs= zip(*sample)
        #self.trainer.train_step(states,actions,rewards,next_states,ggs)

    def trainShortMem(self,state,action,reward,next_state,gg):
        self.trainer.train_step(state,action,reward,next_state,gg)

    def getAction(self,state):
        self.epsilon=max(2*self.starting_epislon_percentage -self.n_games, int(self.starting_epislon_percentage/10)) #Start 30%, lower 1% every 2 games, lowest=3%
        move=[0,0,0]
        r=random.randint(0,200)
        if (r<self.epsilon):# or r<20):
            move[random.randint(0,2)]=1

        else:
            stateTorch=torch.tensor(state,dtype=torch.float)
            prediction=self.model(stateTorch)
            if (self.sampler=="argmax"):
                x=torch.argmax(prediction,dim=-1).item()
            else:
                x=torch.multinomial(torch.softmax(prediction,dim=-1),1).item()
            move[x]=1
        return move

    def train(self,file=None):
        plot_scores=[]
        plot_mean_scores=[]
        plot_window_mean_scores=[]
        total_scores=0
        record=0
        game=SnakeAI()
        while (self.early_stopping_counter<self.early_stopping_threshold):
            state=self.getState(game)
            action=self.getAction(state)
            reward,gg,score=game.play_step(action)
            new_state=self.getState(game)
            self.trainShortMem(state,action,reward,new_state,gg)
            self.remember(state,action,reward,new_state,gg)

            if gg:# train long memory
                self.early_stopping_counter+=1
                self.n_games+=1
                game.reward_decay_weight=max(1-self.n_games,0)
                game.reset()
                
                self.trainLongMem()
                if (score>record):
                    record=score
                    self.early_stopping_counter=0
                    self.model.save(self.hyperparams_to_string()+"#bestModel.pth")
                print ("Game: ",self.n_games,"Score: ",score,"Record: ",record,"Reward: ",10*(score-1))
                if (file):
                    print ("Game: ",self.n_games,"Score: ",score,"Record: ",record,"Reward: ",10*(score-1),file=file)
                plot_scores.append(score)
                total_scores+=score
                self.last_n_score.append(score)
                if (len(self.last_n_score)>50):
                    self.last_n_score.pop(0)
                window_mean_scores=sum(self.last_n_score)/len(self.last_n_score)
                mean_scores=total_scores/self.n_games
                plot_mean_scores.append(mean_scores)
                plot_window_mean_scores.append(window_mean_scores)
                plot(plot_scores,plot_mean_scores,plot_window_mean_scores)

        plot(plot_scores,plot_mean_scores,plot_window_mean_scores).savefig(self.hyperparams_to_string()+"#plot.png")
        return record
        

def plot(scores,mean_scores,window_mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Model progress")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.plot(window_mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1,scores[-1],str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1],str(mean_scores[-1]))
    plt.text(len(window_mean_scores)-1, window_mean_scores[-1],str(window_mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
    return plt

import time
import os
if __name__=="__main__":

    gammas=[0.9]
    starting_epislon_percentages=[10,60]
    hidden_layers=[128,1024]
    lrs=[0.001]
    batch_sizes=[2000]
    samplers=["argmax"]
    add_obsticles=[True,False]
    plt.ion()



    highest_record=0

    #for gamma in gammas:
    #    for starting_epislon_percentage in starting_epislon_percentages:
    #        for hidden_layer in hidden_layers:
    #            for lr in lrs:
    #                for batch_size in batch_sizes:
    #                    for sampler in samplers:
    #                        for add_obsticle in add_obsticles:
    #                            os.chdir("/home/makis/Documents/Github/SnakeAI/results")
    #                            start_time=time.time()
    #                            agent=Agent(gamma=gamma,starting_epislon_percentage=starting_epislon_percentage,hidden_layers=hidden_layer,lr=lr,batch_size=batch_size, sampler=sampler, add_obsticles_to_state=add_obsticle)
    #                            agent_name=agent.hyperparams_to_string()
    #                            path="/home/makis/Documents/Github/SnakeAI/results/"+agent_name
    #                            print(path)
    #                            os.mkdir(path)
    #                            os.chdir(path)
    #                            with open(agent_name+".log", "w") as f:
    #                                record=agent.train(file=f)
    #                                end_time=time.time()
    #                                print(f"Time taken for {agent.n_games} with High Score of {record}: {end_time-start_time}",file=f)
    #                            if (record>highest_record):
    #                                highest_record=record
    #                                highest_record_holder=agent_name
    #print(f"High Score: {highest_record} from {highest_record_holder}")

    Agent(gamma=0.9, starting_epislon_percentage=10, hidden_layers=128, batch_size=2000, lr=0.001, add_obsticles_to_state=False).train()

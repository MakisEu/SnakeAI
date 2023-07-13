import pygame
import random
import enum
import pdb


pygame.init()

my_font = pygame.font.Font('/home/makis/Documents/Programming/ML/Deep_Learning/SnakeGame/arial.ttf', 25)
Block=20#pixels
Speed= 30

black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
green_light=pygame.Color(0, 225, 0)
blue = pygame.Color(0, 0, 255)



class NSWE(enum.Enum):
    North =1
    South =3
    West  =2
    East  =0
class SnakeAI:
     def __init__(self,width=640,height=480):
         self.width=width
         self.height=height
         #flags=pygame.OPENGL
         self.game_window=pygame.display.set_mode((self.width, self.height))#, flags=flags)
         pygame.display.set_caption("Snake")
         self.time=pygame.time.Clock()
         self.reset()

     def reset(self):
         self.direction=NSWE.East
         self.snake=(self.width/2,self.height/2)
         self.body=[self.snake,(self.snake[0]-Block,self.snake[1]),(self.snake[0]-2*Block,self.snake[1])]
         self.score=0
         self.food=None
         self._place_food()
         self.iter=0

     def _move(self,action):

             direction=self.direction
             if (action[1]==1):
                direction=(direction.value+1)%4
             elif (action[2]):
                 direction=(direction.value-1)%4
                 if (direction<0):
                     direction=3
             else:
                 direction=direction.value
             self.direction=NSWE(direction)
             x=self.snake[0]
             y=self.snake[1]
             if (self.direction==NSWE.East):
                 x+=Block
             if (self.direction==NSWE.West):
                 x-=Block
             if (self.direction==NSWE.North):
                 y+=Block
             if (self.direction==NSWE.South):
                 y-=Block
             self.snake=(x,y)
          

     def play_step(self,action):
         self.iter+=1
         for event in pygame.event.get():
             if (event==pygame.QUIT):
                 pygame.quit()
                 quit()
             if (event.type==pygame.KEYDOWN):
                 if (event.key==pygame.K_LEFT and (self.direction!=NSWE.East)):
                     self.direction=NSWE.West
                 elif (event.key==pygame.K_RIGHT and (self.direction!=NSWE.West)):
                     self.direction=NSWE.East
                 if (event.key==pygame.K_UP and (self.direction!=NSWE.North)):
                     self.direction=NSWE.South
                 if (event.key==pygame.K_DOWN and (self.direction!=NSWE.South)):
                     self.direction=NSWE.North
         self._move(action)
         self.body.insert(0,self.snake)

         reward=0
         gg=self.isCollision()
         if (gg or self.iter>100*len(self.body)):
             gg=True
             reward=-10
             return reward,gg,self.score
         if (self.snake==self.food):
             self.score+=1
             reward=10
             self._place_food()
         else:
            self.body.pop()

         self.updateUI()
         self.time.tick(int(Speed+0.1*len(self.body)))
         return reward,gg,self.score

     def isCollision(self,point=None):
         if point is None:
             point=self.snake
         if (point[0]>self.width-Block or point[0]<0 or point[1]>self.height-Block or point[1]<0):
             return True
         if point in self.body[1:]:
             return True
         return False

     def _place_food(self):
        x=random.randint(0,(self.width-Block)//Block)*Block
        y=random.randint(0,(self.height-Block)//Block)*Block
        self.food=(x,y)
        if self.food in self.body:
            self._place_food()
     def updateUI(self):
        self.game_window.fill(black)
        for pos in self.body:
            pygame.draw.rect(self.game_window,green_light,pygame.Rect(pos[0],pos[1],Block,Block))        
        pygame.draw.rect(self.game_window,green,pygame.Rect(self.snake[0],self.snake[1],Block,Block))
        pygame.draw.rect(self.game_window,red,pygame.Rect(self.food[0],self.food[1],Block,Block))
        text=my_font.render("Score: "+str(self.score),True,white)
        self.game_window.blit(text,[0,0])
        pygame.display.flip()


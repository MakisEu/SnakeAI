import pygame
import random
import enum
import pdb


pygame.init()

my_font = pygame.font.Font('/home/makis/Documents/Programming/ML/Deep_Learning/SnakeGame/arial.ttf', 25)
Block=30#pixels
Speed= 10

black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
green_light=pygame.Color(0, 225, 0)
blue = pygame.Color(0, 0, 255)



class NSWE(enum.Enum):
    North =0
    South =2
    West  =1
    East  =3
class Snake:
     def __init__(self,width=600,height=400):
         self.width=width
         self.height=height
         print (width,height)
         #flags=pygame.OPENGL
         self.game_window=pygame.display.set_mode((self.width, self.height),pygame.RESIZABLE)#, flags=flags)
         pygame.display.set_caption("Snake")
         self.time=pygame.time.Clock()

         self.direction=NSWE.East
         self.snake=(self.width/2,self.height/2)
         self.body=[self.snake,(self.snake[0]-Block,self.snake[1]),(self.snake[0]-2*Block,self.snake[1])]
         self.score=0
         self.food=(-1,-1)
         self._place_food()

     def _move(self,direction):
             x=self.snake[0]
             y=self.snake[1]
             if (direction==NSWE.East):
                 x+=Block
             if (direction==NSWE.West):
                 x-=Block
             if (direction==NSWE.North):
                 y+=Block
             if (direction==NSWE.South):
                 y-=Block
             self.snake=(x,y)
          

     def play_step(self):
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
         self._move(self.direction)
         self.body.insert(0,self.snake)

         gg=self.isCollision()
         if (gg):
             return gg,self.score
         if (self.snake==self.food):
             self.score+=1
             self._place_food()
         else:
            self.body.pop()


         self.updateUI()
         self.time.tick(Speed)#+len(self.body))
         return gg,self.score

     def isCollision(self):
         if (self.snake[0]>self.width-Block or self.snake[0]<0 or self.snake[1]>self.height-Block or self.snake[1]<0):
             return True
         if self.snake in self.body[1:]:
             return True
         return False
         

     def _place_food(self):
        x=random.randint(0,self.width//Block-1)*Block
        y=random.randint(0,self.height//Block-1)*Block
        self.food=(x,y)
        if self.food in self.body:
            self._place_food()
        print (self.food)
     def updateUI(self):
        self.game_window.fill(black)
        for pos in self.body:
            pygame.draw.rect(self.game_window,green_light,pygame.Rect(pos[0],pos[1],Block,Block))        
        pygame.draw.rect(self.game_window,green,pygame.Rect(self.snake[0],self.snake[1],Block,Block))
        pygame.draw.rect(self.game_window,red,pygame.Rect(self.food[0],self.food[1],Block,Block))
        text=my_font.render("Score: "+str(self.score),True,white)
        self.game_window.blit(text,[0,0])
        pygame.display.flip()



if __name__=="__main__":
    game=Snake(pygame.display.Info().current_w//2,pygame.display.Info().current_h//2)
    while (True):
        gg,score=game.play_step()
        #if (score==3):
            #pdb.set_trace()
        if gg:
            break
    print ("Lol you suck bro #ripbozo #packwatch. Score:",score)
    pygame.quit()

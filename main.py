import numpy as np 
import random
random.seed(42)
class Environment:
    def __init__(self,row,column,goal="d"):
        self.row=row
        self.column=column
        self.goal=(self.row-1,self.column-1) if goal=="d" else goal
        self.done=False
        self.state=(0,0)
        
    def reset(self):
        self.grid=np.full((self.row,self.column),"_",dtype="str")
        self.state=(0,0)
        self.done=False
        
    def render(self):
        x,y=self.state
        self.grid[x,y]="A"
        x,y=self.goal
        self.grid[x,y]="G"

        print(self.grid)

    
    def step(self,action):
        x,y=self.state
        self.grid[x,y]="_"
        self.reward=-0.1
        if action==0:
            if x>0:
                x=x-1
           

        elif action==1:
            if self.row-1>x:
                x=x+1
           
        elif action==2:
            if y>0:
                y=y-1
           
        elif action==3:
            if self.column-1>y:
                y=y+1
           
        self.state=(x,y)

        if self.state==self.goal:
            self.reward=10
            self.done=True
            print("yes")
            
        
        return self.state,self.reward,self.done

    def get_size(self):
        return self.state,self.row,self.column

        
class Reinforement_Learning:
    def __init__(self,env,learning_rate=0.1,epsilon=0.9,gamma=0.9):
        self.state,self.row,self.column=env.get_size()
        self.qtable=np.zeros((self.row*self.column,4))
        self.learning_rate=learning_rate
        self.epsilon=epsilon
        self.gamma=gamma
        self.env=env
        
        
    def q_update(self,state,new_state,reward,action):
        x,y=state
        index=(self.column)*x+y
        current_reward=self.qtable[index,action]      
        new_x,new_y=new_state
        index=new_x*self.column+new_y
        future_reward=np.max(self.qtable[index) 
        self.qtable[index,action]=current_reward+self.learning_rate*(reward+self.gamma*future_reward-current_reward)         
        
    def action(self,state):
        if random.uniform(0,1)<self.epsilon:
            action=random.randint(0,3)
        else:
            x,y=state
            index=x*self.column+y
            action=np.argmax(self.qtable[index])
        return action
        

env=Environment(3,4)
rl=Reinforement_Learning(env)
for e in range(1000):
    state=env.reset()
    done=False
    total_reward=0
    state=(0,0)
    while not done:
        action=rl.action(state)
        next_state,reward,done=env.step(action)
        rl.q_update(state,next_state,reward,action)
        state=next_state
        
    if e%100==0:
        print("episode"+str(e))

print(rl.qtable)
    

    
    
        
    
    
    
    
    

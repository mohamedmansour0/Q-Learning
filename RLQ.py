import numpy as np
import cv2
from PIL import Image
#import matplotlib.pyplot as plt
import pickle
#from matplotlib import style
import time

#style.use('ggplot')

#setting parameters

SIZE = 5 #layout size
Total_Iterations = 10 #25000 for training and 100 for validation
Move_Cost = 1 #discount for every move
Failure_Cost = 25 #discount for hitting an obstacle
Extinguishing_Reward = 100 #reward for achieving the fire
epsilon = 0.0 #0.9 at start_q_table = None, smaller value at a file name inserted
Eps_Decay = 0.9998 #the rate by which the epsilon is changed
SHOW_EVERY = 1 #show every how many episodes, 3000 at start_q_table is None, smaller value otherwise
start_q_table = "qtable-1606925723.pickle" #None or file name
Learning_Rate = 0.1
discount = 0.95
episode_rewards = []
steps = 200

successfull = []
failed = []

#setting features

#Making a dictionary to represent the features of the layout as numbers
ROBOT = 1
FIRE = 2
OBSTACLES = 3

#setting the color for each feature, robot is blue, fire is red, obstacle is green
d = {1: (255,0,0), 2: (0,0,255), 3: (0,255,0)}

#Features attributes class

class Blob:
  def __init__ (self): #initializing method
    self.x = np.random.randint(0, SIZE) #initialize a random x location at the grid
    self.y = np.random.randint(0, SIZE) #initialize a random y location at the grid

  def __str__(self): #string method
    return f"{self.x}, {self.y}" #print a string with the blob location

  def __sub__(self, other): #subtractor operator blob - another blob
    return (self.x - other.x, self.y - other.y)

  def action(self, choice): #action method to move the agent diagonally by each choice
    if choice == 0:
      self.move(x=1, y=1)
    elif choice == 1:
      self.move(x=-1, y=-1)
    elif choice == 2:
      self.move(x=-1, y=1)
    elif choice == 3:
      self.move(x=1, y=-1)
    if choice == 4:
      self.move(x=0, y=1)
    elif choice == 5:
      self.move(x=0, y=-1)
    elif choice == 6:
      self.move(x=-1, y=0)
    elif choice == 7:
      self.move(x=1, y=0)
  
  def move(self, x=False, y=False): #move method
    if not x: #move randomly if no value were passed
      self.x += np.random.randint(-1, 2) #randomly between -1, 0, 1
    else: #move specifically if a value of x passed
      self.x += x #move with respect to the x value passed
    if not y: #move randomly if no value were passed
      self.y += np.random.randint(-1, 2) #randomly between -1, 0, 1
    else: #move specifically if a value of y passed
      self.y += y #move with respect to the y value passed
    
    #maintaining the movement within the layout boundaries
    if self.x < 0: 
      self.x = 0 #not to move less than the lower x boundary
    elif self.x > SIZE-1:
      self.x = SIZE-1 #not to move more than the higher x boundary

    if self.y < 0: 
      self.y = 0 #not to move less than the lower y boundary
    elif self.y > SIZE-1:
      self.y = SIZE-1 #not to move more than the higher y boundary

#Retrive the Q table

if start_q_table is None: #in case there's no pre-trained Q table to load
  q_table = {} #create an empty place holder to handle the Q table

  #loop over all the coordinates between the agent and (fire or obstacles)
  for x1 in range(-SIZE+1, SIZE):
    for y1 in range(-SIZE+1, SIZE):
      for x2 in range(-SIZE+1, SIZE):
        for y2 in range(-SIZE+1, SIZE):
            for x3 in range(-SIZE+1, SIZE):
                for y3 in range(-SIZE+1, SIZE):
                    #set a random numbers initaially to all the values in the Q table states
                    q_table[((x1, y1),(x2, y2),(x3, y3))] = [np.random.uniform(-5, 0) for i in range(8)]

else: #in case there's a pre-trained Q table to load
  with open(start_q_table, "rb") as f:
    q_table = pickle.load(f)

#Create the layout

for episode in range(Total_Iterations): #loop a number of episodes
  robot = Blob() #initialize the robot Blob
  fire = Blob() #initialize the fire Blob
  dynamic_obstacle = Blob() #initialize the dynamic_obstacle Blob
  static_obstacle = Blob() #initialize static_obstacle blob

  if episode % SHOW_EVERY == 0: #every SHOW_EVERY episode
    print(f"on # {episode}, epsilon: {epsilon}") #print the episode number and the epsilon value
    print(f"{SHOW_EVERY} ep reward mean {np.mean(episode_rewards[-SHOW_EVERY:])}") #suppose to print the mean of the episodes but idk how this works
    show = True #display setting to true
  else:
    show = False #display setting to false

  episode_reward = 0 #initial reward for every episode
  for i in range(steps):
    obs = (robot-fire, robot-dynamic_obstacle, robot-static_obstacle) #relative observation of the robot from the obstacles and the fire at each step
    if np.random.random() > epsilon: #np.random.random() return a random variable between 0 and 1
      action = np.argmax(q_table[obs]) #favor the values in the q_table
    else: #this case is used often initaially but later on the q_table is more favorable
      action = np.random.randint(0, 8) #choose an arbitrary action
    
    robot.action(action) #move the agent with respect to action decided

    dynamic_obstacle.move() #to move an obstacle randomly

    time.sleep(0.1)

    #setting the reward at special cases

    if robot.x == dynamic_obstacle.x and robot.y == dynamic_obstacle.y:
      #in case the robot landed on the dynamic obstacle
      reward = - Failure_Cost #high negative reward given to the robot

    if robot.x == static_obstacle.x and robot.y == static_obstacle.y:
      #in case the robot landed on the static obstacle
      reward = - Failure_Cost #high negative reward given to the robot

    elif robot.x == fire.x and robot.y == fire.y:
      #in case the robot landed on the fire
      reward = Extinguishing_Reward #high positive reward given to the robot   

    else: #in case the robot is still roaming around
      reward = - Move_Cost #a small negative reward is given to robot inorder to favor the shortest paths

    new_obs = (robot-fire, robot-dynamic_obstacle, robot-static_obstacle) #the new relative observation of the robot from the obstacles and the fire at the step after robot action taken
    max_future_q = np.max(q_table[new_obs]) #get the maximum q value at the new observation max q(Sn+1,An+1)
    current_q = q_table[obs][action] #the current q value of the current state, action pair q(Sn,An)
    
    #once we reached fire we are done
    if reward == Extinguishing_Reward:
      new_q == Extinguishing_Reward

      success = 1
      successfull.append(success)

    #once we hit an obstacle we are done
    elif reward == -Failure_Cost:
      new_q = -Failure_Cost

      fail = 1
      failed.append(fail)

    #update with q_table function regularly while agent is navigating
    else:
      new_q = (1 - Learning_Rate) * current_q + Learning_Rate * (reward + discount * max_future_q)
    
    #place the new q value in its relative place at the q table
    q_table[obs][action] = new_q

    #show the grid enviroment of the selected episodes
    if show:
      env = np.zeros((SIZE,SIZE,3), dtype=np.uint8)
      env[fire.y][fire.x] = d[FIRE]
      env[robot.y][robot.x] = d[ROBOT]
      env[dynamic_obstacle.y][dynamic_obstacle.x] = d[OBSTACLES]
      env[static_obstacle.y][static_obstacle.x] = d[OBSTACLES]

      #show the grid as an image
      img = Image.fromarray(env, "RGB")
      img = img.resize((300,300))
      cv2.imshow("", np.array(img))

      #when we hit an obstacle or reach a target freeze for some time otherwise keep going
      if reward == Extinguishing_Reward or reward == -Failure_Cost:
        if cv2.waitKey(200) & 0xFF == ord("q"):
          break
      else:
        if cv2.waitKey(1) & 0xFF == ord("q"):
          break
    #to see the enviroment
    episode_reward = reward
    if reward == Extinguishing_Reward or reward == - Failure_Cost:
      break
  episode_rewards.append(episode_reward)

  #decay the epsilon
  epsilon *= Eps_Decay

print('total successfull attempts =', len(successfull))
print('total failed attempts =', len(failed))
print('success ratio is ', (len(successfull)/Total_Iterations))

#save the q table
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
  pickle.dump(q_table, f)
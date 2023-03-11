# https://github.com/shankar9522/ENPM661_dijkstra.git
import cv2
import numpy as np
import pandas as pd
import time 

canvas = np.zeros((250, 600, 3), dtype=np.uint8)            # create empty img for visualization
fourcc = cv2.VideoWriter_fourcc(*'XVID')                    # to write output as video
out = cv2.VideoWriter('node_exploration.avi', fourcc, 20.0, (600, 250))

for k in range(5):                          # color the obstacles
    for i in range(canvas.shape[0]):
        for j in range(canvas.shape[1]):
            canvas[i][k] = (0,0,255)
            canvas[k][j] = (0,0,255)
            canvas[canvas.shape[0]-1-k][j] = (0,0,255)
            canvas[i][canvas.shape[1]-1-k] = (0,0,255)

rect1 = np.array([[95,0],[155,0],[155,105],[95,105]], np.int32)
rect2 = np.array([[95,145],[155,145],[155,250],[95,250]], np.int32)
triangle = np.array([[455,20],[515,125],[455,230]], np.int32)

center = (300, 125)
length = 75 + 2*5*np.arctan(np.radians(30))
vertices = []
for i in range(6):
    x = int(center[0] + length * np.cos((i+0.5) * 2 * np.pi / 6))
    y = int(center[1] + length * np.sin((i+0.5) * 2 * np.pi / 6))
    vertices.append((x, y))
hexagon = np.array(vertices)

cv2.fillPoly(canvas, [rect1], (0,0,255))
cv2.fillPoly(canvas, [rect2], (0,0,255))
cv2.fillPoly(canvas, [triangle], (0,0,255))
cv2.fillPoly(canvas, [hexagon], (0,0,255))

b, g, r = cv2.split(canvas)
r=r.T
###################################################### create children #################################################################
def actions(OL_top_node, CL_nodes):

    children = []     #(c2c, parent ID, x,y)

    if(not r[int(OL_top_node[3])-1][int(OL_top_node[4])+1]):  # checks if there is no obstacle
        val = np.where((CL_nodes[:, 3] == OL_top_node[3]-1) & (CL_nodes[:, 4] == OL_top_node[4]+1))[0]
        if(val.size==0):
            children.append([OL_top_node[0]+1.4,OL_top_node[1],OL_top_node[3]-1,OL_top_node[4]+1]) 

    if(not r[int(OL_top_node[3])][int(OL_top_node[4])+1]):
        val = np.where((CL_nodes[:, 3] == OL_top_node[3]) & (CL_nodes[:, 4] == OL_top_node[4]+1))[0]
        if(val.size==0):       
            children.append([OL_top_node[0]+1,OL_top_node[1],OL_top_node[3],OL_top_node[4]+1])

    if(not r[int(OL_top_node[3])+1][int(OL_top_node[4])+1]):
        val = np.where((CL_nodes[:, 3] == OL_top_node[3]+1) & (CL_nodes[:, 4] == OL_top_node[4]+1))[0]
        if(val.size==0):
            children.append([OL_top_node[0]+1.4,OL_top_node[1],OL_top_node[3]+1,OL_top_node[4]+1])

    if(not r[int(OL_top_node[3])+1][int(OL_top_node[4])]):
        val = np.where((CL_nodes[:, 3] == OL_top_node[3]+1) & (CL_nodes[:, 4] == OL_top_node[4]))[0]
        if(val.size==0):
            children.append([OL_top_node[0]+1,OL_top_node[1],OL_top_node[3]+1,OL_top_node[4]])

    if(not r[int(OL_top_node[3])+1][int(OL_top_node[4])-1]):
        val = np.where((CL_nodes[:, 3] == OL_top_node[3]+1) & (CL_nodes[:, 4] == OL_top_node[4]-1))[0]
        if(val.size==0):
            children.append([OL_top_node[0]+1.4,OL_top_node[1],OL_top_node[3]+1,OL_top_node[4]-1])

    if(not r[int(OL_top_node[3])][int(OL_top_node[4])-1]):
        val = np.where((CL_nodes[:, 3] == OL_top_node[3]) & (CL_nodes[:, 4] == OL_top_node[4]-1))[0]
        if(val.size==0):
            children.append([OL_top_node[0]+1,OL_top_node[1],OL_top_node[3],OL_top_node[4]-1])

    if(not r[int(OL_top_node[3])-1][int(OL_top_node[4])-1]):
        val = np.where((CL_nodes[:, 3] == OL_top_node[3]-1) & (CL_nodes[:, 4] == OL_top_node[4]-1))[0]
        if(val.size==0):
            children.append([OL_top_node[0]+1.4,OL_top_node[1],OL_top_node[3]-1,OL_top_node[4]-1])

    if(not r[int(OL_top_node[3])-1][int(OL_top_node[4])]):
        val = np.where((CL_nodes[:, 3] == OL_top_node[3]-1) & (CL_nodes[:, 4] == OL_top_node[4]))[0]
        if(val.size==0):
            children.append([OL_top_node[0]+1,OL_top_node[1],OL_top_node[3]-1,OL_top_node[4]])

    return children
#################################################### main ####################################
correct_input = False
while(not correct_input):           # take input from user until correct input is entered
    initial_x_config = int(input('enter starting x coordinate: '))
    initial_y_config = int(input('enter starting y coordinate: '))
    final_x_config = int(input('enter final x coordinate: '))
    final_y_config = int(input('enter final y coordinate: '))

    if(initial_x_config > canvas.shape[1] or initial_y_config > canvas.shape[0] or final_x_config > canvas.shape[1] or final_y_config > canvas.shape[0]):
        print('invalid input')
    elif(r[initial_x_config][initial_y_config]==255 or r[final_x_config][final_y_config]==255):
        print('invalid input')
    else:
        correct_input=True
print('----------------------------')
print('calculating shortest path.....')

OL_nodes = np.array([[0,1,0,initial_x_config,initial_y_config]])        # open list
CL_nodes = np.array([[-1,-1,-1,-1,-1]])                                 # close list
NodeID = 1                                                              # initialize node id
################################################################## Start loop#######################
start_time = time.time()
while( (not(CL_nodes[-1][3]==final_x_config and CL_nodes[-1][4]==final_y_config)) and (not OL_nodes.shape[0]==0)): # run loop until goal is reached or the open list becomes empty

    OL_nodes = OL_nodes[OL_nodes[:,0].argsort()]        # arrange open list according to cost to come value
    children = actions(OL_nodes[0],CL_nodes)            # calls function to generate children 

    for i in range(len(children)): 
        val = np.where((OL_nodes[:, 3] == children[i][2]) & (OL_nodes[:, 4] == children[i][3]))[0]  # checks if child is present in open list
        if(val.size>0):
            if (children[i][0] < OL_nodes[int(val)][0]):  # checks if the child has a lower cost to come than earlier
                    OL_nodes[int(val)][0] = children[i][0]     # update cost to come
                    OL_nodes[int(val)][2] = children[i][1]     # update the parent
        else:
                OL_nodes = np.vstack([OL_nodes, [children[i][0],NodeID+1,children[i][1],children[i][2],children[i][3]]])   # add the child to open list
                NodeID +=1

    CL_nodes = np.vstack([CL_nodes, OL_nodes[0]])   # pops the lowest cost to come element from open list and add it to closed list
    OL_nodes = np.delete(OL_nodes, 0, axis=0)

print('--------------------------')
print('execution time: ' + str(time.time() - start_time) + ' sec') 
########################################## backtrack #########################################
backtrack = np.array([[final_x_config,final_y_config]])         # initialize backtrack 
val = np.where((CL_nodes[:, 3] == final_x_config) & (CL_nodes[:, 4] == final_y_config))[0]      #checks for the goal node parent
parent = CL_nodes[int(val)][2]

while(parent):
    val = np.where(CL_nodes[:, 1] == parent)[0]
    backtrack = np.vstack([backtrack, [CL_nodes[int(val)][3],CL_nodes[int(val)][4]]])
    parent = CL_nodes[int(val)][2]

backtrack = np.flip(backtrack,axis = 0)
backtrack = backtrack.astype(int)
#################### create video for visualization and output backtrack in excel ##########################
print('-------------------------')
print('writing output as video and excel...........')
cv2.circle(canvas, (final_x_config,canvas.shape[0]-final_y_config), 1, (255, 255, 255), 2)
cv2.circle(canvas, (initial_x_config,canvas.shape[0]-initial_y_config), 1, (255, 255, 255), 2)

for i in range(1,CL_nodes.shape[0]):
    canvas[canvas.shape[0]-int(CL_nodes[i][4])][int(CL_nodes[i][3])][0]=255
    canvas[canvas.shape[0]-int(CL_nodes[i][4])][int(CL_nodes[i][3])][1]=255
    out.write(canvas)

for i in range(backtrack.shape[0]):
    cv2.circle(canvas, (int(backtrack[i][0]),canvas.shape[0]-int(backtrack[i][1])), 1, (0, 200, 0), 1)
    out.write(canvas)
out.release()

df = pd.DataFrame(backtrack)
df.to_excel(excel_writer = "backtrack.xlsx")
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxXXXX






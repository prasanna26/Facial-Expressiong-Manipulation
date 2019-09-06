import numpy as np
import math
#import scipy
import cv2 
from matplotlib import pyplot as plt
#import argparse

chosen_points = []

x1=[]
x_=[]
y1=[]
y_=[]
#USE THE MOUSECLICK EVENTS TO SELECT AND DISPLAY THE SELECTED LANDMARKS
x1=[128,186,147,166,154]
y1=[206,207,207,208,220]
x_=[121,194,149,160,154]
y_=[212,216,208,209,219]

# x1=[132,171,152,152]
# y1=[174,173,182,171]
# x_=[125,178,152,152]
# y_=[171,171,180,172]
def select_points(event, x, y, flags, param):

    global chosen_points
 
    #when mouse button is clicked first
    if event == cv2.EVENT_LBUTTONDOWN:
        print('selected point in the source image')
        print(x,y)
        x1.append(x)
        y1.append(y)
        chosen_points = [(x, y)]
        
 
    #when mouse button is released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        chosen_points.append((x, y))
        #print(chosen_points)

        cv2.rectangle(img1, chosen_points[0], chosen_points[1], (0, 255, 0), 2)
        #print("here")
        cv2.imshow("image", img1)
        if(len(x1)==3):
            dest="source_3_landmarks.jpg"
        else:
            dest="source_overconstrained.jpg"
        cv2.imwrite(dest,img1)
#GET POINTS IN THE FIRST IMAGE VIA MOUSE CLICK ACTIONS
def get_point(image):
    
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", select_points)
    cv2.imshow("image", image)
    #wait till a key is pressed, till then keep storing the landmark points
    cv2.waitKey(0)

#USE THE MOUSECLICK EVENTS TO SELECT AND DISPLAY THE SELECTED LANDMARKS IN THE TARGET IMAGE
def select_points2(event, x, y, flags, param):

    global chosen_points
 
    #when mouse button is clicked first
    if event == cv2.EVENT_LBUTTONDOWN:
        print('selected point in the target image')
        print(x,y)
        x_.append(x)
        y_.append(y)
        chosen_points = [(x, y)]
        
 
    # when mouse button is released
    elif event == cv2.EVENT_LBUTTONUP:
        
        chosen_points.append((x, y))
        #print(chosen_points)
        cv2.rectangle(img2, chosen_points[0], chosen_points[1], (0, 255, 0), 2)
        #print("here")
        cv2.imshow("image2", img2)

        if(len(x1)==3):
            dest="target_3_landmarks.jpg"
        else:
            dest="target_overconstrained.jpg"
        cv2.imwrite(dest,img2)

#GET POINTS IN THE FIRST IMAGE VIA MOUSE CLICK ACTIONS
def get_point2(image):
    
    
    cv2.namedWindow("image2")
    cv2.setMouseCallback("image2", select_points2)
    cv2.imshow("image2", image)
    #wait till a key is pressed, till then keep storing the landmark points
    cv2.waitKey(0)


def get_landmarks(image):
    X_s=[]
    Y_s=[]
    
    for i in range(0,len(x1)):
        temp_row_1=[x1[i]]
        temp_row_2=[x_[i]]
        
        X_s.append(temp_row_1)
        Y_s.append(temp_row_2)

    for i in range(0,len(x1)):

        temp_row_1=[y1[i]]
        temp_row_2=[y_[i]]
        
        X_s.append(temp_row_1)
        Y_s.append(temp_row_2)
    
    Source_points=np.asarray(X_s)
    Target_points=np.asarray(Y_s)
    print(Source_points)
    print('target is '+str(Target_points))

    B=np.zeros((2*len(x1),2*len(x1)))
    dr=image.shape[0]+image.shape[1]
    n=(len(x1)*2)-1
    for i in range(0,len(x1)):
        for j in range(0,len(x1)):
            #if(i==j):
            #    B[i][j]=1.
            #else:
            B[i][j]=phi2(Source_points[i][0],Source_points[j][0],Source_points[i+len(x1)][0],Source_points[j+len(x1)][0],dr)
            B[n-i][n-j]=B[i][j]

    print('b is '+str(B))

    #Calculate the K parameters
    K= np.matmul(np.linalg.pinv(B),(Target_points-Source_points))
    K=-1*K
    #print("K is "+str(K))
    return K
    #transform(image,K,Target_points)

def transform4(image,Affine):
    temp=image.copy()
    print('x_ is '+str(x_))
    print('y_ is '+str(y_))
    #temp=np.zeros((image.shape[0],image.shape[1],image.shape[2]))
    for i in range(0,image.shape[1]):

        phi_x=phi(i,x_[0],image.shape[1])
        if(phi_x==1):
            print("phi_x is 0 at "+str(i))
        if(i==x_[0]):
            print("phi_x at "+str(i)+" is "+str(phi_x))
        #phi_x_2=phi()
        #delta_x=phi_x*kx
        for j in range(0,image.shape[0]):
            X_=np.array([[i],[j],[1]])
            phi_y=phi(j,y_[0],image.shape[0])
            
            #if(j==y_[0]):
            #    print("phi_y at "+str(j)+" is "+str(phi_y))
            if(i==x_[0] and j==y_[0]):
                print('holallallalll')
                print('phix and phi_y at ('+str(i)+","+str(j)+" is "+ str(phi_x)+" and"+str(phi_y))
            phil=0
            delta_x_=0
            delta_y_=0
            for k in range(0,len(x1)):
                phil=phi2(i,x_[k],j,y_[k],image.shape[0]+image.shape[1])
                delta_x_+=(phil*Affine[k][0])
                delta_y_+=(phil*Affine[k+len(x1)][0])
            #print('delta x is'+str(delta_x_))
            #phil1=phi2(i,x0_bar,j,y0_bar,image.shape[0]+image.shape[1])
            #phil2=phi2(i,x1_bar,j,y1_bar,image.shape[0]+image.shape[1])
            
            #delta_x_1=phil1*k1x
            #delta_x_2=phil2*k2x
            #delta_x=delta_x_1+delta_x_2
            #print('delta here ')
            #delta_y_1=phil1*k1y
            #delta_y_2=phil2*k2y
            #delta_y=delta_y_1+delta_y_2
            A=np.array([[1,0,delta_x_],[0,1,delta_y_],[0,0,1]])
            #A=np.array([[1,0,delta],[0,1,delta],[0,0,1]])
            result=np.matmul(A,X_)
            #result=np.matmul(np.linalg.inv(A),X_)
            temp_x=result[0][0]
            temp_y=result[1][0]
            if(phi_x==1 and phi_y==1):
                #print('delta here is'+str(delta))
                print('x and y is'+str(i)+', '+str(j))
                print('source image point is '+str(temp_x)+", "+str(temp_y))
            x1a=math.floor(temp_x)
            x2a=math.ceil(temp_x)
            y1a=math.floor(temp_y)
            y2a=math.ceil(temp_y)


            if(x1a>=0 and y1a>=0):
                if(x2a<image.shape[1] and y2a<image.shape[0]):
                    q11=image[y1a][x1a]

                    if(phi_x==1 and phi_y==1):
                        print('here'+str(q11))
                        print('here'+str(q22))
                        
                    q21=image[y2a][x1a]
                    q12=image[y1a][x2a]
                    q22=image[y2a][x2a]

                    if(temp_x==x1a):
                        R1=q11
                        R2=q22
                    else:
                        R1=(((x2a-temp_x)*q11)+((temp_x-x1a)*q21))/(x2a-x1a)
                        R2=(((x2a-temp_x)*q12)+((temp_x-x1a)*q22))/(x2a-x1a)

                    if(temp_y==y1a):
                        P=R1
                    else:
                        P=(((y2a-temp_y)*R1)+((temp_y-y1a)*R2))/(y2a-y1a)

                    P=P.astype(int)
                    #print(P)
                    #print(P[0])
                    if(phi_x==1 and phi_y==1):
                        print('p here is'+str(P))
                    temp[j][i]=P

                    if(phi_x==1 and phi_y==1):
                        print('temp[j][i]'+str(temp[y_[0]][x_[0]]))
                

                    #temp=temp.astype(int)
                    #print(temp[i][j])
                else:
                    temp[j][i]=[0,0,0]
            else:
                temp[j][i]=[0,0,0]


            

    #temp=temp.astype(int)
    #print('temp shape is'+str(temp.shape))
    #print(temp)
    #plt.imshow(temp,cmap='gray')
    '''
    plt.imshow(temp)
    plt.show()
    '''
    cv2.namedWindow("result2")
    cv2.imshow("result2",temp)
    if(len(x1)==3):
        dest="result_3_landmarks.jpg"
    else:
        dest="result_overconstrained.jpg"
    
    cv2.imwrite(dest,temp)
    cv2.waitKey(0)
    
    return temp


def transform_now(image,Affine):
    temp=image.copy()

    for i in range(0,image.shape[1]):
        phi_x=phi(i,x_[0],image.shape[1])

        for j in range(0,image.shape[0]):

            X_=np.array([[i],[j],[1]])

            for k in range(0,len(x1)):
                phil=phi2(i,x_[k],j,y_[k],image.shape[0]+image.shape[1])
                if(phil==1):
                    print('****')
                    print('i and j is '+str(i)+', '+str(j))

            phi_y=phi(j,y_[0],image.shape[0])
            if(i==x_[0] and j==y_[0]):
                print('holalalal')
            if(phi_x==1 and phi_y==1):
                print('i and j is '+str(i)+', '+str(j))


def phi(a,b,d):
    return math.exp(-((math.pow((a-b),2))/math.pow((d/12),2)))

#Multi Quadratic Radial Basis Function
def phi2(a,b,c,d,e):

    par=48
    #return math.exp(-((math.pow((a-b),2)+math.pow((c-d),2))/math.pow((e/48),2)))
    
    return (math.pow((e/par),2))/(math.pow((e/par),2)+(math.pow((a-b),2)+math.pow((c-d),2)))
    #return (math.pow((a-b),2)+math.pow((c-d),2))*math.log((math.pow((math.pow((a-b),2)+math.pow((c-d),2)),0.5)),2)

#Inverse Quadric Radial Basis
# def phi2(a,b,c,d,e):

#     par=12
#     inv = (math.pow((e/par),2))/(math.pow((e/par),2)+(math.pow((a-b),2)+math.pow((c-d),2)))

#     quad = 1/inv
#     return math.pow(quad, 0.5)




#GAUSSIAN RADIAL BASIS 
# def phi2(a,b,c,d,e):
#     par=12
#     #print('sigma is: '+str(e/par))
#     return math.exp(-((math.pow((a-b),2)+math.pow((c-d),2))/math.pow((e/par),2)))


def phi_calc(inp1,inp2,denominator):
    a=inp1[0][0]
    b=inp2[0][0]
    c=inp1[1][0]
    d=inp1[1][0]
    #print(a,b,c,d)
    return phi2(a,b,c,d,denominator)

def k_calc(source,target,sym):

    ans= np.matmul(np.linalg.pinv(sym),(target-source))
    return ans

#-----------------------------------------
'''
img1=cv2.imread("a1.jpeg",0)

print(img1.shape)
get_point(img1)

img2=cv2.imread("a1.jpeg",0)
get_point2(img2)

print(img2.shape)
'''
img1=cv2.imread("source_gb.jpeg")
# #img1=cv2.imread("grid.jpeg")

# print(img1.shape)
# get_point(img1)

# img2=cv2.imread("grid.jpeg")
# #img2=cv2.imread("grid.jpeg")
# get_point2(img2)

# print(img2.shape)

mat=get_landmarks(img1)
print(mat)

#img1=cv2.imread("source_man_grid.jpeg")
transform4(img1,mat)
#transform4(img,mat)
#transform_now(img1,mat)
'''
x0=x1[0]
x2=x1[1]
y0=y1[0]
y2=y1[1]
x0_bar=x_[0]
x1_bar=x_[1]
y0_bar=y_[0]
y1_bar=y_[1]

Target_points=np.array([[x0_bar],[x1_bar],[y0_bar],[y1_bar]])
Source_points=np.array([[x0],[x2],[y0],[y2]])
print(Target_points)
print(Source_points)
'''
#print('test print')
#print(img1[y0][x0])
#kx=x0_bar-x0
#ky=y0_bar-y0
#kx=x0-x0_bar
#img=cv2.imread("t2.jpeg")
#transform2(img1)
#plt.imshow(img1)
#plt.show()
#print(x0,y0,x0_bar,y0_bar=

#print(phi(2,2,10))

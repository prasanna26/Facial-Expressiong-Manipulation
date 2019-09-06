import numpy as np
import math
#import scipy
import cv2 
import sys
from matplotlib import pyplot as plt
#import argparse

chosen_points = []

x1=[]
x_=[]
y1=[]
y_=[]
#USE THE MOUSECLICK EVENTS TO SELECT AND DISPLAY THE SELECTED LANDMARKS
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
        cv2.imshow("Source", img1)
        if(len(x1)==3):
            dest="source_landmarks_all_in_one.jpeg"
        else:
            dest="source_landmarks_all_in_one.jpeg"
        cv2.imwrite(dest,img1)
#GET POINTS IN THE FIRST IMAGE VIA MOUSE CLICK ACTIONS
def get_point(image):
    
    cv2.namedWindow("Source")
    cv2.setMouseCallback("Source", select_points)
    cv2.imshow("Source", image)
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
        cv2.imshow("Target", img2)

        if(len(x1)==3):
            dest="target_landmarks_all_in_one.jpeg"
        else:
            dest="target_landmarks_all_in_one.jpeg"
        cv2.imwrite(dest,img2)

#GET POINTS IN THE FIRST IMAGE VIA MOUSE CLICK ACTIONS
def get_point2(image):
    
    
    cv2.namedWindow("Target")
    cv2.setMouseCallback("Target", select_points2)
    cv2.imshow("Target", image)
    #wait till a key is pressed, till then keep storing the landmark points
    cv2.waitKey(0)

'''
def transform3(image):
    temp=image.copy()
    #temp=np.zeros((image.shape[0],image.shape[1],image.shape[2]))
    for i in range(0,image.shape[1]):
        phi_x=phi(i,x0_bar,image.shape[1])
        #phi_x_2=phi()
        #delta_x=phi_x*kx
        for j in range(0,image.shape[0]):
            X_=np.array([[i],[j],[1]])
            phi_y=phi(j,y0_bar,image.shape[0])
            phil1=phi2(i,x0_bar,j,y0_bar,image.shape[0]+image.shape[1])
            phil2=phi2(i,x1_bar,j,y1_bar,image.shape[0]+image.shape[1])
            
            delta_x_1=phil1*k1x
            delta_x_2=phil2*k2x
            delta_x=delta_x_1+delta_x_2
            #print('delta here ')
            delta_y_1=phil1*k1y
            delta_y_2=phil2*k2y
            delta_y=delta_y_1+delta_y_2
            A=np.array([[1,0,delta_x],[0,1,delta_y],[0,0,1]])
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
    
    plt.imshow(temp)
    plt.show()
    
    cv2.namedWindow("result2")
    cv2.imshow("result2",temp)
    if(len(x1)==3):
        dest="result_3_landmarks.jpg"
    else:
        dest="result_overconstrained.jpg"
    
    cv2.imwrite(dest,temp)
    cv2.waitKey(0)
    
    return temp

'''
#CONSTRUCT THE PARAMETERS OF THE TRANSFORMATION
def get_landmarks(image,number):
    X_s=[]
    Y_s=[]
    if(len(x1)!=len(x_)):
        sys.exit("Exit : Equal landmarks not selected in source and target image")
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
    #print(Source_points)
    #print('target is '+str(Target_points))

    B=np.zeros((2*len(x1),2*len(x1)))
    dr=image.shape[0]+image.shape[1]
    n=(len(x1)*2)-1
    for i in range(0,len(x1)):
        for j in range(0,len(x1)):
            #if(i==j):
            #    B[i][j]=1.
            #else:
            if(number==1):
                B[i][j]=phi1(Source_points[i][0],Source_points[j][0],Source_points[i+len(x1)][0],Source_points[j+len(x1)][0],dr)
            elif(number==2):
                B[i][j]=phi2(Source_points[i][0],Source_points[j][0],Source_points[i+len(x1)][0],Source_points[j+len(x1)][0],dr)
            elif(number==3):
                B[i][j]=phi3(Source_points[i][0],Source_points[j][0],Source_points[i+len(x1)][0],Source_points[j+len(x1)][0],dr)
            B[n-i][n-j]=B[i][j]

    #print('b is '+str(B))

    #Calculate the K parameters
    K= np.matmul(np.linalg.pinv(B),(Target_points-Source_points))
    K=-1*K
    #print("K is "+str(K))
    return K
    #transform(image,K,Target_points)

#INVERSE MAPPING TO PERFORM IMAGE TRANSFORMATION, BILINEAR INTERPOLATION IS USED 
def transform4(image,Affine,number):
    temp=image.copy()
    # print('x is '+str(x1))
    
    # print('x_ is '+str(x_))
    # print('y is '+str(y1))
    # print('y_ is '+str(y_))
    #temp=np.zeros((image.shape[0],image.shape[1],image.shape[2]))
    for i in range(0,image.shape[1]):

        phi_x=phi(i,x_[0],image.shape[1])
        # if(phi_x==1):
        #     print("phi_x is 0 at "+str(i))
        # if(i==x_[0]):
        #     print("phi_x at "+str(i)+" is "+str(phi_x))
        #phi_x_2=phi()
        #delta_x=phi_x*kx
        for j in range(0,image.shape[0]):
            X_=np.array([[i],[j],[1]])
            phi_y=phi(j,y_[0],image.shape[0])
            
            #if(j==y_[0]):
            #    print("phi_y at "+str(j)+" is "+str(phi_y))
            # if(i==x_[0] and j==y_[0]):
            #     #print('holallallalll')
            #     print('phix and phi_y at ('+str(i)+","+str(j)+" is "+ str(phi_x)+" and"+str(phi_y))
            phil=0
            delta_x_=0
            delta_y_=0
            for k in range(0,len(x1)):
                if(number==1):
                    phil=phi1(i,x_[k],j,y_[k],image.shape[0]+image.shape[1])
                elif(number==2):
                    phil=phi2(i,x_[k],j,y_[k],image.shape[0]+image.shape[1])
                elif(number==3):
                    phil=phi3(i,x_[k],j,y_[k],image.shape[0]+image.shape[1])
                    
                    
                delta_x_+=(phil*Affine[k][0])
                delta_y_+=(phil*Affine[k+len(x1)][0])
            
            A=np.array([[1,0,delta_x_],[0,1,delta_y_],[0,0,1]])
            #A=np.array([[1,0,delta],[0,1,delta],[0,0,1]])
            result=np.matmul(A,X_)
            #result=np.matmul(np.linalg.inv(A),X_)
            temp_x=result[0][0]
            temp_y=result[1][0]
            #if(phi_x==1 and phi_y==1):
                #print('delta here is'+str(delta))
                #print('x and y is'+str(i)+', '+str(j))
                #print('source image point is '+str(temp_x)+", "+str(temp_y))
            x1a=math.floor(temp_x)
            x2a=math.ceil(temp_x)
            y1a=math.floor(temp_y)
            y2a=math.ceil(temp_y)


            if(x1a>=0 and y1a>=0):
                if(x2a<image.shape[1] and y2a<image.shape[0]):
                    q11=image[y1a][x1a]

                    #if(phi_x==1 and phi_y==1):
                        #print('here'+str(q11))
                        #print('here'+str(q22))
                        
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
                    # if(phi_x==1 and phi_y==1):
                    #     print('p here is'+str(P))
                    temp[j][i]=P

                    # if(phi_x==1 and phi_y==1):
                    #     print('temp[j][i]'+str(temp[y_[0]][x_[0]]))
                

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
    if(number==1):
        dest="Result_after_warp_quadratic.jpeg"
    elif(number==2):
        dest="Result_after_warp_inverse_multi_quadric.jpeg"
    elif(number==3):
        dest="Result_after_warp_gaussian.jpeg"
    cv2.namedWindow(dest)
    cv2.imshow(dest,temp)
    
    
    
    
    cv2.imwrite(dest,temp)
    cv2.waitKey(0)
    
    return temp



#---------------------------POTENTIAL FUNCTION DEFINITIONS------------------------
def phi(a,b,d):
    return math.exp(-((math.pow((a-b),2))/math.pow((d/12),2)))


#CHANGE THE VALUE OF PAR VARIABLE TO ALTER SIGMA

#Quadratic Radial Basis Function
def phi1(a,b,c,d,e):
    #this value par can be changed to alter the value of sigma, for par=48, sigma=12.5
    #for par =24, sigma=25 and par=12, sigma=50
    par=48
    #return math.exp(-((math.pow((a-b),2)+math.pow((c-d),2))/math.pow((e/48),2)))
    
    return (math.pow((e/par),2))/(math.pow((e/par),2)+(math.pow((a-b),2)+math.pow((c-d),2)))
    #return (math.pow((a-b),2)+math.pow((c-d),2))*math.log((math.pow((math.pow((a-b),2)+math.pow((c-d),2)),0.5)),2)

#Inverse Multi Quadric Radial Basis
def phi2(a,b,c,d,e):
    #this value par can be changed to alter the value of sigma, for par=48, sigma=12.5
    #for par =24, sigma=25 and par=12, sigma=50

    par=48
    inv = (math.pow((e/par),2))/(math.pow((e/par),2)+(math.pow((a-b),2)+math.pow((c-d),2)))

    quad = 1/inv
    return math.pow(quad, 0.5)




#GAUSSIAN RADIAL BASIS 
def phi3(a,b,c,d,e):

    #this value par can be changed to alter the value of sigma, for par=48, sigma=12.5
    #for par =24, sigma=25 and par=12, sigma=50
    par=48
    #print('sigma is: '+str(e/par))
    return math.exp(-((math.pow((a-b),2)+math.pow((c-d),2))/math.pow((e/par),2)))
#--------------------------------END OF POTENTIAL FUNCTIONS----------------------

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

#--------------------------START OF PROGRAM-----------------------
'''
img1=cv2.imread("a1.jpeg",0)

print(img1.shape)
get_point(img1)

img2=cv2.imread("a1.jpeg",0)
get_point2(img2)

print(img2.shape)
'''
print('Select the corresponding landmarks in the source and target')
img1=cv2.imread("a1_grid.jpeg")
#img1=cv2.imread("grid.jpeg")

#print(img1.shape)
get_point(img1)

img2=cv2.imread("a1_grid.jpeg")
#img2=cv2.imread("grid.jpeg")
get_point2(img2)

#print(img2.shape)

for i in range(1,4):
    mat=get_landmarks(img1,i)
    #print(mat)
    img=cv2.imread("a1_grid.jpeg")
    transform4(img1,mat,i)
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

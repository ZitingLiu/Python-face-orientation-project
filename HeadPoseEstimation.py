import cv2 #OpenCV:Image processing library
import dlib #Machine learning library
import imutils #OpenCV assistance
from imutils import face_utils
import numpy as np
import glob
from PIL import Image

#Gets a VideoCapture object
DEVICE_ID = 0 #ID 0 is standard web cam
capture = cv2.VideoCapture('C:/Users/liuzi/Desktop/2022S/5642DataVis/Proj/clip2.mp4')#Read dlib trained data
if (capture.isOpened()== False):
    print("Error opening video stream or file")

predictor_path = "C:/Users/liuzi/Desktop/2022S/5642DataVis/Proj/shape_predictor_68_face_landmarks.dat"

file=open("C:/Users/liuzi/Desktop/2022S/5642DataVis/Proj/angle.txt","w")


detector = dlib.get_frontal_face_detector() #Call the face detector. Only the face is detected.
predictor = dlib.shape_predictor(predictor_path) #Output landmarks such as eyes and nose from the face
count=1
counter=0
while(True): #Get images continuously from the camera
    ret, frame = capture.read() #Capture from the camera and put one frame of image data in the frame
    counter=counter+1
    if ret==False:
        break
    frame = imutils.resize(frame, width=942) #Adjust the display size of the frame image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert to gray scale
    rects = detector(gray, 0) #Detect face from gray
    image_points = None
     
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #for (x, y) in shape: #Plot 68 landmarks on the entire face
            #cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        #print(shape[0])
        

        image_points = np.array([
                tuple(shape[30]),#Nose tip
                tuple(shape[21]),
                tuple(shape[22]),
                tuple(shape[39]),
                tuple(shape[42]),
                tuple(shape[31]),
                tuple(shape[35]),
                tuple(shape[48]),
                tuple(shape[54]),
                tuple(shape[57]),
                tuple(shape[8]),
                ],dtype='double')
    
    if len(rects) > 0:
        
        model_points = np.array([
                (0.0,0.0,0.0), # 30
                (-30.0,-125.0,-30.0), # 21
                (30.0,-125.0,-30.0), # 22
                (-60.0,-70.0,-60.0), # 39
                (60.0,-70.0,-60.0), # 42
                (-40.0,40.0,-50.0), # 31
                (40.0,40.0,-50.0), # 35
                (-70.0,130.0,-100.0), # 48
                (70.0,130.0,-100.0), # 54
                (0.0,158.0,-10.0), # 57
                (0.0,250.0,-50.0) # 8
                ])

        size = frame.shape

        focal_length = size[1]
        center = (size[1] // 2, size[0] // 2) #Face center coordinates

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype='double')

        dist_coeffs = np.zeros((4, 1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        #Rotation matrix and Jacobian
        (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
        mat = np.hstack((rotation_matrix, translation_vector))

        #yaw,pitch,Take out roll
        (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
        yaw = eulerAngles[1]
        pitch = eulerAngles[0]
        roll = eulerAngles[2]
        ang=max(abs(yaw),abs(pitch))
        print("Angle:",int(ang))#Extraction of head posture data
        if counter%15==0:
            file.write(str(int(ang[0]))+"\n")

        if ang<10:
            color=[0,255,0]
        elif ang>=10 and ang<20:
            color=[90,205,162]
        elif ang>=20 and ang<30:
            color=[0, 215, 255]
        elif ang>=30 and ang<45:
            color=[0, 127, 255]
        else :
            color=[0,0,255]
        cv2.putText(frame, 'Angle : ' + str(int(ang)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        #cv2.putText(frame, 'pitch : ' + str(int(pitch)), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        #cv2.putText(frame, 'roll : ' + str(int(roll)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)
        #Plot of points used in the calculation/Display of face direction vector
        #for p in image_points:
            #cv2.drawMarker(frame, (int(p[0]), int(p[1])),  (0.0, 1.409845, 255),markerType=cv2.MARKER_CROSS, thickness=1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.arrowedLine(frame, p1, p2, color, 2)

        cv2.line(frame,shape[0],shape[17],color,2)
        cv2.line(frame,shape[17],shape[26],color,2)
        cv2.line(frame,shape[26],shape[16],color,2)
        cv2.line(frame,shape[16],shape[11],color,2)
        cv2.line(frame,shape[11],shape[8],color,2)
        cv2.line(frame,shape[8],shape[5],color,2)
        cv2.line(frame,shape[0],shape[5],color,2)
    else:
        if counter%15==0:
            file.write("90\n")
        print("face not detected")
        cv2.putText(frame, 'Face not detected', (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    #cv2.line(frame,(count,509),(count,528),(255,255,255),2)
    #cv2.imshow('frame',frame) #Display image
    cv2.imwrite("C:/Users/liuzi/Desktop/2022S/5642DataVis/Proj/frames/frame%03d.jpg" % count, frame)
    count=count+1
    if cv2.waitKey(1) & 0xFF == ord('q'): #Press q to break and exit while
        break




capture.release() #Exit video capture
cv2.destroyAllWindows() #close window
file.close()

f = open('C:/Users/liuzi/Desktop/2022S/5642DataVis/Proj/angle.txt','r')
lines=f.readlines()

angles=[]

count=0
for line in lines:
    count += 1
    angles.append(int(line.strip()))



width=int(942/count)+1
print(width)

w, h = 942, 30
data = np.zeros((h, w, 3), dtype=np.uint8)
index=0
for i in angles:
    if i <10 :
        data[0:30, index*width:(index+1)*width] = [0, 255, 0]
    elif i>=10 and i<20:
        data[0:30, index*width:(index+1)*width] = [162,205,90]
    elif i>=20 and i<30:
        data[0:30, index*width:(index+1)*width] = [255, 215, 0]
    elif i>=30 and i< 45:
        data[0:30, index*width:(index+1)*width] = [255, 127, 0]
    else:
        data[0:30, index*width:(index+1)*width] = [255, 0, 0]
    index +=1
#data[0:30, 0:30] = [255, 0, 0] # red patch in upper left
img = Image.fromarray(data, 'RGB')


c=0
for filename in glob.glob('C:/Users/liuzi/Desktop/2022S/5642DataVis/Proj/frames/*.jpg'):
    img2=Image.open(filename)
    image1_size = img.size
    image2_size = img2.size
    new_image = Image.new('RGB',(942, 592), (250,250,250))
    new_image.paste(img2,(0,0))
    new_image.paste(img,(0,image2_size[1]))
    new_image.save("C:/Users/liuzi/Desktop/2022S/5642DataVis/Proj/frames2/frame"+str(c).zfill(3) +".jpg","JPEG")
    c+=1


img_array=[]
dir='C:/Users/liuzi/Desktop/2022S/5642DataVis/Proj/frames2/'
index=0
for filename in glob.glob('C:/Users/liuzi/Desktop/2022S/5642DataVis/Proj/frames2/*.jpg'):
    print(filename)
    index=index+1
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('C:/Users/liuzi/Desktop/2022S/5642DataVis/Proj/project2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 29.81, (942,592))

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()




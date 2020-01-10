import cv2 
import numpy as np
 
print(dir(cv2)) 
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def rescale_frame_w_h(frame, target_w = 640, target_h = 480):
    dim = (target_w, target_h)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def norm_int(frame):
    cv2.normalize(frame,frame,0,255,cv2.NORM_MINMAX)
    return frame.astype(int)

    
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('Close_Up_Of_Pot_On_Wood_Fire_1235.mp4',0)

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'FMP4')
fourcc = 0
print(cap.get(3),cap.get(4))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#out = cv2.VideoWriter('flipped_flame.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame_width,frame_height))
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps ",fps)

# Ouverture d'un flux vidéo en écriture
out = cv2.VideoWriter('reduced_flame_color_segmented.avi',fourcc, int(fps), (frame_width,frame_height))
#out2 = cv2.VideoWriter('reduced_flame_gray.avi',fourcc, fps, (frame_width,frame_height))
#out3 = cv2.VideoWriter('reduced_flame_gradienty.avi',fourcc, fps, (frame_width,frame_height))

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")


# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:

    # Affichage de la vidéo avant filtrage
    cv2.imshow('Source',frame)
    # Convert to binary
    # Application d'un filtre binaire sur l'intensité du canal Rouge
    ret, binary = cv2.threshold(frame[:,:,2],127,255,cv2.THRESH_BINARY)
    # Affichage du premier filtrage
    cv2.imshow('1st filter',binary)
    # Deuxième filtrage R > G > B pour isoler les flammes
    binary = binary*(frame[:,:,0]<frame[:,:,1])*(frame[:,:,1]<frame[:,:,2])
    # Affichage après 2ème filtre (les fenêtres sont superposées)
    cv2.imshow('All filters',binary)
    binary = cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)     
    # Ecriture dans le flux de sortie  
    out.write(binary)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
   
  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
out.release()
#out2.release()
# Closes all the frames
cv2.destroyAllWindows()




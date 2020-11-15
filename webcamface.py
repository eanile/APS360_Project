import cv2
import numpy as np

# Load the classifier and create a cascade object for face detection
PATH_TO_VIOLA = 'ViolaJonesModels/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(PATH_TO_VIOLA)

# Webcam display parameters
WIDTH = 480
HEIGHT = 320
MODEL_WIDTH = 48
MODEL_HEIGHT = 48
COLOUR = (0,255,0)
FONT_SCALE = 1.8
BOLD_SCALE = 2
FONT = cv2.FONT_HERSHEY_PLAIN

# Initialization of webcam capture and setting the dimensions
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Gets the frame in colour so it can be displayed w/ colour 
    frame_image = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # Apply the Viola-Jones Algorithm on the frame to retrieve the faces
    detected_faces = face_cascade.detectMultiScale(frame_image)

    # Iterate through all of the detected faces
    for (column, row, width, height) in detected_faces:
        
        # Isolate, convert to GRAYSCALE, and resize the face from the image so it can be passed to the model (face is a np array)
        face = frame_image[row:row+height,column:column+width]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (MODEL_WIDTH, MODEL_HEIGHT))
        
        # TODO pass in the face to the model and calculate the predicion accuracy / confidence

        cv2.putText(
            frame_image, 
            'This is a face!',  # Replace w/ model prediction
            (column, row), 
            FONT, 
            FONT_SCALE, 
            COLOUR, 
            BOLD_SCALE
        )
        cv2.rectangle(
            frame_image,
            (column, row),
            (column + width, row + height),
            COLOUR,
            BOLD_SCALE
        )

    # Display the resulting frame
    cv2.imshow('frame', frame_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
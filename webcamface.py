import cv2
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class KLARR_NET7(nn.Module):
    def __init__(self):
        super(KLARR_NET7, self).__init__()
        self.name = 'KLARR_NET7'

        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm5 = nn.BatchNorm2d(256)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2,2)

        self.conv1 = nn.Conv2d(1, 16, 3, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=2)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=2)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=2)

        '''
        48x48 (1ch) --> conv1 --> 50x50 (16ch)
        50x50 (16ch) --> maxpool --> 25x25 (16ch)
        25x25 (16ch) --> conv2 --> 27x27 (32ch)
        27x27 (32ch) --> maxpool --> 13x13 (32ch)
        13x13 (32ch) --> conv3 --> 15x15 (64ch)
        15x15 (64ch) --> maxpool --> 7x7 (64ch)
        7x7 (64ch) --> conv4 --> 9x9 (128ch)
        9x9 (128ch) --> maxpool --> 4x4 (128ch)
        4x4 (128ch) --> conv5 --> 6x6 (256ch)
        6x6 (256ch) --> avgpool --> 3x3 (256ch)
        '''

        self.fc1 = nn.Linear(3*3*256, 50)
        self.fc2 = nn.Linear(50, 6)

    def forward(self, x):
        # Convolutional layers
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.batchnorm2(self.maxpool(F.relu(self.conv2(x))))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.maxpool(F.relu(self.conv4(x)))
        x = self.batchnorm5(self.avgpool(F.relu(self.conv5(x))))

        # Linear layers
        x = x.view(-1,3*3*256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)             # applies softmax activation on output

        return x

# Load the classifier and create a cascade object for face detection
PATH_TO_VIOLA = 'ViolaJonesModels/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(PATH_TO_VIOLA)

# Load the CNN model that will detect the facial expression
MODEL_PATH = 'models/'
model_final = KLARR_NET7()
model_final.load_state_dict(torch.load("models/model_KLARR_NET7_bs128_lr0.002_iter13_date23_11_2020-03_09"))

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

# Class list to determine what the model predicts
classes = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


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
        
        # Reshape the face so it it is 1 sample, with MODEL_WIDTH and  MODEL_HEIGHT dimensions, and 1 channel
        input_face_to_model = np.reshape(face, (1,MODEL_WIDTH, MODEL_HEIGHT,1)).astype(np.float32) / 255

        model_output = model_final(torch.tensor(input_face_to_model).permute(0, 3, 1, 2))

        predicion = classes[np.argmax(model_output.detach().numpy())]

        cv2.putText(
            frame_image, 
            predicion,  # Replace w/ model prediction
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
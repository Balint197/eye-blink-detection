# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold

# edit these 3 if necessary
# most likely between 0.2-0.3 - print out its value if not detecting open/close to set it
EYE_AR_THRESH = 0.2
# these twp depend a bit on camera FPS, we used 20 FPS camera
EYE_AR_CONSEC_FRAMES_BLINK = 1
EYE_AR_CONSEC_FRAMES_CLOSED = 6

# eye landmarks
eye_landmarks = "model_landmarks/shape_predictor_68_face_landmarks.dat"
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
EYES_CLOSED_STATE = 0
CLOSED = 0
BLINK = 0

import cv2

capture = cv2.VideoCapture('SNATCH_2sec.mp4')

num_of_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))


out = cv2.VideoWriter('vid_motion.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
frame_count = 1
while frame_count < num_of_frames:
    img = cv2.imread('motionDetected/frame%d.tif' % frame_count)
    out.write(img)
    frame_count += 1
capture.release()
cv2.destroyAllWindows()

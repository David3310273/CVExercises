import cv2 as cv
cap = cv.VideoCapture(0)

# get width and height of the view port, the video's size must be same as them
width = int(cap.get(3))
height = int(cap.get(4))

# set format as mp4 on Mac.
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('helloworld.mp4',fourcc, 10.0, (width, height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # write the frame
        out.write(frame)
        # custiomize the window title
        cv.imshow('demo',frame)
        # program will quit if press 'q' on keyboard
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
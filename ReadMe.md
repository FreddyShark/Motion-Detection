* Run motionDetect.py to execute the program after extracting file contents
* If you would like to preview the output video, before all frames have been processed, run getVideo.py
* Two short (2second) video examples have been included. One is motion against a green screen without shadows present; However, the other is currently specified for processing.
* If using another video ensure, it has a resolution supported by M-JPEG
* output file will be in avi format. File name = vid_motion.avi

* Input video file names:
  * SNATCH_2sec.mp4 (preset)
  * try_greenScreen.mp4 (also available)

* Prepared output video files:
  * vid_motion.avi
  * vid_motion_green_screen.avi

* Modules required: 
  * numpy 1.11.5 
  * opencv-python 3.4.2.1
  * math
  * os
  * sys
  * warnings

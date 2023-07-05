# Code obtained from https://github.com/geaxgx/depthai_hand_tracker

import numpy as np
import mediapipe_utils as mpu
import mediapipe as mp
import cv2
from pathlib import Path
from FPS import FPS, now

class MyHandTracker:
    def __init__(self
                ):
        self.cap = cv2.VideoCapture(0)
        mpHands = mp.solutions.hands
        self.handsolution = mpHands.Hands(model_complexity=0)

    def recognize_gesture(self, r):         
        
        # Finger states
        # state: -1=unknown, 0=close, 1=open
        d_3_5 = mpu.distance(r[3], r[5])
        d_2_3 = mpu.distance(r[2], r[3])
        angle0 = mpu.angle(r[0], r[1], r[2])
        angle1 = mpu.angle(r[1], r[2], r[3])
        angle2 = mpu.angle(r[2], r[3], r[4])
        thumb_angle = angle0+angle1+angle2
        if angle0+angle1+angle2 > 460 and d_3_5 / d_2_3 > 1.2: 
            thumb_state = 1
        else:
            thumb_state = 0

        if r[8][1] < r[7][1] < r[6][1]:
            index_state = 1
        elif r[6][1] < r[8][1]:
            index_state = 0
        else:
            index_state = -1

        if r[12][1] < r[11][1] < r[10][1]:
            middle_state = 1
        elif r[10][1] < r[12][1]:
            middle_state = 0
        else:
            middle_state = -1

        if r[16][1] < r[15][1] < r[14][1]:
            ring_state = 1
        elif r[14][1] < r[16][1]:
            ring_state = 0
        else:
            ring_state = -1

        if r[20][1] < r[19][1] < r[18][1]:
            little_state = 1
        elif r[18][1] < r[20][1]:
            little_state = 0
        else:
            little_state = -1

        # Gesture
        if thumb_state == 1 and index_state == 1 and middle_state == 1 and ring_state == 1 and little_state == 1:
            gesture = "FIVE"
        elif thumb_state == 0 and index_state == 0 and middle_state == 0 and ring_state == 0 and little_state == 0:
            gesture = "FIST"
        elif thumb_state == 1 and index_state == 0 and middle_state == 0 and ring_state == 0 and little_state == 0:
            gesture = "ZOOM" 
        elif thumb_state == 0 and index_state == 1 and middle_state == 1 and ring_state == 0 and little_state == 0:
            gesture = "PEACE"
        elif thumb_state == 0 and index_state == 1 and middle_state == 0 and ring_state == 0 and little_state == 0:
            gesture = "ONE"
        elif thumb_state == 1 and index_state == 1 and middle_state == 0 and ring_state == 0 and little_state == 0:
            gesture = "ZOOM"
        elif thumb_state == 1 and index_state == 1 and middle_state == 1 and ring_state == 0 and little_state == 0:
            gesture = "THREE"
        elif thumb_state == 0 and index_state == 1 and middle_state == 1 and ring_state == 1 and little_state == 1:
            gesture = "FOUR"
        else:
            gesture = None
        # print(gesture)
        return gesture
   
    def next_frame(self):
        self.hands =[]
        gesture='None'
        ret,video_frame = self.cap.read()
        square_frame = video_frame
        img_rgb = cv2.cvtColor(video_frame,cv2.COLOR_BGR2RGB)
        result = self.handsolution.process(img_rgb)
        if result.multi_hand_landmarks:
            self.hands=result.multi_hand_landmarks
            for i,h in enumerate(self.hands):
                r = []
                for i , lm in enumerate(h.landmark):
                    r.append(np.array([lm.x,lm.y,lm.z]))
                gesture = self.recognize_gesture(r)
        return video_frame, self.hands, gesture


    def exit(self):
        self.device.close()
        # Print some stats
        if self.stats:
            print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {self.fps.nb_frames()})")
            if self.body_pre_focusing:
                print(f"# body pose estimation inferences received : {self.nb_bpf_inferences}")
            print(f"# palm detection inferences received       : {self.nb_pd_inferences}")
            if self.use_lm: print(f"# hand landmark inferences received        : {self.nb_lm_inferences}")
            if self.input_type != "rgb":
                if self.body_pre_focusing:
                    print(f"Body pose estimation round trip      : {self.glob_bpf_rtrip_time/self.nb_bpf_inferences*1000:.1f} ms")
                print(f"Palm detection round trip            : {self.glob_pd_rtrip_time/self.nb_pd_inferences*1000:.1f} ms")
                if self.use_lm and self.nb_lm_inferences:
                    print(f"Hand landmark round trip             : {self.glob_lm_rtrip_time/self.nb_lm_inferences*1000:.1f} ms")
            if self.xyz:
                print(f"Spatial location requests round trip : {self.glob_spatial_rtrip_time/self.nb_anchors*1000:.1f} ms")           

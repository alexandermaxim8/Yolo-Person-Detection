import cv2
from ultralytics import YOLO, solutions
import torch

if torch.cuda.is_available():
   device='cuda'
else:
   device='cpu'

model = YOLO("best.pt").to(device)

cap = cv2.VideoCapture(0)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, 30))

line_points = [(w/2, 0), (w/2, h)]

# video_writer = cv2.VideoWriter("object_counting_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps/2, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    count_reg_color=(255, 0, 0),
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
    line_dist_thresh = 1
)

while cap.isOpened():
    success, im0 = cap.read()
    key = cv2.waitKey(30) & 0xFF

    tracks = model.track(im0, persist=True, tracker='bytetrack.yaml', show=True)

    im0 = counter.start_counting(im0, tracks)
    cv2.imshow('frame', im0) 
    # video_writer.write(im0)

    if key == ord('q') or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == -1:
        break

cap.release()
# video_writer.release()
cv2.destroyAllWindows()
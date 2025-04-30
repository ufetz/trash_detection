#!/usr/bin/env python
"""
YOLO trash detection script.
"""

import argparse
import cv2
from ultralytics import YOLO
from bluerov_stream import Video


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO trash detection script.")
    parser.add_argument('--port', type=int, default=5600, help='UDP port for video stream')
    parser.add_argument('--model', type=str, default='yolo11l', help='Path to YOLO model weights')
    args = parser.parse_args()

    # Create the video object
    video = Video(port=args.port)

    model_str = 'model/' + args.model + '.pt'
    # YOLO model
    model = YOLO(model_str)
    #model = YOLO('runs/detect/train/weights/best.pt')

    print('Initialising stream...')
    waited = 0
    while not video.frame_available():
        waited += 1
        print('\r  Frame not available (x{})'.format(waited), end='')
        cv2.waitKey(30)
    print('\nSuccess!\nStarting streaming - press "q" to quit.')

    while True:
        # Wait for the next frame to become available
        if video.frame_available():
            # Only retrieve and display a frame if it's new
            frame = video.frame()
            results = model(frame)
            annotated_frame = results[0].plot()

            cv2.imshow('frame', annotated_frame)
            #cv2.imshow('frame', frame)
        # Allow frame to display, and check if user wants to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

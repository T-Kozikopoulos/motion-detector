import cv2
import pandas as pd
from datetime import datetime as dt

# Create necessary variables and set as None to avoid useless errors.
first_frame = None
status_list = [None, None]
timestamps = []

# Create a dataframe for timestamps of when there is movement
df = pd.DataFrame(columns=['Start', 'End'])

# Choose the webcam.
# Use '0' if you onle have 1 conneceted to the PC, else try out different
# one until you have picked the one you wan to monitor for movement.
video = cv2.VideoCapture(0)

while True:
	check, frame = video.read()
	status = 0
	# Convert video to black and whiteand blur it (same accuracy, less workload).
	gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	gray = cv2.GaussianBlur(gray,(21, 21), 0)

	# Set the first frame as the point of comparison.
	if first_frame is None:
		first_frame = gray
		continue

	# Create the threshold frames.
	delta_frame = cv2.absdiff(first_frame, gray)
	thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
	thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

	(_,cnts,_) = cv2.findContours(thresh_frame.copy(),
								  cv2.RETR_EXTERNAL,
								  cv2.CHAIN_APPROX_SIMPLE)

	for contour in cnts:
		# Ignore tiny changes (less than 100x100 pixels)
		if cv2.contourArea(contour) < 10000:
			continue
		status = 1
		# Create rectangles that appear around moving objects.
		(x, y, w, h) = cv2.boundingRect(contour)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# Add the movement status to a list.
	status_list.append(status)
	# Attach timestamps to the periods of movement for later examination.
	if status_list[-1] == 1 and status_list[-2] == 0:
		timestamps.append(dt.now().strftime('%d/%m/%y %H:%M:%S'))
	if status_list[-1] == 0 and status_list[-2] == 1:
		timestamps.append(dt.now().strftime('%d/%m/%y %H:%M:%S'))

	# Open preview windows showing the live processing of movement.
	cv2.imshow('Threshold', thresh_frame)
	cv2.imshow('Colored boundaries', frame)

	key = cv2.waitKey(30)

	# Press 'Q' to close all windows.
	if key == ord('q'):
		if status == 1:
			# If the script is shut down while there is movement, add the current
			# time as the end of the period.
			timestamps.append(dt.now().strftime('%d/%m/%y %H:%M:%S'))
		break

# Save timestamps into a dataframe.
for i in range(0, len(timestamps), 2):
	df = df.append({'Start': timestamps[i], 'End': timestamps[i+1]}, ignore_index=True)

# Save dataframe as csv file.
df.to_csv('timestamps.csv')

# All done!
video.release()
cv2.destroyAllWindows

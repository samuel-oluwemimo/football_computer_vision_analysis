from ultralytics import YOLO

model = YOLO(r'C:\Users\632366\Documents\football_analysis\models\best.pt')

results = model.predict(r'C:\Users\632366\Documents\football_analysis\video_data\08fd33_4.mp4', save=True)

print(results[0])
print('*****************************')
for box in results[0]:
    print(box)
import streamlit as st
import cv2
import cvzone
import math
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Set page title and favicon
st.set_page_config(page_title="Forest Fire Detection App", page_icon="🔥")

# Define the main title and description
st.title("Advanced Forest Fire Detection and Monitoring System")
st.write("This app uses YOLO and Ultralytics to detect forest fires in videos. Learn about the effects of forest fires below.")

# Forest fire effects information section
st.markdown("## Effects of Forest Fires")
effects_text = (
    "Forest fires can have significant and lasting impacts on ecosystems, wildlife, and communities. They can lead to:\n"
    "- Loss of wildlife habitat\n"
    "- Air quality deterioration\n"
    "- Soil erosion and reduced water quality\n"
    "- Economic losses from property damage\n"
    "- Disruption of livelihoods\n"
    "- Increased risk of flooding and landslides"
)
st.write(effects_text)

# Display images related to forest fire effects
classes={0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}

# Fact section
st.markdown("## Interesting Facts about Forest Fires")
facts = [
    "Forest fires can be caused by both natural factors, like lightning, and human activities, such as campfires and discarded cigarette butts.",
    "Some plants, like pine trees, have evolved to rely on forest fires to release their seeds and regenerate the forest ecosystem.",
    "Forest fires can create their own weather systems, including fire tornadoes and pyrocumulus clouds.",
    "The largest wildfire in recorded history is the 2003 Siberian Taiga Fires, which burned over 47 million acres of forest.",
    "Forest fires release large amounts of carbon dioxide into the atmosphere, contributing to global warming.",
    "Proper forest management, including controlled burns, can help reduce the severity of future wildfires."
]

fact_idx = st.empty().radio("Choose a Fact:", list(range(len(facts))))

# Placeholder for the chat-like interaction
fact_chat_placeholder = st.empty()

if fact_idx is not None:
    fact_chat_placeholder.write(f"Fascinating Fact: {facts[fact_idx]}")

# File upload and detection section
st.markdown("## Upload and Detect")
uploaded_file = st.file_uploader("Upload a video file...", type=["mp4"])
if uploaded_file:
    # Initialize an empty list to store timestamps
    book_timestamps = []

    st.subheader("Real Time Fire Detection")
    cap = cv2.VideoCapture(uploaded_file.name)
    stframe = st.empty()  # Placeholder for displaying frames

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = math.ceil((box.conf[0] * 100)) / 100
                # st.write(classes[box.cls.item()])

                # st.write(classes[box.cls])

                cvzone.putTextRect(img, f'book {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                   colorB=(0, 0, 255), colorT=(255, 255, 255), colorR=(0, 0, 255), offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # Check if the detected object is a book (you can adjust this condition based on your YOLO model's classes)
                if classes[box.cls.item()] == "book":
                    # Retrieve timestamp from the video
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert milliseconds to seconds
                    # st.write("booksssss")

                    # Append the timestamp to the list
                    book_timestamps.append(timestamp)

                    # Add visualization or any other action as needed
                    # For example, drawing a bounding box or adding text on the frame

        stframe.image(img, channels="BGR", use_column_width=True)

    # Write the list of timestamps to a text file
    output_file_path = "book_timestamps.txt"
    with open(output_file_path, "w") as f:
        
            f.write(str({"books":book_timestamps}))

    # Clean up
    cap.release()
else:
    st.warning("Please upload a video file to get started.")

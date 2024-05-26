import cv2 as cv
import torch
import os
from network import Net

# from cascade import create_cascade
from transforms import ValidationTransform
from PIL import Image
from record import get_haar_cascade_path


# NOTE: This will be the live execution of your pipeline


def live(args):
    # TODO:
    #   Load the model checkpoint from a previous training session (check code in train.py)
    checkpoint = torch.load("model.pt")
    model = Net(nClasses=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model"])
    classes = checkpoint["classes"]

    #   Initialize the face recognition cascade again (reuse code if possible)
    HAAR_CASCADE = get_haar_cascade_path()
    face_cascade = cv.CascadeClassifier(HAAR_CASCADE)

    #   Also, create a video capture device to retrieve live footage from the webcam.
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    #   Attach border to the whole video frame for later cropping.
    #   Run the cascade on each image, crop all faces with border.
    #   Run each cropped face through the network to get a class prediction.
    #   Retrieve the predicted persons name from the checkpoint and display it in the image

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        height, width = frame.shape[:2]
        top_bottom_border = int((height * float(args.border)) / 2)
        left_right_border = int((width * float(args.border)) / 2)

        frame_border = cv.copyMakeBorder(
            frame,
            top_bottom_border,
            top_bottom_border,
            left_right_border,
            left_right_border,
            cv.BORDER_REFLECT,
        )

        gray = cv.cvtColor(frame_border, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        frame_rectangle = frame_border.copy()

        for (x, y, w, h) in faces:
            # Show blue border like at record.py
            top_bottom_border_crop = int((h * float(args.border)) / 2)
            left_right_border_crop = int((w * float(args.border)) / 2)
            left = max(x - left_right_border_crop, 0)
            right = min(x + w + left_right_border_crop, frame_border.shape[1])
            top = max(y - top_bottom_border_crop, 0)
            bottom = min(y + h + top_bottom_border_crop, frame_border.shape[0])

            if top >= bottom or left >= right:
                print("Ungültige Rahmenkoordinaten:", top, bottom, left, right)
                continue

            frame_rectangle = cv.rectangle(
                frame_rectangle, (left, top), (right, bottom), (255, 0, 0), 2
            )
            cropped_face = frame_border[top:bottom, left:right]

            if cropped_face.size == 0:
                print("Leeres Gesichtsfeld bei:", top, bottom, left, right)
                continue

            # Convert frame to image
            frame_image = Image.fromarray(cv.cvtColor(cropped_face, cv.COLOR_BGR2RGB))

            # Transform to tensor
            transformed_image = ValidationTransform(frame_image)

            # Classify tensor using the pre-trained model
            with torch.no_grad():
                model.eval()
                output = model(transformed_image.unsqueeze(0))
                # predicted_class = torch.argmax(output, dim=1).item()
                # class_name = classes[predicted_class]
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                accuracy = probabilities[0, predicted_class].item()

            # Display the name and accuracy near the rectangle
            person_name = classes[predicted_class]
            text = f"{person_name} (Acc: {accuracy:.2f})"

            text_scale = min(w, h) / 200
            text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, text_scale, 2)
            text_x = left + (right - left - text_size[0]) // 2
            text_y = top - 10

            cv.rectangle(
                frame_rectangle,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (0, 0, 0),
                -1,
            )
            cv.putText(
                frame_rectangle,
                text,
                (text_x, text_y),
                cv.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                2,
            )

        cv.imshow("webcam", frame_rectangle)
        if cv.waitKey(1) == ord("q"):
            break

    if args.border is None:
        print("Live mode requires a border value to be set")
        exit()

    args.border = float(args.border)
    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()

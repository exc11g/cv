from ultralytics import YOLO
import cv2
import numpy as np
import os

def train(model):

    model.train(
        data="Russian-Road-Signs-4\\data.yaml",
        epochs=5,
        imgsz=400,
        amp=False,
        workers=0,
        save=True,
        name='segment-signs-train',
    )
    model.save()
    return model

def val(model):
    metrics = model.val(
        data="Russian-Road-Signs-4\\data.yaml",
        workers=0,
        save=True,
        imgsz=640,
        name='segment-signs-val',
    )

    print("map50", metrics.seg.map50)
    print("map75", metrics.seg.map75)
    print("map90", metrics.seg.map) 

def predict_image(model, images):
    results = model.predict(
        source=images,
        save=True,
        conf=0.25,
        workers=0,
    )

    return results

def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def load_mask(label_path, img_shape):
    mask = np.zeros(img_shape[:2])

    h, w = img_shape[:2]

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            shape = parts[1:]

            pts = []
            for i in range(0, len(shape), 2):
                x = int(shape[i] * w)
                y = int(shape[i + 1] * h)
                pts.append([x, y])

            pts = np.array(pts)
            cv2.fillPoly(mask, [pts], 1)

    return mask


def eval_iou(model_path, dataset_path):
    model = YOLO(model_path)

    images_dir = os.path.join(dataset_path, "valid/images")
    labels_dir = os.path.join(dataset_path, "valid/labels")

    ious = []

    for img in os.listdir(images_dir):
        label_path = os.path.join(labels_dir, img.replace(".jpg", ".txt"))

        img = cv2.imread(os.path.join(images_dir, img))
        h, w = img.shape[:2]

        gt_mask = load_mask(label_path, img.shape)

        results = model.predict(img, conf=0.25, verbose=False)

        if results[0].masks is None:
            pred_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            pred_mask = np.zeros((h, w), dtype=np.uint8)
            for m in results[0].masks.data:
                mask = cv2.resize(m.cpu().numpy(), (w, h))

                pred_mask |= (mask > 0.5).astype(np.uint8)

        ious.append(iou(pred_mask, gt_mask))

    ious = np.array(ious)

    p50 = np.mean(ious >= 0.5)
    p75 = np.mean(ious >= 0.75)
    p90 = np.mean(ious >= 0.9)

    print(f"iou 0.5: {p50}")
    print(f"iou 0.75: {p75}")
    print(f"iou 0.9: {p90}")




def center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def track_video(model, input, output, tracker):

    cap = cv2.VideoCapture(input)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    prev_objects = []
    id_switches = 0

    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model.track(
            frame,
            tracker=tracker,
            persist=True,
            verbose=False
        )

        out.write(res[0].plot())

        current_objects = []

        if res[0].boxes is not None and res[0].boxes.id is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy()
            ids = res[0].boxes.id.cpu().numpy()

            for box, obj_id in zip(boxes, ids):
                current_objects.append({
                    "id": int(obj_id),
                    "center": center(box)
                })

        for curr in current_objects:
            best_match = None
            best_dist = 1e9

            for prev in prev_objects:
                dist = np.linalg.norm(curr["center"] - prev["center"])

                if dist < best_dist:
                    best_dist = dist
                    best_match = prev

            if best_match is not None and best_dist < 50:
                    if curr["id"] != best_match["id"]:
                        id_switches += 1

        prev_objects = current_objects
        frame_i += 1

    cap.release()
    out.release()

    print("Output video", output)
    print("id swtich:", id_switches)


model = train(YOLO("yolov8m-seg.pt"))

eval_iou(
    model_path="saved_model.pt",
    dataset_path="Russian-Road-Signs-4"
)


# model = YOLO("saved_model.pt")
# for v in [f"my_photos/sv{j}" for j in range(1, 4)]:
#     for t in ["bytetrack.yaml", "botsort.yaml"]:
#         input = v + ".mp4"
#         out = v + f"-{t}.mp4"
#         track_video(model, input, out, tracker=t)


# val(model)
# model = YOLO("saved_model.pt")

# predict_image(model, [f"my_photos/s{i}.jpg" for i in range(1, 5)])

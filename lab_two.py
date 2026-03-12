import os
from ultralytics import YOLO
from roboflow import Roboflow

IMG_SIZE = 700
BATCH_SIZE = 16
EPOCHS_PRETRAIN = 7
EPOCHS_FINETUNE = 10
DEVICE = '0'

yaml_nd = "numberdetection-2" + "/data.yaml"
yaml_svhn = "YOLOv5-SVHN-1" + "/data.yaml"

def check_my_photos(model, out):
    COUNT_PHOTOS = 7
    os.makedirs("results_" + out, exist_ok=True)

    for i in range(1, COUNT_PHOTOS + 1):
        photo_path = f"my_photos/{i}.jpg"
            
        results = model.predict(
            source=photo_path,
            imgsz=IMG_SIZE,
            conf=0.25,
            save=True,
            project="results_" + out,
            name=os.path.basename(photo_path),
            device=DEVICE
        )
    print("My PHOTOS DONE")


def pretrain():
    model = YOLO('yolov8n.pt')

    results_pretrain = model.train(
        data=yaml_nd,
        epochs=EPOCHS_PRETRAIN,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        name='pretrain_numberdetection',
        patience=10,
        augment=True,
        save=True,
        amp=False,
        workers=0  
    )

    metrics = model.val(
        data=yaml_svhn,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        save_json=True,
        name='eval_numberdetection',
        amp=False,
        workers=0
    )


    print("ND RESULT")
    print("Precision: ", metrics.box.mp)
    print("Recall: ", metrics.box.mr)
    print("mAP 0.5: ", metrics.box.map50)
    print("mAP 0.95: ", metrics.box.map)

    check_my_photos(model, "pretrain")
    return model

def finetune():
    print("FINETUNE SVHN")
    model = YOLO('yolov8n.pt')
    # model = YOLO("runs\\detect\\pretrain_numberdetection24\\weights\\best.pt")
    # model = YOLO("runs\\detect\\finetune_svhn15\\weights\\best.pt")

    results_finetune = model.train(
        data=yaml_svhn,
        epochs=3,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        name='finetune_svhn',
        patience=50,
        lr0=0.01,
        lrf=0.1,
        augment=True,
        save=True,
        amp=False,
        workers=0
    )


    metrics = model.val(
        data=yaml_svhn,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        save_json=True,
        name='eval_svhn',
        amp=False,
        workers=0
    )


    print("SBVH RESULT")
    print("Precision: ", metrics.box.mp)
    print("Recall: ", metrics.box.mr)
    print("mAP 0.5: ", metrics.box.map50)
    print("mAP 0.95: ", metrics.box.map)

    check_my_photos(model, "finetune")
    return model

# pretrain()
# finetune()
model = YOLO("runs\\detect\\finetune_svhn15\\weights\\best.pt")
check_my_photos(model, "test")


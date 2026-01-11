# DISHARI
import cv2
import numpy as np
import onnxruntime as ort

# ---------- Load YOLOv5 ONNX ----------
#session = ort.InferenceSession("yolov5n.onnx", providers=["CPUExecutionProvider"])
import os

MODEL_PATH = r"C:\Users\USER\Desktop\YOLOv5_ONNOX\yolov5\yolov5n.onnx"

assert os.path.exists(MODEL_PATH), "ONNX model file NOT FOUND!"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("YOLOv5 ONNX Runtime loaded successfully")

# ---------- Load COCO Labels ----------
#with open("coco.names", "r") as f:
import os

LABEL_PATH = os.path.join(os.path.dirname(__file__), "coco.names")
with open(LABEL_PATH, "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

    LABELS = [line.strip() for line in f.readlines()]

# ---------- Parameters ----------y
INPUT_SIZE = 640
CONF_TH = 0.3
NMS_TH = 0.4
GRID = 3  # ROI grid (3x3)

# ROI (bottom-middle)
ROI_WIDTH_RATIO = 0.5
ROI_HEIGHT_RATIO = 0.35
ROI_BOTTOM_MARGIN = 10

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    # ---------- Define ROI ----------
    roi_w = int(W * ROI_WIDTH_RATIO)
    roi_h = int(H * ROI_HEIGHT_RATIO)
    roi_x1 = (W - roi_w) // 2
    roi_y1 = H - roi_h - ROI_BOTTOM_MARGIN
    roi_x2 = roi_x1 + roi_w
    roi_y2 = roi_y1 + roi_h

    segment_state = [0] * (GRID * GRID)
    detected_segments = set()

    # ---------- Preprocess Full Frame ----------
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # ---------- Inference ----------
    outputs = session.run([output_name], {input_name: img})[0][0]

    boxes, scores = [], []

    for det in outputs:
        obj_conf = det[4]
        if obj_conf < CONF_TH:
            continue

        cls_id = np.argmax(det[5:])
        score = obj_conf * det[5 + cls_id]
        if score < CONF_TH:
            continue

        cx, cy, bw, bh = det[:4]
        cx *= W / INPUT_SIZE
        cy *= H / INPUT_SIZE
        bw *= W / INPUT_SIZE
        bh *= H / INPUT_SIZE

        x = int(cx - bw / 2)
        y = int(cy - bh / 2)

        boxes.append([x, y, int(bw), int(bh)])
        scores.append(float(score))

    # ---------- NMS ----------
    idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_TH, NMS_TH)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]

            # ---- ROI OVERLAP CHECK ----
            bx1, by1 = x, y
            bx2, by2 = x + bw, y + bh

            inter_x1 = max(bx1, roi_x1)
            inter_y1 = max(by1, roi_y1)
            inter_x2 = min(bx2, roi_x2)
            inter_y2 = min(by2, roi_y2)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                # ---- Multi-Segment Overlap in ROI ----
                step_x = roi_w / GRID
                step_y = roi_h / GRID

                col_start = int((inter_x1 - roi_x1) / step_x)
                col_end   = int((inter_x2 - roi_x1) / step_x)
                row_start = int((inter_y1 - roi_y1) / step_y)
                row_end   = int((inter_y2 - roi_y1) / step_y)

                for r in range(row_start, row_end + 1):
                    for c in range(col_start, col_end + 1):
                        if 0 <= r < GRID and 0 <= c < GRID:
                            detected_segments.add(r * GRID + c)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 255), 2)

    # ---------- Instant Segment Output ----------
    for idx in detected_segments:
        segment_state[idx] = 1

    # ---------- Draw ROI ----------
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)

    # ---------- Draw ROI Grid ----------
    step_x = roi_w // GRID
    step_y = roi_h // GRID

    for i in range(1, GRID):
        cv2.line(frame, (roi_x1 + i * step_x, roi_y1),
                 (roi_x1 + i * step_x, roi_y2), (0, 255, 0), 1)
        cv2.line(frame, (roi_x1, roi_y1 + i * step_y),
                 (roi_x2, roi_y1 + i * step_y), (0, 255, 0), 1)

    for i, val in enumerate(segment_state):
        r, c = divmod(i, GRID)
        cx = roi_x1 + c * step_x + step_x // 2
        cy = roi_y1 + r * step_y + step_y // 2
        color = (0, 0, 255) if val else (0, 255, 0)
        cv2.circle(frame, (cx, cy), 9, color, -1)

    print("ROI Segment Output:", segment_state)

    cv2.imshow("Phase-5 YOLOv5 ROI Overlap", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


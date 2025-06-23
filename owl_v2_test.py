import cv2
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# ────────────────── 설정 ────────────────── #
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
PROC    = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
MODEL   = Owlv2ForObjectDetection.from_pretrained(
              "google/owlv2-base-patch16-ensemble"
          ).to(DEVICE).eval()

PROMPTS = [[
    "debris on the highway",
    "small black dot on the highway",
    "orange cones on the highway",
]]
BOX_TH  = 0.25
VIDEO   = r"D:\data\도로_비정형객체\2.avi"

# ────────────────── 비디오 루프 ────────────────── #
cap = cv2.VideoCapture(VIDEO)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # BGR → PIL(RGB)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 전처리 + GPU 이동
    inputs = PROC(text=PROMPTS, images=pil, return_tensors="pt").to(DEVICE)

    # 추론
    with torch.no_grad():
        out = MODEL(**inputs)

    # 후처리: 모두 키워드 인자로 전달
    sz = torch.tensor([(pil.height, pil.width)]).to(DEVICE)
    results = PROC.post_process_grounded_object_detection(
        outputs=out,
        target_sizes=sz,
        threshold=BOX_TH,
        text_labels=PROMPTS
    )[0]

    # 시각화
    for box, score, label in zip(
            results["boxes"], results["scores"], results["text_labels"]
        ):
        if score < BOX_TH:
            continue
        x0, y0, x1, y1 = map(int, box.tolist())
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x0, max(y0 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
        )

    cv2.imshow("OWL-v2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

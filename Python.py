
import cv2
import torch
import numpy as np
import scipy.special
import torchvision.transforms as transforms

from PIL import Image
from model.model import parsingNet
from utils.common import merge_config
from data.constant import culane_row_anchor


# ---------------- CONFIGURATION ----------------

# VIDEO_PATH = r"D:\tmp\clone\challenge_video.mp4"
VIDEO_PATH = r"D:\tmp\harder_challenge_video.mp4"

# VIDEO_PATH = r"D:\tmp\Adaptive_Lane_Detection_V2 (1)\Adaptive_Lane_Detection_V2\input\challenge_video.mp4"
MODEL_PATH = r"D:\tmp\clone\culane_18.pth"
OUTPUT_PATH = "baja_ufld_output.mp4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    torch.backends.cudnn.benchmark = True

    print("Device:", DEVICE)

    # Load CULane configuration
    import sys
    sys.argv = ["baja_video_inference.py", "configs/culane.py"]
    args, cfg = merge_config()

    # Load model
    net = parsingNet(
        pretrained=False,
        backbone=cfg.backbone,
        cls_dim=(cfg.griding_num + 1, 18, 4),
        use_aux=False
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    state_dict = checkpoint["model"]

    compatible_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        compatible_state_dict[key] = value

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Cannot open input video: " + VIDEO_PATH)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height)
    )

    col_sample = np.linspace(0, 799, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    frame_count = 0

    print("Starting lane detection...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = net(input_tensor)

        # Decode lane predictions
        out = output[0].cpu().numpy()
        out = out[:, ::-1, :]

        prob = scipy.special.softmax(out[:-1], axis=0)

        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)

        loc = np.sum(prob * idx, axis=0)

        out_argmax = np.argmax(out, axis=0)
        loc[out_argmax == cfg.griding_num] = 0

        # Draw detected lane points
        for lane in range(loc.shape[1]):

            if np.sum(loc[:, lane] > 0) > 2:

                for row in range(loc.shape[0]):

                    if loc[row, lane] > 0:

                        x = int(
                            loc[row, lane] *
                            col_sample_w * width / 800
                        ) - 1

                        y = int(
                            height *
                            (culane_row_anchor[17-row] / 288)
                        ) - 1

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 255, 0),
                            -1
                        )

        writer.write(frame)

        frame_count += 1

        if frame_count % 100 == 0:
            print("Processed frames:", frame_count)

    cap.release()
    writer.release()

    print("Processing completed!")
    print("Total frames:", frame_count)
    print("Output saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

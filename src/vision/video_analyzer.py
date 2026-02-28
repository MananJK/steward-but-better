import cv2
import json
import os
import base64
from pathlib import Path
from datetime import datetime
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
VIDEO_PATH = "clips/abu_dhabi_overtake.mp4"
NUM_FRAMES = 4
OUTPUT_PATH = "vision_report.json"

SYSTEM_PROMPT = """You are an F1 steward analyzing racing incident footage. 
Provide detailed, objective observations about vehicle positions, racing room, and track limits.
Focus on factual visual evidence only."""


def extract_key_frames(video_path: str, num_frames: int = 4) -> list:
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames < num_frames:
        num_frames = max(1, total_frames)

    frame_indices = []
    for i in range(num_frames):
        idx = int((total_frames / (num_frames + 1)) * (i + 1))
        frame_indices.append(idx)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def encode_frame_to_base64(frame) -> str:
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def analyze_frames_with_vlm(frames: list, client: InferenceClient) -> dict:
    if not frames:
        return {"error": "No frames to analyze"}

    steward_prompt = """Does the car on the inside leave sufficient racing room at the apex? 
The telemetry reports a 1.8m gap. Confirm if this matches the visual evidence."""

    frame_base64 = encode_frame_to_base64(frames[0])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{frame_base64}"},
                {"type": "text", "text": steward_prompt},
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model="microsoft/Phi-3.5-vision-instruct", messages=messages, max_tokens=500
        )
        return {
            "analysis": response.choices[0].message.content,
            "model": "microsoft/Phi-3.5-vision-instruct",
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_multiple_frames(frames: list, client: InferenceClient) -> dict:
    if not frames:
        return {"error": "No frames to analyze"}

    steward_prompt = """Does the car on the inside leave sufficient racing room at the apex? 
The telemetry reports a 1.8m gap. Confirm if this matches the visual evidence.
Analyze each frame and provide detailed observations about vehicle positions and racing room."""

    all_analyses = []

    for i, frame in enumerate(frames):
        frame_base64 = encode_frame_to_base64(frame)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{frame_base64}",
                    },
                    {"type": "text", "text": steward_prompt},
                ],
            }
        ]

        try:
            response = client.chat.completions.create(
                model="microsoft/Phi-3.5-vision-instruct",
                messages=messages,
                max_tokens=500,
            )
            all_analyses.append(
                {"frame_index": i, "analysis": response.choices[0].message.content}
            )
        except Exception as e:
            all_analyses.append({"frame_index": i, "error": str(e)})

    return {
        "frame_analyses": all_analyses,
        "model": "microsoft/Phi-3.5-vision-instruct",
    }


def save_vision_report(report: dict, output_path: str):
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


def main():
    print(f"Extracting frames from {VIDEO_PATH}...")
    frames = extract_key_frames(VIDEO_PATH, NUM_FRAMES)

    if not frames:
        print(f"Error: Could not extract frames from {VIDEO_PATH}")
        print("Creating report with placeholder data...")
        report = {
            "status": "error",
            "message": f"Video file not found: {VIDEO_PATH}",
            "timestamp": datetime.now().isoformat(),
        }
        save_vision_report(report, OUTPUT_PATH)
        return

    print(f"Extracted {len(frames)} frames")

    if not HF_TOKEN:
        print("Error: HF_TOKEN not found in environment")
        report = {
            "status": "error",
            "message": "HF_TOKEN not configured",
            "timestamp": datetime.now().isoformat(),
        }
        save_vision_report(report, OUTPUT_PATH)
        return

    print("Initializing Hugging Face InferenceClient...")
    client = InferenceClient(token=HF_TOKEN)

    print("Analyzing frames with Vision-Language Model...")
    analysis_result = analyze_multiple_frames(frames, client)

    report = {
        "status": "success",
        "video_file": VIDEO_PATH,
        "frames_extracted": len(frames),
        "timestamp": datetime.now().isoformat(),
        "steward_question": "Does the car on the inside leave sufficient racing room at the apex? The telemetry reports a 1.8m gap. Confirm if this matches the visual evidence",
        "analysis": analysis_result,
    }

    save_vision_report(report, OUTPUT_PATH)
    print(f"Vision report saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

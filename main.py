# main.py
#
# FastAPI 서버
# 역할: 라우팅, 파비콘, 광고 생성 요청, 영상 완성 확인, 음성+영상 합성

# ──────────────────────────────────────────────
# [0] 환경변수 로드 (가장 먼저)
# ──────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()


# ──────────────────────────────────────────────
# [1] 라이브러리 임포트
# ──────────────────────────────────────────────
import os
import subprocess

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, Response  # ← 세 개 한 번에
from fastapi.staticfiles import StaticFiles

from agent import generate_ad_with_graph, extract_first_frame


# ──────────────────────────────────────────────
# [2] 앱 초기화 및 static 마운트
# ──────────────────────────────────────────────
app = FastAPI(title="AI 광고 크리에이터 API")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ──────────────────────────────────────────────
# [3] 파비콘
# static/favicon.ico 파일이 있으면 반환, 없으면 204 (No Content)
# ──────────────────────────────────────────────
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = "static/favicon.ico"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    return Response(status_code=204)


# ──────────────────────────────────────────────
# [4] 홈 페이지
# ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    html_path = os.path.join("templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>templates/index.html 파일을 찾을 수 없습니다.</h1>"


# ──────────────────────────────────────────────
# [5] 헬퍼 함수: 음성 + 영상 합성
# ffmpeg로 무음 영상과 TTS 오디오를 합쳐 final.mp4 생성
# ──────────────────────────────────────────────
def merge_audio_video(video_path: str, audio_path: str, output_path: str) -> bool:
    """
    video_path  : 무음 영상  (예: static/브랜드_video.mp4)
    audio_path  : TTS 오디오 (예: static/narration_Nova.mp3)
    output_path : 합성 결과  (예: static/브랜드_final.mp4)

    -c:v libx264 + -pix_fmt yuv420p : Chrome/Edge 브라우저 호환 필수
    -shortest                        : 짧은 쪽 길이 기준으로 맞춤
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-shortest",
                output_path
            ],
            capture_output=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"[음성 합성 완료]: {output_path}")
            return True
        print(f"[음성 합성 실패]: {result.stderr.decode()}")
        return False
    except Exception as e:
        print(f"[음성 합성 오류]: {e}")
        return False


# ──────────────────────────────────────────────
# [6] 광고 생성 엔드포인트
# LangGraph 파이프라인 실행: 기획→검토→TTS→TTV
# ──────────────────────────────────────────────
@app.get("/create_ad")
def create_ad(
    brand:    str  = Query(...,                description="브랜드명"),
    concept:  str  = Query(...,                description="키워드/컨셉"),
    mood:     str  = Query(...,                description="무드/분위기"),
    target:   str  = Query("일반 소비자",      description="타겟 고객층"),
    style:    str  = Query("모던하고 깔끔한",  description="영상 스타일"),
    duration: str  = Query("15초",             description="영상 길이"),
    voice:    str  = Query("Nova",             description="나레이션 음성"),
    bgm:      bool = Query(False,              description="BGM 생성 여부"),
):
    try:
        result = generate_ad_with_graph({
            "brand": brand, "concept": concept, "mood": mood,
            "target": target, "style": style, "duration": duration,
            "voice": voice, "bgm": bgm
        })

        audio_path = result["audio_url"].lstrip("/")
        print(f"[DEBUG] 오디오 파일 존재: {os.path.exists(audio_path)} / {audio_path}")

        return {
            "status":  "processing",
            "message": "대본·나레이션·영상 생성 완료.",
            "generation_info": {
                "brand":          brand,
                "narration_text": result["pure_narration"],
                "style":          style,
                "duration":       duration,
                "voice":          voice,
                "bgm_included":   "예" if bgm else "아니오",
            },
            "ad_script_full":     result["script"],
            "audio_download_url": result["audio_url"],
            "stillcut_image_url": "",          # /check_video에서 첫 프레임 추출 후 반환
            "video_task_id":      result.get("task_id", ""),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# [7] 영상 완성 확인 엔드포인트
# 완성 시 음성 합성 → 첫 프레임 추출 → video_url + stillcut_url 반환
# ──────────────────────────────────────────────
@app.get("/check_video")
def check_video(
    task_id:   str,
    brand:     str = Query("ad",  description="브랜드명 (파일명 생성용)"),
    audio_url: str = Query("",    description="TTS 오디오 경로 (/static/narration_*.mp3)"),
):
    safe_brand  = brand.replace(" ", "_")
    audio_local = audio_url.lstrip("/") if audio_url else ""

    def process_video(video_local_path: str) -> dict:
        """영상이 준비됐을 때: 음성 합성 → 첫 프레임 추출 → 결과 반환"""
        final_video    = f"static/{safe_brand}_final.mp4"
        stillcut_local = f"static/{safe_brand}_stillcut.png"

        # 음성 합성 (오디오 파일이 있을 때만)
        if audio_local and os.path.exists(audio_local) and os.path.exists(video_local_path):
            merge_audio_video(video_local_path, audio_local, final_video)
            serve_video = final_video
        else:
            serve_video = video_local_path   # 오디오 없으면 무음 그대로

        # 첫 프레임 → 스틸컷 PNG 추출
        stillcut_url = ""
        if extract_first_frame(serve_video, stillcut_local):
            stillcut_url = f"/static/{safe_brand}_stillcut.png"

        return {
            "status":       "success",
            "message":      "영상 생성 완료!",
            "video_url":    "/" + serve_video.replace("\\", "/"),
            "stillcut_url": stillcut_url,
        }

    # LTX-Video 로컬 생성 결과 확인
    if task_id.startswith("LOCAL_") and task_id != "LOCAL_VIDEO_TASK_001":
        video_local = f"static/{safe_brand}_video.mp4"
        if not os.path.exists(video_local):
            return {"status": "generating", "message": "영상 생성 중..."}
        return process_video(video_local)

    # Mock: static/sample.mp4 사용 (개발 테스트용)
    if task_id == "LOCAL_VIDEO_TASK_001":
        video_local = "static/sample.mp4"
        if not os.path.exists(video_local):
            return {"status": "generating", "message": "static/sample.mp4 파일을 준비해주세요."}
        return process_video(video_local)

    return {"status": "generating", "message": "알 수 없는 task_id"}
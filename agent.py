# agent.py
#
# 흐름:
# 광고 기획(Claude) → 검토(Claude) → TTS(OpenAI) → 영상 생성(LTX-Video 로컬) → 종료
# 스틸컷: 영상 완성 후 첫 프레임을 ffmpeg로 자동 추출


# ──────────────────────────────────────────────
# [0] 환경변수 로드
# ──────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()


# ──────────────────────────────────────────────
# [1] 라이브러리 임포트
# ──────────────────────────────────────────────
import os
import re
import subprocess
from typing import TypedDict

from openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END


# ──────────────────────────────────────────────
# [2] LLM 클라이언트 초기화
# ──────────────────────────────────────────────
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
llm_claude = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7)
llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ──────────────────────────────────────────────
# [3] State 정의
# ──────────────────────────────────────────────
class State(TypedDict):
    brand: str
    concept: str
    mood: str
    target: str
    style: str
    duration: str
    voice: str
    bgm: bool

    script: str
    pure_narration: str
    review: str
    iterations: int

    audio_url: str
    video_prompt: str
    video_task_id: str


# ──────────────────────────────────────────────
# [4] LTX-Video 파이프라인 전역 관리
#
# CogVideoX-2B 대비 LTX-Video 장점 (RTX 3060 Laptop 기준):
#   CogVideoX : 스텝당 31초 × 20스텝 = 약 10분
#   LTX-Video : 스텝당  3초 × 20스텝 = 약 1~2분
#   VRAM 사용량도 낮아 6GB에서 안정적
# ──────────────────────────────────────────────
_ltx_pipe = None


def get_ltx_pipeline():
    """
    LTX-Video 파이프라인을 반환합니다.
    첫 호출 시에만 로드하고 이후에는 캐시를 반환합니다.

    필수 설치:
        pip install diffusers --upgrade
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        pip install transformers accelerate sentencepiece imageio-ffmpeg
    """
    global _ltx_pipe

    if _ltx_pipe is not None:
        return _ltx_pipe

    import torch
    from diffusers import LTXPipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 사용 불가. GPU 버전 torch를 설치하세요:\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(f"[LTX-Video] GPU: {gpu_name} / VRAM: {vram_gb} GB")
    print("[LTX-Video] 모델 로딩 중... (첫 실행 시 다운로드 약 5~10분, 이후 빠름)")

    _ltx_pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.bfloat16
    )

    # VRAM 6GB 최적화
    _ltx_pipe.enable_model_cpu_offload()  # GPU↔CPU 자동 이동
    _ltx_pipe.vae.enable_slicing()        # VAE 입력 분할 처리
    _ltx_pipe.vae.enable_tiling()         # VAE 타일 처리

    print("[LTX-Video] 로드 완료! GPU 연산 시작 준비.")
    return _ltx_pipe


# ──────────────────────────────────────────────
# [5] 광고 기획 노드 (Claude)
# ──────────────────────────────────────────────
def planner_node(state: State):
    print("--- [광고 기획자] 대본 구성 중... ---")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "당신은 세계 최고의 AI 광고 크리에이터입니다. "
                "사용자가 입력한 조건에 맞춰 광고 대본을 작성하세요.\n\n"
                "이모지, 이모티콘, 음표, 특수 문장 기호를 절대 포함하지 마십시오. "
                "오직 한국어 텍스트와 표준 문장 부호만 사용하세요.\n\n"
                "반드시 마지막에 아래 형식 포함:\n"
                "[NARRATION: 실제 나레이션]"
            )
        ),
        (
            "user",
            (
                "브랜드명: {brand}\n"
                "컨셉: {concept}\n"
                "무드: {mood}\n"
                "타겟: {target}\n"
                "스타일: {style}\n"
                "영상 길이: {duration}\n"
                "이전 피드백: {review}"
            )
        )
    ])

    chain = prompt | llm_claude
    response = chain.invoke({
        "brand":    state["brand"],
        "concept":  state["concept"],
        "mood":     state["mood"],
        "target":   state["target"],
        "style":    state["style"],
        "duration": state["duration"],
        "review":   state.get("review", "없음")
    })

    return {
        "script":     response.content,
        "iterations": state.get("iterations", 0) + 1
    }


# ──────────────────────────────────────────────
# [6] 검토 노드 (Claude)
# ──────────────────────────────────────────────
def reviewer_node(state: State):
    print("--- [검토자] 기획안 검토 중... ---")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "당신은 브랜드 디렉터입니다.\n"
                "이모지, 음표, 특수 기호가 하나라도 있으면 PASS 없이 수정 피드백을 주세요.\n"
                "문제 없으면 PASS만 출력하세요."
            )
        ),
        ("user", "광고 대본:\n{script}")
    ])

    chain = prompt | llm_claude
    response = chain.invoke({"script": state["script"]})
    return {"review": response.content}


# ──────────────────────────────────────────────
# [7] TTS 생성 노드 (OpenAI TTS)
# ──────────────────────────────────────────────
def tts_generation_node(state: State):
    print("--- [오디오 시스템] TTS 생성 중... ---")

    def remove_emojis(text: str) -> str:
        pattern = re.compile(
            "[\U00010000-\U0010FFFF\u200d\u2600-\u27BF]+",
            flags=re.UNICODE
        )
        return pattern.sub("", text)

    clean_script = remove_emojis(state.get("script", ""))

    # 정규식으로 [NARRATION: ...] 추출 (LLM 불필요, 빠르고 정확)
    narration_match = re.search(r"\[NARRATION:\s*(.+?)\]", clean_script, re.DOTALL)
    if narration_match:
        pure_text = narration_match.group(1).strip()
    else:
        # [NARRATION] 형식이 없으면 대본 마지막 줄을 사용
        lines = [l.strip() for l in clean_script.strip().split("\n") if l.strip()]
        pure_text = lines[-1] if lines else clean_script[:200]
    pure_text = remove_emojis(pure_text)

    print(f"[추출된 나레이션]: {pure_text}")

    os.makedirs("static", exist_ok=True)
    audio_filename = f"static/narration_{state['voice']}.mp3"

    tts_response = client.audio.speech.create(
        model="tts-1",
        voice=state["voice"].lower(),
        input=pure_text
    )
    tts_response.stream_to_file(audio_filename)
    print(f"[오디오 저장 완료]: {audio_filename}")

    return {
        "script":         clean_script,
        "pure_narration": pure_text,
        "audio_url":      f"/static/narration_{state['voice']}.mp3"
    }


# ──────────────────────────────────────────────
# [8] 영상 생성 노드 (LTX-Video 로컬 실행)
# ──────────────────────────────────────────────
def video_order_node(state: State):
    print("--- [영상 시스템] LTX-Video 영상 생성 시작 ---")

    # Step 1: Claude로 LTX-Video 최적화 영어 프롬프트 생성
    motion_prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a video director creating a short advertisement.\n"
                "Write a cinematic prompt in English. Maximum 2 sentences, under 120 words.\n"
                "Include: main subject, scene, camera movement, lighting, atmosphere.\n"
                "Return ONLY the prompt. No labels, no explanation."
            )
        ),
        (
            "user",
            "Ad Script:\n{script}\n\nBrand: {brand}\nStyle: {style}"
        )
    ])

    motion_extractor = motion_prompt_template | llm_claude
    video_prompt = motion_extractor.invoke({
        "script": state["script"],
        "brand":  state["brand"],
        "style":  state["style"]
    }).content.strip()

    print(f"[영상 프롬프트]: {video_prompt}")

    # Step 2: LTX-Video로 영상 생성
    safe_brand = state["brand"].replace(" ", "_")
    video_path = f"static/{safe_brand}_video.mp4"
    os.makedirs("static", exist_ok=True)

    try:
        import torch
        from diffusers.utils import export_to_video

        pipe = get_ltx_pipeline()

        print("[LTX-Video] 영상 생성 중... (약 1~2분 소요)")

        result = pipe(
            prompt=video_prompt,
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
            width=768,
            height=512,
            num_frames=121,          # 121프레임 @ 24fps = 약 5초
            num_inference_steps=20,  # 속도와 품질 균형
            guidance_scale=3.0,      # LTX 권장값
            generator=torch.Generator(device="cuda").manual_seed(42)
        )
        frames = result.frames[0]

        export_to_video(frames, video_path, fps=24)
        print(f"[영상 저장 완료]: {video_path}")

        return {
            "video_prompt":  video_prompt,
            "video_task_id": f"LOCAL_{safe_brand}"
        }

    except Exception as e:
        print(f"[영상 생성 오류]: {e}")
        return {
            "video_prompt":  video_prompt,
            "video_task_id": "LOCAL_VIDEO_TASK_001"
        }


# ──────────────────────────────────────────────
# [9] 영상 첫 프레임 추출 헬퍼
# ──────────────────────────────────────────────
def extract_first_frame(video_path: str, output_path: str) -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vframes", "1", "-q:v", "2", output_path],
            capture_output=True,
            timeout=30
        )
        success = result.returncode == 0
        if success:
            print(f"[스틸컷 추출 완료]: {output_path}")
        else:
            print(f"[스틸컷 추출 실패]: {result.stderr.decode()}")
        return success
    except Exception as e:
        print(f"[스틸컷 추출 오류]: {e}")
        return False


# ──────────────────────────────────────────────
# [10] 조건 분기
# ──────────────────────────────────────────────
def should_continue(state: State):
    if "PASS" in state["review"] or state["iterations"] >= 2:
        return "generate_tts"
    return "continue"


# ──────────────────────────────────────────────
# [11] LangGraph 워크플로우
# ──────────────────────────────────────────────
workflow = StateGraph(State)

workflow.add_node("planner",       planner_node)
workflow.add_node("reviewer",      reviewer_node)
workflow.add_node("tts_generator", tts_generation_node)
workflow.add_node("video_orderer", video_order_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {"continue": "planner", "generate_tts": "tts_generator"}
)
workflow.add_edge("tts_generator", "video_orderer")
workflow.add_edge("video_orderer", END)

app_graph = workflow.compile()


# ──────────────────────────────────────────────
# [12] FastAPI 진입 함수
# ──────────────────────────────────────────────
def generate_ad_with_graph(data: dict):
    initial_state = {
        "brand":    data["brand"],
        "concept":  data["concept"],
        "mood":     data["mood"],
        "target":   data["target"],
        "style":    data["style"],
        "duration": data["duration"],
        "voice":    data["voice"],
        "bgm":      data["bgm"],

        "script":         "",
        "pure_narration": "",
        "review":         "",
        "iterations":     0,
        "audio_url":      "",
        "video_prompt":   "",
        "video_task_id":  ""
    }

    final_output = app_graph.invoke(initial_state)

    return {
        "script":         final_output["script"],
        "pure_narration": final_output["pure_narration"],
        "audio_url":      final_output["audio_url"],
        "task_id":        final_output.get("video_task_id", "")
    }

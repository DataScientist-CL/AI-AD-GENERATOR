# AD.GEN

LangGraph 기반 AI 광고 영상 자동 생성 파이프라인. 브랜드 정보와 키워드를 입력받아 나레이션 스크립트 생성부터 TTS, 영상 생성, 음성 합성까지 단일 파이프라인으로 처리한다.

---

## Architecture

```
POST /create_ad
    └── LangGraph Pipeline
            ├── planner_node       # GPT-4o-mini, 광고 대본 생성
            ├── reviewer_node      # 품질 검토, PASS / 재작성 (max 2회)
            ├── tts_generation_node  # OpenAI TTS → narration_*.mp3
            └── video_generation_node  # LTX-Video (local GPU) → *_video.mp4

GET /check_video (polling, 5s interval)
    └── 완료 시
            ├── merge_audio_video()    # ffmpeg, 음성+영상 합성, H.264 인코딩
            └── extract_first_frame()  # ffmpeg, 0초 PNG 추출 → stillcut_url 반환
```

Whisper STT는 음성 입력을 텍스트로 변환하는 전처리 단계에서 사용된다.  
영상 생성 완료 전까지 프론트엔드는 5초 간격 폴링으로 상태를 확인한다.

---

## Stack

| Layer | Tech |
|---|---|
| API Server | FastAPI |
| Pipeline Orchestration | LangGraph |
| LLM | GPT-4o-mini |
| STT | OpenAI Whisper |
| TTS | OpenAI TTS (tts-1) |
| Video Generation | LTX-Video (local, Apache 2.0) |
| Video Processing | ffmpeg |
| ML Runtime | PyTorch 2.x + CUDA 12.1 |
| Frontend | Vanilla HTML/CSS/JS |

---

## Requirements

- GPU: NVIDIA RTX 3060 이상, VRAM 6GB 이상
- CUDA 12.1
- Python 3.10+
- ffmpeg (PATH 등록 필요)

> CogVideoX-2B는 VRAM 8GB 권장 모델로 6GB 환경에서 안정 실행 불가. LTX-Video로 대체.

---

## Setup

```bash
git clone https://github.com/YOUR_ID/ad-gen.git
cd ad-gen

# torch CPU 버전이 설치된 경우 반드시 재설치
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

`.env` 파일 생성:

```
OPENAI_API_KEY=your_key_here
```

```bash
uvicorn main:app --reload
# → http://127.0.0.1:8000
```

---

## Project Structure

```
ad-gen/
├── agent.py          # LangGraph 노드 정의 (planner / reviewer / tts / video)
├── main.py           # FastAPI 라우터, ffmpeg 처리 (merge, extract_first_frame)
├── .env
├── requirements.txt
├── templates/
│   └── index.html    # 프론트엔드 (커스텀 비디오 플레이어 포함)
└── static/
    ├── narration_*.mp3
    ├── *_video.mp4       # LTX-Video 원본 (무음, mpeg4/mp4v)
    ├── *_final.mp4       # 합성 완료, H.264/yuv420p
    └── *_stillcut.png
```

---

## Key Implementation Notes

**브라우저 재생 호환성**  
LTX-Video 출력 코덱은 `mpeg4/mp4v`로 Chrome/Edge 미지원. ffmpeg 재인코딩 시 `-c:v libx264 -pix_fmt yuv420p` 필수.

**나레이션 추출**  
LLM 응답에서 나레이션만 분리할 때 LLM 재호출 방식은 불안정. `[NARRATION: ...]` 패턴을 정규식(`re.search`)으로 직접 파싱하고, 실패 시 마지막 줄을 폴백으로 사용.

**스틸컷**  
별도 이미지 생성 API 없이 `ffmpeg -ss 0 -vframes 1`로 첫 프레임 추출. 0초가 검은 프레임인 경우 프론트엔드에서 `video.src`에 `#t=0.1` 추가로 회피.

**VAE 메모리 최적화**  
6GB VRAM 환경 기준:
```python
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

---

## Dependency Version Lock

transformers/diffusers 버전 충돌로 인해 아래 버전을 고정한다.

```
transformers==4.44.2
diffusers==0.31.0
sentencepiece  # T5 토크나이저 의존성
```

---

## Troubleshooting

| 증상 | 원인 | 해결 |
|---|---|---|
| `torch.cuda.is_available()` → False | torch CPU 버전 설치 | cu121 버전으로 재설치 |
| T5 토크나이저 로딩 오류 | transformers/diffusers 버전 충돌 | 버전 고정 (위 참조) |
| `No module named sentencepiece` | 패키지 누락 | `pip install sentencepiece` |
| 브라우저 재생 불가 (검은 화면) | mp4v 코덱 미지원 | libx264 + yuv420p 재인코딩 |
| 영상에 음성 없음 | 파일 분리 저장 | `merge_audio_video()` 추가 |
| 나레이션에 대본 전체 출력 | LLM 추출 실패 | 정규식 파싱으로 교체 |

---

## Roadmap

- [ ] BGM 자동 생성
- [ ] `num_inference_steps` 튜닝 (현재 20, 30 비교 예정)
- [ ] 클라우드 GPU 배포 전환

---

## License

- LTX-Video: Apache 2.0
- 본 프로젝트: MIT

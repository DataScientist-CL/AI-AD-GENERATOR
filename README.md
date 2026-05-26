AD.GEN 🎬
AI 기반 광고 영상 자동 생성 파이프라인

브랜드명과 키워드만 입력하면, 나레이션 스크립트 → 음성 합성 → 광고 영상 생성까지 자동으로 완성되는 AI 에이전트입니다.


📌 프로젝트 개요
소상공인·1인 창업자가 전문 지식 없이도 광고 영상을 제작할 수 있도록,
LangGraph 기반 멀티 노드 파이프라인으로 전 과정을 자동화합니다.

✅ 현재 구현 완료 기능
기능설명사용자 입력 폼 UI브랜드명, 키워드, 타겟, 스타일, 음성 옵션 입력나레이션 스크립트 생성GPT-4o-mini 기반 광고 대본 자동 생성스크립트 품질 검토reviewer 노드가 PASS / 재작성 판단 (최대 2회)Whisper STT음성 입력을 텍스트로 변환TTS 음성 합성OpenAI TTS(Nova 등)로 나레이션 mp3 생성 및 미리듣기광고 영상 생성LTX-Video (로컬 GPU) 로 무음 mp4 생성음성·영상 합성ffmpeg으로 나레이션 + 영상 합성 → _final.mp4브라우저 재생H.264 + yuv420p 재인코딩으로 Chrome/Edge 호환스틸컷 추출ffmpeg으로 영상 첫 프레임 PNG 자동 추출커스텀 영상 플레이어반투명 컨트롤바, 진행바 포함 인라인 미리보기다운로드브랜드명_YYYY-MM-DD.mp4 형식으로 저장

🔄 파이프라인 흐름
사용자 입력 폼
(브랜드명 · 키워드 · 스타일 · 음성)
        │
        ▼
  [ Whisper STT ]         ← 음성 입력 시 텍스트 변환
        │
        ▼
  [ planner_node ]        ← GPT-4o-mini · 광고 대본 생성
        │
        ▼
  [ reviewer_node ]       ← 품질 검토 · PASS 또는 재작성 (최대 2회)
        │ PASS
        ▼
  [ tts_generation_node ] ← OpenAI TTS · narration_*.mp3 저장
        │
        ▼
  [ video_generation_node ] ← LTX-Video (로컬 GPU) · *_video.mp4
        │
        ▼
  [ merge_audio_video ]   ← ffmpeg · 음성+영상 합성 · H.264 인코딩
        │
        ▼
  [ extract_first_frame ] ← ffmpeg · 0초 지점 PNG 추출 (스틸컷)
        │
        ▼
  결과 반환 (프론트엔드)
  나레이션 텍스트 · 스틸컷 · 영상 플레이어 · 다운로드 버튼

🛠 기술 스택
분류사용 기술백엔드 프레임워크FastAPIAI 파이프라인LangGraphLLMGPT-4o-mini (스크립트 생성·검토)STTOpenAI WhisperTTSOpenAI TTS (tts-1)영상 생성LTX-Video (로컬 GPU · Apache 2.0)영상 처리ffmpeg (합성·인코딩·프레임 추출)딥러닝 런타임PyTorch + CUDA 12.1프론트엔드HTML / CSS / JavaScript (Vanilla)환경 관리python-dotenv

💻 실행 환경
항목사양OSWindows 10/11GPUNVIDIA RTX 3060 Laptop (VRAM 6GB)CUDA12.1Python3.10+주요 라이브러리 버전transformers==4.44.2 / diffusers==0.31.0

📁 프로젝트 구조
ad-gen/
├── agent.py            # LangGraph 파이프라인 노드 정의
├── main.py             # FastAPI 서버 (엔드포인트 · ffmpeg 처리)
├── .env                # API 키 (OPENAI_API_KEY)
├── requirements.txt
├── templates/
│   └── index.html      # 프론트엔드 UI
└── static/             # 생성 파일 저장
    ├── narration_*.mp3     # TTS 음성
    ├── *_video.mp4         # LTX-Video 원본 (무음)
    ├── *_final.mp4         # 음성 합성 + H.264 완성본
    └── *_stillcut.png      # 첫 프레임 스틸컷

⚙️ 설치 및 실행
bash# 1. 저장소 클론
git clone https://github.com/YOUR_ID/ad-gen.git
cd ad-gen

# 2. 패키지 설치
pip install -r requirements.txt

# 3. PyTorch GPU 버전 설치 (CPU 버전이 설치된 경우 재설치 필요)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. 환경 변수 설정
# .env 파일 생성 후 아래 내용 입력
OPENAI_API_KEY=your_openai_api_key_here

# 5. 서버 실행
uvicorn main:app --reload

# 6. 브라우저 접속
http://127.0.0.1:8000

🐛 주요 트러블슈팅
증상원인해결torch.cuda.is_available() → Falsetorch CPU 버전 설치됨GPU 버전(cu121)으로 재설치T5 토크나이저 로딩 오류transformers / diffusers 버전 충돌transformers==4.44.2 / diffusers==0.31.0sentencepiece 없음패키지 미설치pip install sentencepiece영상에 음성 없음파일 분리 저장ffmpeg 합성 함수 추가브라우저 재생 불가 (검은 화면)mpeg4/mp4v 코덱 미지원libx264 + yuv420p 재인코딩나레이션에 대본 전체 출력LLM 추출 실패정규식(re.search) 직접 파싱으로 교체

🗺 향후 계획

 BGM 자동 생성 기능 추가
 num_inference_steps 최적값 탐색 (20 vs 30 품질 비교)
 다양한 브랜드·스타일 품질 테스트
 클라우드 GPU 환경 배포 전환 검토


📄 라이선스
LTX-Video 모델: Apache 2.0
본 프로젝트 코드: MIT

AI Agent 기반 실무형 개발자 양성과정 | 2026

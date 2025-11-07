# 원클릭 배포 가이드

이 프로젝트를 웹에 배포하는 방법을 안내합니다. **Railway**와 **Render** 두 가지 플랫폼을 지원합니다.

## 🚀 방법 1: Render 배포 (추천)

Render는 Docker Compose를 지원하며, 무료 티어를 제공합니다.

### 1단계: Render 계정 생성
1. [Render.com](https://render.com)에 가입
2. GitHub 계정으로 로그인

### 2단계: 새 Blueprint 배포
1. Render 대시보드에서 **"New +"** 클릭
2. **"Blueprint"** 선택
3. GitHub 저장소 연결
4. 저장소에서 `render.yaml` 파일을 자동으로 감지
5. **"Apply"** 클릭

### 3단계: 환경 변수 설정
배포 후 각 서비스의 환경 변수를 설정하세요:

**Backend 서비스:**
- `EMAIL_ADDRESS`: Gmail 주소
- `EMAIL_PASSWORD`: Gmail 앱 비밀번호
- `JWT_SECRET_KEY`: 최소 32자 이상의 랜덤 문자열 (자동 생성됨)
- `OPENAI_API_KEY`: OpenAI API 키

**Frontend 서비스:**
- `API_BASE_URL`: 백엔드 서비스 URL (예: `https://leareng-backend.onrender.com/api`)

### 4단계: 배포 완료
배포가 완료되면 각 서비스에 고유한 URL이 생성됩니다:
- Frontend: `https://leareng-frontend.onrender.com`
- Backend: `https://leareng-backend.onrender.com`

---

## 🚂 방법 2: Railway 배포

Railway는 간단한 UI로 빠르게 배포할 수 있습니다.

### 1단계: Railway 계정 생성
1. [Railway.app](https://railway.app)에 가입
2. GitHub 계정으로 로그인

### 2단계: 프로젝트 생성
1. Railway 대시보드에서 **"New Project"** 클릭
2. **"Deploy from GitHub repo"** 선택
3. 저장소 선택

### 3단계: 서비스 추가

#### MySQL 데이터베이스 추가
1. **"New"** → **"Database"** → **"Add MySQL"** 선택
2. 데이터베이스가 자동으로 생성됩니다

#### Backend 서비스 추가
1. **"New"** → **"GitHub Repo"** 선택
2. 저장소 선택
3. **Root Directory**를 `backend`로 설정
4. **Dockerfile Path**를 `backend/Dockerfile`로 설정
5. 환경 변수 설정:
   ```
   SPRING_DATASOURCE_URL=jdbc:mysql://${{MySQL.MYSQLHOST}}:${{MySQL.MYSQLPORT}}/${{MySQL.MYSQLDATABASE}}?useSSL=false&serverTimezone=UTC
   DB_USERNAME=${{MySQL.MYSQLUSER}}
   DB_PASSWORD=${{MySQL.MYSQLPASSWORD}}
   EMAIL_ADDRESS=your-email@gmail.com
   EMAIL_PASSWORD=your-app-password
   JWT_SECRET_KEY=your-secret-key-min-32-characters
   OPENAI_API_KEY=your-openai-api-key
   ```
6. **"Deploy"** 클릭

#### Frontend 서비스 추가
1. **"New"** → **"GitHub Repo"** 선택
2. 저장소 선택
3. **Root Directory**를 `frontend`로 설정
4. **Dockerfile Path**를 `frontend/Dockerfile`로 설정
5. 환경 변수 설정:
   ```
   API_BASE_URL=https://your-backend-url.railway.app/api
   ```
6. **"Deploy"** 클릭

### 4단계: 도메인 설정 (선택사항)
각 서비스에서 **"Settings"** → **"Generate Domain"** 클릭하여 공개 URL 생성

---

## 🔧 방법 3: Docker Compose 직접 배포

VPS나 클라우드 서버에 직접 배포할 수 있습니다.

### 1단계: 서버 준비
```bash
# Docker 및 Docker Compose 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

### 2단계: 프로젝트 클론
```bash
git clone <your-repo-url>
cd gonzalolearing
```

### 3단계: 환경 변수 설정
`.env` 파일 생성:
```env
EMAIL_ADDRESS=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
JWT_SECRET_KEY=your-secret-key-min-32-characters
OPENAI_API_KEY=your-openai-api-key
```

### 4단계: 배포
```bash
docker-compose up -d
```

### 5단계: Nginx 리버스 프록시 설정 (선택사항)
도메인을 사용하는 경우 Nginx를 리버스 프록시로 설정:

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location /api {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://localhost:80;
        proxy_set_header Host $host;
    }
}
```

---

## 📝 환경 변수 설명

| 변수명 | 설명 | 필수 |
|--------|------|------|
| `EMAIL_ADDRESS` | Gmail 주소 (이메일 인증용) | ✅ |
| `EMAIL_PASSWORD` | Gmail 앱 비밀번호 | ✅ |
| `JWT_SECRET_KEY` | JWT 토큰 서명 키 (최소 32자) | ✅ |
| `OPENAI_API_KEY` | OpenAI API 키 | ✅ |
| `API_BASE_URL` | 백엔드 API URL (프론트엔드만) | ✅ |

### Gmail 앱 비밀번호 생성 방법
1. Google 계정 설정 → 보안
2. 2단계 인증 활성화
3. 앱 비밀번호 생성
4. 생성된 비밀번호를 `EMAIL_PASSWORD`에 사용

---

## 🔍 문제 해결

### 백엔드가 시작되지 않음
- 데이터베이스 연결 확인
- 환경 변수가 올바르게 설정되었는지 확인
- 로그 확인: `docker-compose logs backend`

### 프론트엔드에서 API 호출 실패
- CORS 설정 확인
- `API_BASE_URL` 환경 변수 확인
- 브라우저 콘솔에서 에러 확인

### 데이터베이스 연결 실패
- 데이터베이스가 실행 중인지 확인
- 연결 문자열 확인
- 방화벽 설정 확인

---

## 💡 팁

1. **프로덕션 환경에서는 반드시 강력한 `JWT_SECRET_KEY`를 사용하세요**
2. **HTTPS를 사용하도록 설정하세요** (Let's Encrypt 무료 인증서 사용 가능)
3. **정기적으로 데이터베이스 백업을 수행하세요**
4. **환경 변수는 절대 코드에 커밋하지 마세요**

---

## 📞 지원

문제가 발생하면 이슈를 등록하거나 다음으로 문의하세요:
- 전화: 010-9493-6576


// API Base URL - 배포 환경에 따라 자동 감지
const API_BASE_URL = (() => {
    // 1. 메타 태그에서 API URL 확인 (배포 시 주입 가능)
    const metaApiUrl = document.querySelector('meta[name="api-base-url"]');
    if (metaApiUrl && metaApiUrl.content) {
        return metaApiUrl.content;
    }
    
    // 2. window 객체에 설정된 경우 (런타임 주입)
    if (window.API_BASE_URL) {
        return window.API_BASE_URL;
    }
    
    // 3. 프로덕션 환경 감지
    const hostname = window.location.hostname;
    if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
        // Render, Railway 등에서 백엔드가 다른 서브도메인에 있는 경우
        // 환경 변수로 백엔드 URL을 설정하거나
        // 같은 포트에서 프록시를 사용하는 경우를 대비
        
        // 같은 호스트의 /api 경로 사용 (Nginx 프록시 설정 시)
        if (window.location.pathname === '/' || window.location.pathname.startsWith('/index')) {
            return '/api';
        }
        
        // 백엔드가 별도 도메인인 경우 (예: backend.onrender.com)
        // 이 경우 환경 변수로 설정해야 함
        // 기본값으로 같은 호스트의 8080 포트 시도
        const protocol = window.location.protocol;
        const port = window.location.port ? `:${window.location.port}` : '';
        return `${protocol}//${hostname}${port}/api`;
    }
    
    // 4. 로컬 개발 환경
    return 'http://localhost:8080/api';
})();

class ApiClient {
    constructor() {
        this.baseUrl = API_BASE_URL;
    }

    getToken() {
        return localStorage.getItem('token');
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const token = this.getToken();
        
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };

        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const config = {
            ...options,
            headers
        };

        try {
            const response = await fetch(url, config);
            
            // Handle 401 Unauthorized
            if (response.status === 401) {
                localStorage.removeItem('token');
                localStorage.removeItem('userEmail');
                localStorage.removeItem('isAdmin');
                window.location.href = 'login.html';
                throw new Error('인증이 만료되었습니다. 다시 로그인해주세요.');
            }

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || data.message || `요청 실패 (${response.status})`);
            }
            
            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    }

    async post(endpoint, body) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(body)
        });
    }

    async postFormData(endpoint, formData) {
        const url = `${this.baseUrl}${endpoint}`;
        const token = this.getToken();
        
        const headers = {};
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers,
                body: formData
            });

            // Handle 401 Unauthorized
            if (response.status === 401) {
                localStorage.removeItem('token');
                localStorage.removeItem('userEmail');
                localStorage.removeItem('isAdmin');
                window.location.href = 'login.html';
                throw new Error('인증이 만료되었습니다. 다시 로그인해주세요.');
            }

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || data.message || `요청 실패 (${response.status})`);
            }
            
            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }
}

const api = new ApiClient();


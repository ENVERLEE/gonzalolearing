// Main app initialization
document.addEventListener('DOMContentLoaded', () => {
    if (!authManager.isAuthenticated()) {
        window.location.href = 'login.html';
        return;
    }
    
    // Show user info
    const userEmail = authManager.userEmail;
    document.getElementById('userEmail').textContent = userEmail;
    document.getElementById('userInfo').style.display = 'flex';

    // Logout handler
    document.getElementById('logoutBtn').addEventListener('click', () => {
        authManager.logout();
    });

    // Mode tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.mode-content').forEach(c => c.style.display = 'none');
            
            btn.classList.add('active');
            const mode = btn.dataset.mode;
            document.getElementById(mode + 'Mode').style.display = 'block';
        });
    });

    // Load subscription status
    loadSubscriptionStatus();

    // Load saved passages
    loadSavedPassages();

    // Add admin link if admin
    if (authManager.isAdmin) {
        const userInfo = document.getElementById('userInfo');
        const adminLink = document.createElement('a');
        adminLink.href = 'admin.html';
        adminLink.textContent = '관리자';
        adminLink.style.marginRight = '10px';
        adminLink.style.color = '#007AFF';
        userInfo.insertBefore(adminLink, userInfo.firstChild);
    }
});

async function loadSubscriptionStatus() {
    try {
        const status = await api.get('/subscription/status');
        const statusEl = document.getElementById('subscriptionStatus');
        
        if (status.isSubscribed) {
            statusEl.innerHTML = `
                <div class="message success">
                    <p>🌟 프리미엄 구독 활성화</p>
                    <p>구독 만료일: ${new Date(status.expiryDate).toLocaleDateString()}</p>
                </div>
            `;
        } else {
            statusEl.innerHTML = `
                <div class="message info">
                    <p>무료 버전 (남은 문제 수: ${status.remainingQuestions})</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Failed to load subscription status:', error);
    }
}

async function loadSavedPassages() {
    try {
        const passages = await api.get('/passages');
        const container = document.getElementById('savedPassages');
        
        if (passages.length === 0) {
            container.innerHTML = '<p class="message info">저장된 지문이 없습니다.</p>';
            return;
        }

        container.innerHTML = passages.map(passage => `
            <div class="passage-card">
                <h4>${passage.title}</h4>
                <p>${passage.text.substring(0, 200)}...</p>
                <button class="btn-primary" onclick="generateQuestionsForPassage(${passage.id})">문제 생성</button>
            </div>
        `).join('');
    } catch (error) {
        console.error('Failed to load passages:', error);
    }
}

async function generateQuestionsForPassage(passageId) {
    showLoading('문제를 생성하고 있습니다...');
    try {
        const passages = await api.get('/passages');
        const passage = passages.find(p => p.id === passageId);
        
        if (!passage) {
            throw new Error('지문을 찾을 수 없습니다.');
        }

        await api.post('/questions/generate', {
            passageId: passageId,
            text: passage.text,
            title: passage.title
        });

        hideLoading();
        alert('문제 생성이 완료되었습니다!');
        loadSavedPassages();
    } catch (error) {
        hideLoading();
        alert('오류: ' + error.message);
    }
}

function showLoading(text = '처리 중...') {
    const overlay = document.getElementById('loadingOverlay');
    overlay.querySelector('.loading-text').textContent = text;
    overlay.style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}


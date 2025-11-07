// PDF upload handler
if (document.getElementById('pdfUpload')) {
    document.getElementById('pdfUpload').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        showLoading('PDF 파일을 분석하고 있습니다...');
        const container = document.getElementById('pdfPassages');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await api.postFormData('/pdf/upload', formData);
            hideLoading();

            container.innerHTML = response.passages.map((passage, idx) => `
                <div class="passage-card">
                    <h4>지문 ${idx + 1}</h4>
                    <textarea id="pdfPassage${idx}" rows="10">${passage}</textarea>
                    <input type="text" id="pdfTitle${idx}" placeholder="지문 제목">
                    <button class="btn-primary" onclick="savePdfPassage(${idx}, \`${passage.replace(/`/g, '\\`')}\`)">저장</button>
                    <button class="btn-primary" onclick="generateFromPdf(${idx}, \`${passage.replace(/`/g, '\\`')}\`)">문제 생성</button>
                </div>
            `).join('');
        } catch (error) {
            hideLoading();
            container.innerHTML = `<div class="message error">오류: ${error.message}</div>`;
        }
    });
}

async function savePdfPassage(idx, passageText) {
    const title = document.getElementById(`pdfTitle${idx}`).value;
    if (!title) {
        alert('제목을 입력해주세요.');
        return;
    }

    showLoading('지문을 저장하고 있습니다...');
    try {
        await api.post('/passages', { title, text: passageText });
        hideLoading();
        alert('저장되었습니다!');
        loadSavedPassages();
    } catch (error) {
        hideLoading();
        alert('오류: ' + error.message);
    }
}

async function generateFromPdf(idx, passageText) {
    const title = document.getElementById(`pdfTitle${idx}`).value;
    if (!title) {
        alert('제목을 입력해주세요.');
        return;
    }

    showLoading('문제를 생성하고 있습니다...');
    try {
        await api.post('/questions/generate', { text: passageText, title });
        hideLoading();
        alert('문제 생성이 완료되었습니다!');
        loadSavedPassages();
    } catch (error) {
        hideLoading();
        alert('오류: ' + error.message);
    }
}

// Manual passage handlers
if (document.getElementById('savePassageBtn')) {
    document.getElementById('savePassageBtn').addEventListener('click', async () => {
        const text = document.getElementById('manualPassage').value;
        const title = document.getElementById('manualTitle').value;

        if (!title) {
            alert('제목을 입력해주세요.');
            return;
        }

        if (!text) {
            alert('지문 내용을 입력해주세요.');
            return;
        }

        showLoading('지문을 저장하고 있습니다...');
        try {
            await api.post('/passages', { title, text });
            hideLoading();
            alert('저장되었습니다!');
            document.getElementById('manualPassage').value = '';
            document.getElementById('manualTitle').value = '';
            loadSavedPassages();
        } catch (error) {
            hideLoading();
            alert('오류: ' + error.message);
        }
    });
}

if (document.getElementById('generateQuestionsBtn')) {
    document.getElementById('generateQuestionsBtn').addEventListener('click', async () => {
        const text = document.getElementById('manualPassage').value;
        const title = document.getElementById('manualTitle').value;

        if (!title) {
            alert('제목을 입력해주세요.');
            return;
        }

        if (!text) {
            alert('지문 내용을 입력해주세요.');
            return;
        }

        showLoading('문제를 생성하고 있습니다...');
        try {
            await api.post('/questions/generate', { text, title });
            hideLoading();
            alert('문제 생성이 완료되었습니다!');
            loadSavedPassages();
        } catch (error) {
            hideLoading();
            alert('오류: ' + error.message);
        }
    });
}

// Subscription request handler
if (document.getElementById('subscribeBtn')) {
    document.getElementById('subscribeBtn').addEventListener('click', async () => {
        if (!confirm('프리미엄 구독을 신청하시겠습니까?')) {
            return;
        }

        try {
            await api.post('/subscription/request', {});
            alert('구독 신청이 완료되었습니다. 입금 확인 후 승인됩니다.');
            loadSubscriptionStatus();
        } catch (error) {
            alert('오류: ' + error.message);
        }
    });
}


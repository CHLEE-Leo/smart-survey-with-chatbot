const conditions = ['천식·세기관지염', '알레르기비염', '알레르기결막염', '만성두드러기', '아토피피부염', '약물알레르기'];
const pages = [
  {key: 'A', title: '기본정보·환경', fields: `
    <label>자녀 이름 초성<input name="child_initials" placeholder="예: ㄱㅁㅅ"></label>
    <div class="grid"><label>생년월일<input type="date" name="birth_date"></label><label>성별<select name="sex"><option>모름·응답하지 않음</option><option>남</option><option>여</option></select></label></div>
    <div class="grid"><label>키 (cm)<input name="height_cm" type="number" min="0" step="0.1"></label><label>체중 (kg)<input name="weight_kg" type="number" min="0" step="0.1"></label></div>
    <div class="grid"><label>출생 주수<input name="gestational_age" placeholder="예: 38주"></label><label>출생 체중 (kg)<input name="birth_weight_kg" type="number" min="0" step="0.01"></label></div>
    <label>자녀를 포함한 형제자매 수<input name="sibling_count_including_child" type="number" min="1"></label>
    <label>수유력<textarea name="feeding_history" placeholder="모유·분유·혼합 및 기간"></textarea></label>
    <label>이유식과 계란흰자·대두·밀·땅콩·견과류 도입 시기<textarea name="food_introduction"></textarea></label>
    <label>개·고양이 노출, 간접흡연, 어린이집 등 집단생활<textarea name="environment"></textarea></label>`},
  {key: 'B', title: '질환력', fields: conditions.map((name, i) => `<label>${name}<select name="condition_${i}"><option>모름</option><option>현재 있음</option><option>과거에 있었으나 현재 없음</option><option>없음</option></select></label>`).join('')},
  {key: 'C', title: '가족력', fields: ['어머니', '아버지', '형제자매'].map((name, i) => `<label>${name}의 알레르기 질환<textarea name="family_${i}" placeholder="천식·비염·아토피피부염·식품알레르기, 없음 또는 모름"></textarea></label>`).join('')},
  {key: 'D', title: '의심 식품', fields: '<label>반응이 의심되었던 음식<input name="suspected_foods" placeholder="예: 계란, 우유, 땅콩"></label><p class="hint">쉼표로 구분해 주세요. 의심 식품이 없으면 비워두고, 원인을 모르면 “원인 미상”이라고 적어주세요.</p>'}
];
const $ = selector => document.querySelector(selector);
let page = 0, profile = {}, sessionId = null, questionId = null;

function renderPage() {
  $('#steps').innerHTML = pages.map((p, i) => `<span class="${i === page ? 'active' : i < page ? 'done' : ''}">${p.key}<small>${p.title}</small></span>`).join('');
  $('#fields').innerHTML = `<p class="eyebrow">SECTION ${pages[page].key}</p><h2>${pages[page].title}</h2>${pages[page].fields}`;
  $('#fields').querySelectorAll('[name]').forEach(el => { if (el.name in profile) el.value = profile[el.name]; });
  $('#back').style.visibility = page ? 'visible' : 'hidden';
  $('#next').textContent = page === 3 ? '프로필 확인' : '다음';
}
function savePage() {
  if (!$('#profileForm').reportValidity()) return false;
  $('#fields').querySelectorAll('[name]').forEach(el => { profile[el.name] = el.value; });
  return true;
}
async function request(url, body) {
  const response = await fetch(url, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {});
  const data = await response.json();
  if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : '입력과 서버 연결을 확인해 주세요.');
  return data;
}
function bubble(text, role) {
  const el = document.createElement('div');
  el.className = `bubble ${role}`; el.textContent = text;
  $('#messages').appendChild(el); $('#messages').scrollTop = $('#messages').scrollHeight;
}
function handle(data) {
  if (data.done) {
    $('#chat').classList.add('hidden'); $('#done').classList.remove('hidden');
    $('#result').replaceChildren();
    const status = {sufficient: '설정된 연구 분류에 필요한 근거를 모았습니다.', max_turns: '질문 한도에 도달했습니다. 추가 확인이 필요합니다.'};
    const names = {symptoms: '증상', latency: '발현 시간', repeatability: '반복·재섭취', current_intake: '현재 섭취'};
    for (const item of data.summaries) {
      const heading = document.createElement('h3'); heading.textContent = item.food_id;
      const detail = document.createElement('p'); detail.textContent = status[item.termination] || item.termination;
      const missing = document.createElement('p'); missing.textContent = `미확인: ${item.missing_information.map(x => names[x] || x).join(', ') || '없음'}`;
      const record = document.createElement('pre'); record.textContent = item.messages.map(m => `${m.role === 'user' ? '답변' : '질문'}: ${m.content}`).join('\n\n');
      $('#result').append(heading, detail, missing, record);
    }
    if (!data.summaries.length) $('#result').textContent = data.message;
    return;
  }
  const changed = questionId !== data.question_id;
  questionId = data.question_id;
  $('#foodLabel').textContent = data.progress.food_id;
  $('#progress').textContent = `${data.progress.food_index}번째 식품 / ${data.progress.food_count}개 · ${data.progress.turns}회 응답`;
  if (changed) bubble(data.message, 'coach');
  $('#answer').value = ''; $('#answer').focus();
}
$('#next').onclick = () => {
  if (!savePage()) return;
  if (page < 3) { page++; renderPage(); return; }
  const labels = {};
  for (const p of pages) {
    const container = document.createElement('div'); container.innerHTML = p.fields;
    container.querySelectorAll('[name]').forEach(el => { labels[el.name] = el.closest('label').firstChild.textContent; });
  }
  $('#profileSummary').textContent = Object.entries(profile).filter(([, value]) => value).map(([key, value]) => `${labels[key]}: ${value}`).join('\n');
  $('#wizard').classList.add('hidden'); $('#review').classList.remove('hidden');
};
$('#back').onclick = () => { if (savePage()) { page--; renderPage(); } };
$('#editProfile').onclick = () => { $('#review').classList.add('hidden'); $('#wizard').classList.remove('hidden'); };
$('#startInterview').onclick = async () => {
  $('#startInterview').disabled = true;
  try {
    const foods = [...new Set((profile.suspected_foods || '').split(',').map(x => x.trim()).filter(Boolean))];
    const data = await request('/api/start', {profile: {...profile, suspected_foods: foods}});
    sessionId = data.session_id;
    $('#review').classList.add('hidden'); $('#chat').classList.remove('hidden'); handle(data);
  } catch (error) { $('#reviewError').textContent = error.message; }
  finally { $('#startInterview').disabled = false; }
};
$('#answerForm').onsubmit = async event => {
  event.preventDefault();
  const value = $('#answer').value.trim(), sentQuestionId = questionId;
  if (!value) return;
  $('#sendAnswer').disabled = true;
  try {
    const data = await request('/api/answer', {session_id: sessionId, question_id: sentQuestionId, value});
    bubble(value, 'user'); $('#chatError').textContent = ''; handle(data);
  } catch (error) {
    $('#chatError').textContent = error.message;
    // The answer can commit before generation/network failure. Recover the
    // current question before allowing a retry of the previous answer.
    try {
      const recovered = await request(`/api/session?session_id=${encodeURIComponent(sessionId)}`);
      if (recovered.done || recovered.question_id !== sentQuestionId) { bubble(value, 'user'); handle(recovered); }
    } catch (_) { /* Preserve the answer and error for a later retry. */ }
  } finally { $('#sendAnswer').disabled = false; }
};
request('/api/config').then(data => {
  if (data.offline_demo) $('#runtimeNotice').textContent = '화면 테스트 모드입니다. 자유로운 답변을 이해하는 LLM은 연결되지 않았습니다.';
}).catch(error => { $('#runtimeNotice').textContent = error.message; });
renderPage();

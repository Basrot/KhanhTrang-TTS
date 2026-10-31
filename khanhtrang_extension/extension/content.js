(() => {
  if (window.__ktReaderLoaded) return;
  window.__ktReaderLoaded = true;
  const host = document.createElement('div');
  host.id = 'kt-news-reader-root';
  host.style.cssText = 'position:fixed;bottom:20px;right:20px;z-index:2147483647;pointer-events:none';
  const shadow = host.attachShadow({mode:'closed'});
  shadow.innerHTML = `
  <style>
    *{box-sizing:border-box}
    button{cursor:pointer;font:inherit}
    button:focus-visible{outline:2px solid #2463b4;outline-offset:2px}
    .shell{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;color:#252b33;pointer-events:auto}
    .launcher{margin-left:auto;display:flex;gap:8px;align-items:center;background:#fff;color:#20252d;border:1px solid #d9dee5;border-radius:12px;padding:11px 15px;box-shadow:0 5px 20px #0002;font-weight:600}
    .launcher:hover{background:#f5f7f9}
    .dot{height:9px;width:9px;background:#29745a;border-radius:50%}
    .panel{display:none;flex-direction:column;width:min(380px,calc(100vw - 28px));max-height:min(660px,calc(100vh - 75px));overflow:hidden;background:white;border:1px solid #dce1e6;border-radius:15px;box-shadow:0 12px 45px #0003;margin-bottom:10px}
    .panel.open{display:flex}
    .head{display:flex;justify-content:space-between;gap:10px;padding:17px 18px;border-bottom:1px solid #edf0f3}
    h2{font-size:17px;margin:0 0 3px;font-weight:700;line-height:1.4;color:#1e2935}
    .sub{margin:0;color:#697583;font-size:12px}
    .close{background:transparent;border:0;color:#66717e;font-size:23px;line-height:1;align-self:flex-start}
    .body{padding:17px 18px;overflow:auto}
    .label{font-size:11px;color:#77818e;font-weight:700;margin:0 0 7px}
    .title{font-weight:650;font-size:13px;line-height:1.55;max-height:66px;overflow:hidden}
    .domain{font-size:11px;color:#87919b;margin:5px 0 16px}
    .primary{width:100%;background:#245c9d;color:#fff;border:0;border-radius:9px;padding:12px;font-weight:700;font-size:13px}
    .primary:hover{background:#1b4d88}
    .primary:disabled{opacity:.65;cursor:default}
    .status{font-size:12px;color:#65717d;margin:12px 0 0;white-space:pre-wrap}
    .error{color:#b13d3d}
    .progress{display:none;margin-top:13px}
    .progress.show{display:block}
    .progress-row{display:flex;justify-content:space-between;font-size:12px;color:#536172;margin-bottom:7px}
    .track{height:9px;border-radius:100px;background:#e7edf4;overflow:hidden}
    .fill{height:100%;width:0%;background:#245c9d;border-radius:100px;transition:width .4s ease}
    .progress-note{margin-top:7px;font-size:11px;color:#7b8793}
    .results{display:none;margin-top:15px;border-top:1px solid #edf0f3;padding-top:14px}
    .results.show{display:block}
    .summary{font-size:13px;line-height:1.7;white-space:pre-wrap;max-height:195px;overflow:auto;background:#f6f7f9;border:1px solid #e8ecf0;border-radius:8px;padding:12px}
    audio{width:100%;height:42px;margin:4px 0 14px}
    .auto{display:flex;align-items:center;gap:9px;font-size:12px;color:#4b5563;margin-top:13px;cursor:pointer}
    .auto input{accent-color:#245c9d;width:16px;height:16px}
    .foot{font-size:11px;color:#9098a1;margin-top:13px;text-align:center}
  </style>
  <div class="shell">
    <section class="panel" id="panel" role="dialog" aria-label="Nghe bản tin Khánh Trang">
      <header class="head"><div><h2>Nghe bản tin</h2><p class="sub">Tóm tắt bài báo và nghe bằng giọng Khánh Trang</p></div><button class="close" id="close" aria-label="Đóng">×</button></header>
      <div class="body">
        <div class="label">BÀI BÁO ĐANG MỞ</div>
        <div class="title" id="title"></div>
        <div class="domain" id="domain"></div>
        <button class="primary" id="run">Tóm tắt & đọc</button>
        <div class="status" id="status" role="status">Sẵn sàng.</div>
        <div class="progress" id="progress"><div class="progress-row"><span id="stage">Đang bắt đầu…</span><strong id="percent">0%</strong></div><div class="track" role="progressbar" aria-label="Tiến độ tạo bản tin" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" id="track"><div class="fill" id="fill"></div></div><div class="progress-note">Tiến độ theo từng bước xử lý, không phải % thời gian còn lại.</div></div>
        <div class="results" id="results">
          <div class="label">BẢN GHI ÂM</div><audio id="audio" controls preload="metadata"></audio>
          <div class="label">NỘI DUNG TÓM TẮT</div><div class="summary" id="summary"></div>
        </div>
        <label class="auto"><input type="checkbox" id="auto"/> Tự động tạo bản tin khi mở bài báo</label>
        <div class="foot">Bạn có thể đóng bảng hoặc chuyển tab trong khi xử lý.</div>
      </div>
    </section>
    <button class="launcher" id="launcher" aria-expanded="false"><span class="dot"></span> Nghe bản tin</button>
  </div>`;
  // Avoid overlap with page content and ensure the launcher receives clicks.
  (document.body || document.documentElement).appendChild(host);
  const $ = id => shadow.getElementById(id);
  const panel=$('panel'), launcher=$('launcher'), run=$('run'), status=$('status');
  const key='kt_job_'+location.href.split('#')[0];
  let currentJob=null, pollTimer=null, checking=false, audioJob=null, audioObjectUrl=null;
  function toggle(force) {
    const open=typeof force==='boolean'?force:!panel.classList.contains('open');
    panel.classList.toggle('open',open);launcher.style.display=open?'none':'flex';
    launcher.setAttribute('aria-expanded',String(open));
  }
  $('launcher').addEventListener('click', event=>{event.preventDefault();event.stopPropagation();toggle(true);});
  $('close').addEventListener('click', event=>{event.preventDefault();toggle(false);});
  chrome.runtime.onMessage.addListener(msg=>{if(msg?.type==='KT_TOGGLE')toggle();});
  $('title').textContent=(document.querySelector('meta[property="og:title"]')?.content||document.title||'Bài báo đang mở').trim();
  $('domain').textContent=location.hostname;
  function setStatus(message,error=false) {status.textContent=message;status.classList.toggle('error',error);}
  function setRunning(running) {run.disabled=running;run.textContent=running?'Đang tạo bản tin…':'Tóm tắt & đọc';}
  function send(msg) {
    return new Promise((resolve,reject)=>{
      chrome.runtime.sendMessage(msg,reply=>{
        if(chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
        if(!reply?.ok) return reject(new Error(reply?.error||'Không kết nối được backend'));
        resolve(reply.job);
      });
    });
  }
  function rawMessage(message) {
    return new Promise((resolve,reject)=>chrome.runtime.sendMessage(message, reply=>{
      if(chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
      if(!reply?.ok) return reject(new Error(reply?.error || 'Không tải được âm thanh'));
      resolve(reply);
    }));
  }
  async function loadAudio(jobId) {
    if (audioJob===jobId && $('audio').src) return;
    audioJob=jobId;
    $('audio').removeAttribute('src');
    $('audio').load();
    setStatus('Đang tải bản ghi âm để phát...');
    try {
      const info=await rawMessage({type:'KT_AUDIO_INFO',jobId});
      if(info.size > 150*1024*1024) throw new Error('Bản ghi âm quá lớn');
      const chunks=[];
      for(let offset=0;offset<info.size;) {
        const part=await rawMessage({type:'KT_AUDIO_CHUNK',jobId,offset});
        const binary=atob(part.data);
        const bytes=new Uint8Array(binary.length);
        for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
        chunks.push(bytes);
        if(part.end<=offset) throw new Error('Không nhận được dữ liệu âm thanh');
        offset=part.end;
      }
      if(audioJob!==jobId)return;
      if(audioObjectUrl)URL.revokeObjectURL(audioObjectUrl);
      audioObjectUrl=URL.createObjectURL(new Blob(chunks,{type:'audio/wav'}));
      $('audio').src=audioObjectUrl;
      $('audio').load();
      setStatus('Đã có bản tin. Nhấn Play để nghe.');
    } catch(e) {
      audioJob=null;
      setStatus('Không tải được âm thanh: '+e.message,true);
    }
  }
  $('audio').addEventListener('error',()=>setStatus('Trình duyệt không phát được file âm thanh này.',true));
  function progress(job) {
    const visible=job.status==='running';
    $('progress').classList.toggle('show',visible);
    if(!visible)return;
    const pct=Math.min(99,Math.max(0,Number(job.progress)||0));
    $('fill').style.width=pct+'%';
    $('percent').textContent=pct+'%';
    $('stage').textContent=job.stage||'Đang tạo bản tin…';
    $('track').setAttribute('aria-valuenow',String(pct));
  }
  function showJob(job) {
    progress(job);
    currentJob=job.id;
    if(job.status==='running') {
      setRunning(true);setStatus('Đang tạo bản tin. Bạn có thể đóng bảng, chuyển tab và quay lại sau.');
      $('results').classList.remove('show');
      schedulePoll();
    } else if(job.status==='done') {
      clearTimeout(pollTimer);setRunning(false);
      setStatus('Đã tạo xong bản tin.');
      $('summary').textContent=job.summary||'Không có bản tóm tắt.';
      loadAudio(job.id);
      $('results').classList.add('show');
      if ($('auto').checked) toggle(true);
    } else {
      clearTimeout(pollTimer);setRunning(false);
      setStatus('Không thể tạo bản tin: '+(job.error||'Không rõ nguyên nhân'),true);
    }
  }
  function schedulePoll() {
    clearTimeout(pollTimer);
    pollTimer=setTimeout(checkStatus,2500);
  }
  async function checkStatus() {
    if(!currentJob||checking)return;
    checking=true;
    try {
      const job=await send({type:'KT_STATUS',jobId:currentJob});
      showJob(job);
    } catch(e) {
      setStatus('Không kết nối được backend. Khi backend hoạt động lại, mở trang này để kiểm tra.',true);
      schedulePoll();
    } finally {checking=false;}
  }
  async function startJob(){
    setRunning(true);setStatus('Đang gửi bài báo...');
    $('results').classList.remove('show');
    try {
      const job=await send({type:'KT_START',url:location.href.split('#')[0]});
      chrome.storage.local.set({[key]:job.id});
      showJob(job);
    } catch(e) {setRunning(false);$('progress').classList.remove('show');setStatus('Lỗi: '+e.message,true);}
  }
  run.addEventListener('click', e=>{e.preventDefault();startJob();});
  const autoInput=$('auto');
  const isArticle=()=>Boolean(document.querySelector('article, meta[property="article:published_time"], script[type="application/ld+json"]')) && (document.querySelectorAll('p').length>=3);
  autoInput.addEventListener('change',()=>chrome.storage.local.set({kt_auto_news:autoInput.checked}));
  chrome.storage.local.get(['kt_auto_news',key], async data=>{
    const jobId=data[key];
    autoInput.checked=Boolean(data.kt_auto_news);
    if(!jobId){
      if(autoInput.checked && isArticle()) startJob();
      return;
    }
    currentJob=jobId;
    try {showJob(await send({type:'KT_STATUS',jobId}));}
    catch(_) {setRunning(false);setStatus('Bản tin cũ chưa truy cập được. Kiểm tra backend rồi mở lại trang.');}
  });
})();
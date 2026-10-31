const API = 'http://127.0.0.1:8765';
chrome.action.onClicked.addListener(async tab => {
  if (!tab.id) return;
  try { await chrome.tabs.sendMessage(tab.id, {type:'KT_TOGGLE'}); } catch (_) {}
});
chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (!['KT_START', 'KT_STATUS', 'KT_AUDIO_INFO', 'KT_AUDIO_CHUNK'].includes(message?.type)) return false;
  (async () => {
    try {
      if (message.type === 'KT_AUDIO_INFO' || message.type === 'KT_AUDIO_CHUNK') {
        const jobId = String(message.jobId || '');
        if (!/^[a-f0-9]{32}$/.test(jobId)) throw new Error('Mã audio không hợp lệ');
        const response = await fetch(API + '/api/audio/' + jobId);
        if (!response.ok) throw new Error('Không tải được bản ghi âm: HTTP ' + response.status);
        const buffer = await response.arrayBuffer();
        const bytes = new Uint8Array(buffer);
        if (message.type === 'KT_AUDIO_INFO') {
          respond({ok:true, size:bytes.length});
        } else {
          const offset = Math.max(0, Number(message.offset) || 0);
          const end = Math.min(bytes.length, offset + 196608);
          const slice = bytes.subarray(offset,end);
          let binary='';
          for (let i=0;i<slice.length;i+=8192) binary += String.fromCharCode(...slice.subarray(i,i+8192));
          respond({ok:true, data:btoa(binary), end, size:bytes.length});
        }
        return;
      }
      const start = message.type === 'KT_START';
      const path = start ? '/api/jobs' : `/api/jobs/${encodeURIComponent(message.jobId)}`;
      const response = await fetch(API + path, start ? {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({url:message.url})
      } : {});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      respond({ok:true, job:{...data, audio_url:data.audio_url ? API + data.audio_url : null}});
    } catch(e) { respond({ok:false,error:String(e.message || e)}); }
  })();
  return true;
});

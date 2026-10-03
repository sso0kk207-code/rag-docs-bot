PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Вопросы по документам</title>
<style>
:root{--bg:#0b1020;--card:#141b34;--line:#26305a;--txt:#e8ecf5;--mut:#8f9bc0;--acc:#5eead4}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:15px/1.5 'Segoe UI',Arial,sans-serif;display:flex;height:100vh}
aside{width:300px;border-right:1px solid var(--line);padding:22px;background:#0e1530;overflow:auto}
aside h2{margin:0 0 14px;font-size:14px;color:var(--mut);letter-spacing:1px;text-transform:uppercase}
.doc{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:8px;font-size:13px}.doc small{color:var(--mut);display:block}
label.up{display:block;text-align:center;border:1.5px dashed var(--line);border-radius:10px;padding:10px;color:var(--mut);cursor:pointer;margin-top:12px;font-size:13px}
main{flex:1;display:flex;flex-direction:column;padding:22px 32px;min-width:0}h1{margin:0 0 4px;font-size:22px}.sub{color:var(--mut);margin-bottom:16px;font-size:13px}
#chat{flex:1;overflow:auto;display:flex;flex-direction:column;gap:14px;padding-bottom:12px}
.q{align-self:flex-end;background:#2b4a8a;border-radius:14px 14px 4px 14px;padding:10px 14px;max-width:70%}
.a{align-self:flex-start;background:var(--card);border:1px solid var(--line);border-radius:14px 14px 14px 4px;padding:12px 16px;max-width:85%}
.cite{display:inline-block;background:#134e4a;color:var(--acc);border-radius:6px;padding:0 6px;font-size:12px;font-weight:700}
.src{margin-top:10px;padding-top:8px;border-top:1px solid var(--line);font-size:12.5px;color:var(--mut)}.src div{margin-top:5px}.src b{color:var(--acc)}
.meta{margin-top:8px;font-size:11.5px;color:var(--mut)}.warn{color:#fbbf24}
form{display:flex;gap:10px}input[type=text]{flex:1;background:var(--card);border:1px solid var(--line);color:var(--txt);border-radius:12px;padding:12px 16px;font-size:15px}
button{background:var(--acc);color:#052e2b;border:0;border-radius:12px;padding:0 22px;font-weight:700;font-size:15px;cursor:pointer}
</style></head><body>
<aside><h2>Документы</h2><div id="docs"></div>
<label class="up">+ загрузить (txt, md, pdf, docx)<input id="file" type="file" hidden></label></aside>
<main><h1>Вопросы по документам</h1><div class="sub" id="llm"></div>
<div id="chat"></div>
<form id="f"><input id="q" type="text" placeholder="Например: сколько дней отпуска положено?" autocomplete="off"><button>Спросить</button></form></main>
<script>
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const chat=document.getElementById('chat');
async function docs(){const d=await fetch('/documents').then(r=>r.json());
  document.getElementById('docs').innerHTML=d.map(x=>`<div class="doc">${esc(x.filename)}<small>${x.chunks} фрагментов${x.pages>1?', '+x.pages+' стр.':''}</small></div>`).join('')||'<div class="sub">Пока пусто</div>';}
async function ask(question){
  chat.insertAdjacentHTML('beforeend',`<div class="q">${esc(question)}</div>`);
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})}).then(r=>r.json());
  const text=esc(r.answer).replace(/\\[(\\d+)\\]/g,'<span class="cite">$1</span>');
  const src=(r.citations||[]).map(c=>`<div><b>[${c.n}]</b> ${esc(c.document)}${c.page?', стр. '+c.page:''}: ${esc(c.snippet)}…</div>`).join('');
  const meta=r.model?`${esc(r.model)} · токенов: ${r.usage.input_tokens}+${r.usage.output_tokens}`+(r.grounded?'':' · <span class="warn">без ссылок на источники</span>'):'модель не вызывалась: в документах нет подходящих фрагментов';
  chat.insertAdjacentHTML('beforeend',`<div class="a">${text}${src?`<div class="src">${src}</div>`:''}<div class="meta">${meta}</div></div>`);
  chat.scrollTop=chat.scrollHeight;
}
document.getElementById('f').onsubmit=e=>{e.preventDefault();const i=document.getElementById('q');if(i.value.trim()){ask(i.value.trim());i.value=''}};
document.getElementById('file').onchange=async e=>{const fd=new FormData();fd.append('file',e.target.files[0]);const r=await fetch('/documents',{method:'POST',body:fd});if(!r.ok)alert((await r.json()).detail);docs()};
fetch('/health').then(r=>r.json()).then(h=>document.getElementById('llm').textContent='Модель: '+h.llm);
docs().then(async()=>{const qs=new URLSearchParams(location.search).getAll('q');for(const q of qs)await ask(q)});
</script></body></html>"""

'use strict';
const $ = id => document.getElementById(id);
let token = '', state = null, timer = null, busy = false, polling = false, initialized = false;
const selected = new Set();
const names = {pending:'待处理', completed:'完成 · 待复核', reused:'已核验复用', failed:'失败', running:'运行中', queued:'正在启动', stopped:'已停止', interrupted:'已中断', completed_with_errors:'部分失败', interrupted_or_stale:'记录待核查'};
const phases = {input_check:'检查输入', total:'分割静脉', heartchambers_highres:'分割心腔', coronary_arteries:'分割冠脉', merge_and_qc:'合并与核验', idle:'空闲'};
const operations = {install:'安装运行环境', probe:'检查环境', license:'验证并保存许可', plan:'检查输入', run:'粗分割'};
function notice(text, error=false){$('notice').hidden=false;$('notice').textContent=text;$('notice').className=error?'error':'';}
async function api(path, payload){
  const options={headers:{Authorization:'Bearer '+token}};
  if(payload!==undefined){options.method='POST';options.headers['Content-Type']='application/json';options.body=JSON.stringify(payload);}
  const response=await fetch('/api/'+path,options);const data=await response.json();
  if(!response.ok)throw new Error(data.error||'请求失败');return data;
}
function updateButtons(){
  const active=state&&(state.active||state.locked), env=state?.environment;
  $('selected').textContent='已选 '+selected.size;
  $('start').disabled=busy||active||!selected.size||!env?.license_configured||env?.versions?.TotalSegmentator!=='2.18.0';
  $('plan').disabled=busy||active||!selected.size;
  $('stop').disabled=busy||!state?.active||state.stop_requested||(state.managed&&state.operation.action!=='run');
  for(const id of ['installEnv','probeEnv','saveLicense','saveSettings','manifestUpload','buildSelect','deviceSelect','manifestInput','outputInput','threadInput','gpuConfirm'])$(id).disabled=busy||active;
}
function fillSettings(force=false){
  if(!state.managed)return;
  const device=$('deviceSelect'), previous=initialized&&!force?device.value:state.config.device;
  device.replaceChildren(new Option('CPU','cpu'));
  for(const d of state.environment.devices||[])device.add(new Option(`GPU ${d.index} · ${d.name} · ${d.total_gib} GiB`,'gpu:'+d.index));
  if(![...device.options].some(o=>o.value===previous))device.add(new Option(previous+'（需重新检查）',previous));
  device.value=previous;
  if(!initialized||force){
    const c=state.config;
    $('manifestInput').value=c.manifest;$('outputInput').value=c.output_root;$('buildSelect').value=c.torch_build;
    $('threadInput').value=c.threads;$('gpuConfirm').checked=c.gpu_confirmed;initialized=true;
  }
}
function settings(){return {manifest:$('manifestInput').value.trim(),output_root:$('outputInput').value.trim(),device:$('deviceSelect').value,threads:Number($('threadInput').value),torch_build:$('buildSelect').value,gpu_confirmed:$('gpuConfirm').checked};}
function renderCases(){
  const body=$('cases');body.replaceChildren();const query=$('filter').value.trim().toLowerCase();
  const available=new Set(state.cases.map(r=>r.case_id));for(const id of selected)if(!available.has(id))selected.delete(id);
  for(const row of state.cases){
    if(!row.case_id.toLowerCase().includes(query))continue;
    const tr=document.createElement('tr'), choice=document.createElement('td'), check=document.createElement('input');
    check.type='checkbox';check.checked=selected.has(row.case_id);check.setAttribute('aria-label','选择 '+row.case_id);
    check.addEventListener('change',()=>{if(check.checked)selected.add(row.case_id);else selected.delete(row.case_id);updateButtons();});choice.append(check);tr.append(choice);
    const id=document.createElement('td');id.textContent=row.case_id;tr.append(id);
    const status=document.createElement('td'), badge=document.createElement('span');badge.className='status '+row.status;badge.textContent=names[row.status]||row.status;status.append(badge);tr.append(status);
    const warning=document.createElement('td');warning.textContent=row.error||(row.empty_masks?.length?'空掩膜 '+row.empty_masks.length+' 类':'—');tr.append(warning);body.append(tr);
  }updateButtons();
}
function showPlan(data){
  $('preflight').hidden=false;
  $('planResult').textContent=data.cases.map(row=>row.status==='ready'?`${row.case_id}  ✓  ${row.geometry.shape.join(' × ')}；间距 ${row.geometry.spacing_mm.map(x=>x.toFixed(3)).join(' × ')} mm\n${row.geometry.warnings.join('\n')}`:`${row.case_id}  ✕  ${row.error}`).join('\n\n');
}
async function refresh(){
  if(polling)return;polling=true;
  try{
    state=await api('status');$('workspace').hidden=false;$('login').hidden=true;
    $('connection').textContent='● 已连接';$('connection').className='connection connected';
    $('serverSetup').hidden=!state.managed;fillSettings();
    $('total').textContent=state.cases.length;$('done').textContent=state.cases.filter(x=>['completed','reused'].includes(x.status)).length;
    const p=state.progress, op=state.operation||{};
    $('failed').textContent=p.failed||0;
    $('phase').textContent=state.active?(op.action&&op.action!=='run'?operations[op.action]:(phases[p.phase]||'准备中')):(names[p.status]||'待开始');
    $('phaseDetail').textContent=state.stop_requested?'已请求停止，等待任务退出':(p.current_case||'选择体积后开始');
    $('device').textContent=state.device+' · '+state.threads+' 个 CPU 线程';$('visible').textContent=state.environment.cuda_visible_devices;
    $('version').textContent=state.environment.versions?.TotalSegmentator||'未安装 TotalSegmentator';
    $('license').textContent=state.environment.license_configured?'已配置':'未配置，暂不能开始推理';
    $('manifest').textContent=state.manifest||'尚未配置';$('output').textContent=state.output_root;
    $('progress').max=p.total||1;$('progress').value=p.finished||0;
    $('progressText').textContent=p.total?`${p.finished||0} / ${p.total} 个体积已结束`:'尚未开始';
    $('current').textContent=p.current_case?`当前 ${p.current_case} · ${phases[p.phase]||p.phase}`:'模型内部进度请查看日志；粗分割完成后需逐例人工复核。';
    if(state.managed){
      $('licensePending').textContent=state.license_pending?'已导入私有许可，安装环境后点击验证。':(state.environment.license_configured?'此工作区已配置许可；留空可保留原许可。':'输入许可码后点击验证。');
      $('operationTitle').textContent=op.action?operations[op.action]:'尚未提交后台操作';
      $('operationState').textContent=op.active?'后台运行中':(op.status==='completed'?'已完成':(names[op.status]||'待准备'));
      $('operationDetail').textContent=op.error||state.manifest_error||(op.active?'可关闭网页或断开 SSH；重连后会恢复此任务的进度。':(op.ended_at?'结束时间 '+op.ended_at:'先安装环境，再验证许可并配置清单。'));
      if(state.preflight)showPlan(state.preflight);
    }
    renderCases();$('logs').textContent=(await api('logs')).text||'暂无日志。';
  }finally{polling=false;}
}
async function guarded(fn){busy=true;updateButtons();try{await fn();}catch(error){notice(error.message,true);}finally{busy=false;updateButtons();}}
async function host(){const h=await api('host');$('hostInfo').textContent=`${h.system}\nPython ${h.python} · ${h.cpu_count} CPU 线程\n工作区磁盘剩余 ${h.disk_free_gib} GiB\n`+(h.gpus.map(g=>`${g.name} · 驱动 ${g.driver}\n${g.uuid} · 已用 ${g.used_mb} / ${g.total_mb} MB`).join('\n')||h.gpu_error);}
$('connect').addEventListener('click',()=>guarded(async()=>{
  token=$('token').value.trim();await refresh();$('token').value='';notice('已连接。后台任务会在重连后自动恢复显示。');
  clearInterval(timer);timer=setInterval(()=>{if(!busy)refresh().catch(()=>{$('connection').textContent='连接断开 · 正在重连';notice('页面暂时无法连接。服务器后台任务继续；请恢复 SSH 端口转发。',true);});},3000);
  $('gpu').textContent=(await api('gpu')).text;if(state.managed)await host();
}));
$('token').addEventListener('keydown',event=>{if(event.key==='Enter')$('connect').click();});
$('filter').addEventListener('input',()=>{if(state)renderCases();});
for(const [id,count] of [['first3',3],['all',Infinity],['none',0]])$(id).addEventListener('click',()=>{selected.clear();state.cases.slice(0,count).forEach(row=>selected.add(row.case_id));renderCases();});
$('plan').addEventListener('click',()=>guarded(async()=>{const data=await api('plan',{case_ids:[...selected]});if(data.submitted)notice(data.note);else{showPlan(data);notice(`预检通过 ${data.ready} / ${data.cases.length}。`,data.ready!==data.cases.length);}await refresh();}));
$('start').addEventListener('click',()=>guarded(async()=>{const result=await api('start',{case_ids:[...selected]});notice(result.note+'；本次选择 '+result.count+' 个体积。');await refresh();}));
$('stop').addEventListener('click',()=>guarded(async()=>{notice((await api('stop',{})).note);await refresh();}));
$('gpuRefresh').addEventListener('click',()=>guarded(async()=>{$('gpu').textContent=(await api('gpu')).text;}));
$('hostRefresh').addEventListener('click',()=>guarded(host));
$('deviceSelect').addEventListener('change',()=>{$('gpuConfirm').checked=false;});
$('saveSettings').addEventListener('click',()=>guarded(async()=>{notice((await api('setup/configure',settings())).note);await refresh();fillSettings(true);}));
for(const [button,action] of [['installEnv','install'],['probeEnv','probe']])$(button).addEventListener('click',()=>guarded(async()=>{
  if(action==='install')await api('setup/configure',settings());
  notice((await api('setup/'+action,{})).note);await refresh();
}));
$('saveLicense').addEventListener('click',()=>guarded(async()=>{const value=$('licenseInput').value.trim();$('licenseInput').value='';notice((await api('setup/license',value?{license:value}:{})).note);await refresh();}));
$('manifestUpload').addEventListener('change',()=>guarded(async()=>{
  const file=$('manifestUpload').files[0];if(!file)return;
  if(file.size>1500000)throw new Error('清单过大，请改用服务器路径');
  notice((await api('setup/manifest',{csv:await file.text()})).note);$('manifestUpload').value='';await refresh();fillSettings(true);
}));

import {useEffect,useState} from 'react';
import {Activity,ChartNoAxesCombined,Database,SlidersHorizontal,Play,Pause,Save,Square,CheckCheck,FlaskConical} from 'lucide-react';
import {api,command,useStatus} from './api';
import {Button} from './components/ui/button';
import {ConfirmButton} from './components/common';
import {Overview} from './views/Overview';
import {Diagnostics} from './views/Diagnostics';
import {Dataset} from './views/Dataset';
import {Parameters} from './views/Parameters';
import {Experiments} from './views/Experiments';
import {number} from './lib/utils';

const tabs=[['overview','训练总览',ChartNoAxesCombined],['diagnostics','学习诊断',Activity],['experiments','独立实验',FlaskConical],['dataset','数据与覆盖',Database],['parameters','参数与模型',SlidersHorizontal]] as const;
const labels:Record<string,string>={initializing:'准备离线数据',paused:'已暂停',training:'离线训练',validating:'固定集验证',stopped:'已停止',error:'运行异常'};

export default function App(){
  const {state,connected}=useStatus(),[tab,setTab]=useState('overview'),[message,setMessage]=useState(''),[busy,setBusy]=useState(false);
  const terminal=state.state==='error'||state.state==='stopped';
  useEffect(()=>{const context=(document as any).modelContext;if(!context?.registerTool)return;const lifetime=new AbortController();try{Promise.resolve(context.registerTool({name:'read_bc_training_status',description:'读取离线 BC 训练状态、验证指标和速度，不修改模型。',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true},execute:()=>api('status')},{signal:lifetime.signal})).catch(()=>{});}catch{}return()=>lifetime.abort();},[]);
  const act=async(name:string)=>{setBusy(true);try{const result=await command(name);setMessage(result.message);}catch(e){setMessage(String(e));}finally{setBusy(false);}};
  return <div className="app-shell"><aside className="sidebar"><div className="brand"><span className="brand-mark">S</span><div><strong>SOKU / BC</strong><small>离线训练工作台</small></div></div><nav aria-label="训练模块">{tabs.map(([id,label,Icon])=><button key={id} className={id===tab?'active':''} onClick={()=>setTab(id)}><Icon size={18}/><span>{label}</span></button>)}</nav><div className="sidebar-foot"><span className="local-dot"/>局域网 · {location.host}<small>BC 分类 · 432 / 固定 REP 数据</small></div></aside>
    <div className="workspace"><header className="topbar"><div className="run-status"><span className={`status-dot ${connected?'online':''}`}/><strong>{labels[state.state]||'正在连接'}</strong><span className="muted">Step {number(state.step,0)} · 阶段 {state.stage??0}</span></div><div className="toolbar">
      <Button disabled={!connected||terminal||busy||state.state!=='paused'} onClick={()=>void act('resume')}><Play size={15}/>开始 / 继续</Button>
      <Button variant="secondary" disabled={!connected||terminal||state.state==='initializing'||state.state==='paused'} onClick={()=>void act('pause')}><Pause size={15}/>暂停</Button>
      <Button variant="secondary" disabled={!connected||terminal||busy||state.state==='initializing'} onClick={()=>void act('save')}><Save size={15}/>保存版本</Button>
      <Button variant="secondary" disabled={!connected||busy||state.state!=='paused'} onClick={()=>void act('validate')}><CheckCheck size={15}/>验证</Button>
      <ConfirmButton disabled={!connected||terminal} title="停止并保存 BC 模型" description="等待当前更新结束，保存 last.pt。正式训练数据不会修改。" onConfirm={()=>command('stop')}><Square size={15}/>停止</ConfirmButton>
    </div></header><div className="connection-strip"><span>{connected?'已连接':'连接中断，正在重连；网页断开不会停止训练'}</span><span>878D→1024D 当前状态 · TCN32→256D · Joint432 离线 BC</span></div>
    {(message||state.error)&&<div className="feedback" role="status">{state.error||message}<button aria-label="关闭提示" onClick={()=>setMessage('')}>×</button></div>}
    <div className="runtime-message">{state.message||'等待运行状态'}</div>
    <main key={state.output}>{tab==='overview'?<Overview state={state}/>:tab==='diagnostics'?<Diagnostics state={state}/>:tab==='experiments'?<Experiments state={{...state,connected}}/>:tab==='dataset'?<Dataset state={state}/>:<Parameters state={state}/>}</main>
    <footer><span>{state.output||'准备模型目录'}</span><span>累计训练帧 {number(state.samples,0)}（含重复采样） · 最近保存 Step {number(state.last_saved?.step,0)}</span></footer></div>
  </div>;
}

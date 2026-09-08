import {useEffect,useState} from 'react';
import {api,command,type Row} from '../api';
import {Button} from '../components/ui/button';
import {Card,DataTable} from '../components/common';
import {moduleLabels} from './Diagnostics';

export function Parameters({state}:{state:Row}){
  const [schema,setSchema]=useState<Row>({fields:[],modules:[],config:{training:{}}}),[draft,setDraft]=useState<Row>({});
  const [locks,setLocks]=useState<string[]>([]),[frozen,setFrozen]=useState<string[]>([]),[message,setMessage]=useState(''),[busy,setBusy]=useState(false),[models,setModels]=useState<Row[]>([]);
  const refresh=async()=>{const data=await api('parameters');setSchema(data);setDraft({...data.config.training});setLocks(data.locks);setFrozen(data.config.training.frozen_modules);setModels(await api('models'));};
  useEffect(()=>{void refresh().catch(e=>setMessage(String(e)));},[state.stage]);
  const paused=state.state==='paused',disabled=!paused||busy;
  const apply=async()=>{setBusy(true);try{
    const changes:Row={};for(const field of schema.fields){const text=draft[field.key];if(text===''||text==null)throw new Error(`${field.label} 不能为空`);const value=Number(text);if(value!==schema.config.training[field.key])changes[field.key]=value;}
    if(JSON.stringify(frozen)!==JSON.stringify(schema.config.training.frozen_modules))changes.frozen_modules=frozen;
    if(!Object.keys(changes).length){setMessage('没有参数变化');return;}
    const result=await command('configure',{training:changes,expected_stage:schema.stage});setMessage(result.message);if(result.status==='completed')await refresh();
  }catch(e){setMessage(String(e));}finally{setBusy(false);}};
  return <><div className="section-intro"><div><h1>超参数与模型记录</h1><p>暂停后应用；保留优化器状态，每次修改保存快照并标记新阶段。</p></div><Button variant="secondary" disabled={busy} onClick={()=>void refresh().catch(e=>setMessage(String(e)))}>读取当前配置</Button></div>
    {!paused&&<p className="notice">当前不是暂停状态。参数可以预先编辑，待暂停完成后应用。</p>}{message&&<p role="status" className="feedback">{message}</p>}
    <Card title="可调训练参数" note={`运行值来源：${schema.source||'读取中'}`}><div className="table-scroll parameter-table"><table><thead><tr><th>锁定</th><th>参数</th><th>编辑值</th><th>运行值</th><th>范围</th></tr></thead><tbody>{schema.fields.map((field:Row)=><tr key={field.key}><td><input type="checkbox" aria-label={`锁定 ${field.label}`} checked={locks.includes(field.key)} disabled={disabled} onChange={event=>setLocks(event.target.checked?[...locks,field.key]:locks.filter(x=>x!==field.key))}/></td><td><label htmlFor={field.key}>{field.label}</label><small className="block">{field.key}</small></td><td><input id={field.key} type="number" min={field.min} max={field.max} step={field.type==='int'?1:'any'} value={draft[field.key]??''} disabled={busy||locks.includes(field.key)} onChange={event=>setDraft({...draft,[field.key]:event.target.value})}/></td><td>{String(schema.config.training[field.key])}</td><td>{field.min}–{field.max}</td></tr>)}</tbody></table></div>
      <div className="inline-controls parameter-actions"><Button disabled={disabled} onClick={()=>void apply()}>应用参数与冻结设置</Button><Button variant="secondary" disabled={disabled} onClick={async()=>{setBusy(true);try{const result=await command('locks',{keys:locks});setMessage(result.message);}catch(e){setMessage(String(e));}finally{setBusy(false);}}}>保存参数锁</Button></div>
    </Card>
    <Card title="模块冻结" note="联合 Q 头共享主干；仅冻结 Q 头不能固定输出，资源 embedding 属于 Current Encoder"><div className="inline-controls"><Button variant="secondary" disabled={busy||locks.includes('frozen_modules')} onClick={()=>setFrozen([])}>全部训练</Button><Button variant="secondary" disabled={busy||locks.includes('frozen_modules')} onClick={()=>setFrozen(schema.modules.filter((x:string)=>!x.endsWith('_head')))}>仅训练 Joint Q 头</Button></div>
      <div className="check-grid">{schema.modules.map((name:string)=><label className="check-label" key={name}><input type="checkbox" checked={frozen.includes(name)} disabled={busy||locks.includes('frozen_modules')} onChange={event=>setFrozen(event.target.checked?[...frozen,name]:frozen.filter(x=>x!==name))}/>{moduleLabels[name]||name}</label>)}</div>
      <label className="check-label"><input type="checkbox" disabled={disabled} checked={locks.includes('frozen_modules')} onChange={e=>setLocks(e.target.checked?[...locks,'frozen_modules']:locks.filter(x=>x!=='frozen_modules'))}/>锁定冻结设置</label><p className="muted">勾选表示冻结；实际生效模块可在「学习诊断」中检查权重变化。参数锁需要单独保存。</p>
    </Card>
    <Card title="固定配置与网络结构" note="输入、GRU 维度、序列长度、burn-in、AMP、缓存和划分在 YAML 中修改后重启"><pre>{JSON.stringify(schema.config,null,2)}</pre></Card>
    <Card title="已保存模型" note="last.pt 用于续训；best.pt 为当前阶段最低验证 TD MSE"><p className="muted">{state.output}</p><DataTable rows={models} columns={[{key:'name',title:'相对输出目录'},{key:'size_mb',title:'MB'},{key:'modified',title:'保存时间',render:v=>new Date(v*1000).toLocaleString()}]}/></Card>
  </>;
}

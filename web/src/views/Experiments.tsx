import {useState} from 'react';
import {command,useQuery,type Row} from '../api';
import {Card,ConfirmButton,DataTable,Stat} from '../components/common';
import {Trend} from '../components/Trend';
import {number,percent} from '../lib/utils';

export function Experiments({state}:{state:Row}){
  const [mode,setMode]=useState('tcn'),[name,setName]=useState(''),[freeze,setFreeze]=useState(false);
  const [message,setMessage]=useState(''),[busy,setBusy]=useState(false);
  const {data,error}=useQuery('experiments',2000);
  const runs:Row[]=data?.rows||[],active=state.temporal||{};
  const currentHash=data?.current_conditions_hash;
  const [selected,setSelected]=useState<string[]|null>(null);
  const checked=selected===null?runs.slice(0,2):runs.filter(row=>selected.includes(row.output));
  const colors=['#42d6b0','#70a7ff','#edbf6e','#f38caa'];
  // 相同步数共用横坐标，每组使用独立系列；不能把两组记录首尾连成同一曲线。
  const points=new Map<number,Row>();
  checked.forEach((run,index)=>(run.validation||[]).forEach((row:Row)=>{
    const point=points.get(row.step)||{step:row.step,stage:0};
    point[`nll_${index}`]=row.nll;point[`change_${index}`]=row.val_action_change_accuracy;
    points.set(row.step,point);
  }));
  const chartRows=[...points.values()].sort((a,b)=>a.step-b.step);
  const tables=runs.map(run=>{const val=run.validation?.at(-1)||{};return {...run,
    mode:run.temporal_mode==='tcn'?'TCN32（GRU 旁路）':run.module_status?.gru?.frozen?'GRU 冻结前向':'GRU',
    comparable:run.conditions_hash===currentHash?'除时序结构外条件一致':'条件不同，请查看明细',
    nll:val.nll,top1:val.joint_accuracy,change:val.val_action_change_accuracy,val_step:val.step,
  };});
  const create=async()=>{
    setBusy(true);setMessage('');
    try{const result=await command('experiment',{name,temporal_mode:mode,freeze_gru:mode==='gru'&&freeze,
      expected_stage:state.stage,expected_output:state.output,confirm:true});setMessage(result.message);}
    catch(error){setMessage(String(error));throw error;}finally{setBusy(false);}
  };
  return <><div className="section-intro"><div><h1>GRU / TCN32 对照实验</h1><p>动作空间、状态字段和 BC 交叉熵不变；新分支随机初始化，不迁移旧 GRU 权重。</p></div></div>
    <div className="stats-grid"><Stat title="当前时序结构" value={active.mode==='tcn'?'TCN · 32 帧':active.mode==='gru'?'GRU · 128D':'尚未加载'} note={active.gru_bypassed?'GRU 强制冻结且不执行':'冻结 GRU 不等于旁路；仍会影响输出'}/>
      <Stat title="前导上下文 / 监督长度" value={`${active.prefix_frames??'—'} / ${active.sequence_length??'—'} 帧`} note={active.mode==='tcn'?'每帧只使用当前帧及前 31 帧':'前导帧 no_grad 恢复 GRU 记忆'}/>
      <Stat title="新实验共同设置" value="31 帧前导" note="沿用相同数据、划分、种子和其余训练参数；冻结列表重新选择"/></div>
    <Card title="创建独立随机分支" note="原模型先保存；新实验创建后保持暂停。已有名称会拒绝，不覆盖 last.pt。">
      <div className="inline-controls"><label htmlFor="temporal-mode">时序结构 <select id="temporal-mode" value={mode} disabled={busy} onChange={e=>setMode(e.target.value)}><option value="tcn">TCN32，GRU 旁路</option><option value="gru">GRU 对照组</option></select></label>
        <label htmlFor="temporal-run-name">实验名 <input id="temporal-run-name" value={name} maxLength={64} placeholder="tcn32_seed42_a" disabled={busy} onChange={e=>setName(e.target.value)}/></label>
        {mode==='gru'&&<label className="check-label"><input type="checkbox" checked={freeze} disabled={busy} onChange={e=>setFreeze(e.target.checked)}/>冻结 GRU 权重但保留前向</label>}
        <ConfirmButton disabled={busy||!state.connected||state.state!=='paused'||!name||!!state.locked_parameters?.length}
          title={`创建 ${mode.toUpperCase()} 随机实验 ${name}`}
          description="保存并保留当前模型，在 outputs/temporal_experiments 下创建全新模型和优化器；不加载旧权重，不改变 NPZ。新组使用 31 帧前导，清空现有冻结选择，创建后不会自动训练。"
          onConfirm={create}>保存当前并创建实验</ConfirmButton></div>
      <p className="muted">实验名只能用英文字母、数字、下划线和连字符。请先暂停并解除参数锁；旧分支之后仍可通过其 last.pt 单独续训。</p>
      {(message||error)&&<p role="status" className="feedback">{message||error}</p>}
    </Card>
    <Card title="实验比较" note="默认展示当前组及最近一组。按相同训练步、有效训练帧和验证条件比较；缺失指标显示未记录。">
      <div className="check-grid">{runs.map(run=><label className="check-label" key={run.output}><input type="checkbox" checked={checked.some(x=>x.output===run.output)} onChange={e=>{const previous=checked.map(x=>x.output);setSelected(e.target.checked?[...previous,run.output].slice(-4):previous.filter(x=>x!==run.output));}}/>{run.name} · {run.temporal_mode.toUpperCase()}</label>)}</div>
      <DataTable rows={tables} columns={[{key:'name',title:'实验'},{key:'mode',title:'结构'},{key:'step',title:'训练步'},{key:'updates',title:'实际更新'},{key:'samples',title:'累计监督帧'},{key:'val_step',title:'验证步'},{key:'nll',title:'验证 NLL ↓',render:v=>number(v,5)},{key:'top1',title:'Top-1',render:percent},{key:'change',title:'切换 Top-1',render:percent},{key:'samples_per_second',title:'监督帧/秒',render:v=>number(v,0)},{key:'comparable',title:'条件核对'}]}/>
    </Card>
    {checked.some(run=>run.conditions_hash!==currentHash)&&<p className="notice">选中组与当前组存在数据、归一化、前导长度、冻结或超参数差异，不能只把效果差异归因于 TCN/GRU。参数明细见下方。</p>}
    <div className="two-columns"><Trend title="同训练步：验证 NLL" rows={chartRows} numericSteps series={checked.map((run,index)=>({key:`nll_${index}`,name:run.name,color:colors[index%4]}))}/>
      <Trend title="同训练步：切换帧 Top-1" rows={chartRows} numericSteps series={checked.map((run,index)=>({key:`change_${index}`,name:run.name,color:colors[index%4]}))}/></div>
    <Card title="对照条件与解释" note="验证误差改善不等于实战必然变强；用同一 CPU 难度、按键和采样设置分别评估两份模型。">
      <p>TCN 的感受野固定为 32 帧，GRU 则累积循环记忆；训练对照使用相同前导长度，但实战记忆机制和参数量本身仍不同。冻结 GRU 只停止权重更新，不会固定整个策略输出。条件核对不包含实际 GPU 型号；比较速度时还需使用相同硬件和负载。</p>
      {checked.map(run=><details key={run.output}><summary>{run.name} · 阶段 {run.stage} · 条件明细</summary><pre>{JSON.stringify(run.conditions,null,2)}</pre><p>{run.output}</p></details>)}
    </Card>
  </>;
}

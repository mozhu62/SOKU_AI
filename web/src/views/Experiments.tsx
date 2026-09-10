import {useState} from 'react';
import {command,useQuery,type Row} from '../api';
import {Card,ConfirmButton,DataTable,Stat} from '../components/common';
import {Trend} from '../components/Trend';
import {number,percent} from '../lib/utils';

export function Experiments({state}:{state:Row}){
  const [name,setName]=useState(''),[message,setMessage]=useState(''),[busy,setBusy]=useState(false);
  const {data,error}=useQuery('experiments',2000);
  const runs:Row[]=data?.rows||[],active=state.temporal||{};
  const currentHash=data?.current_conditions_hash;
  const [selected,setSelected]=useState<string[]|null>(null);
  const checked=selected===null?runs.slice(0,2):runs.filter(row=>selected.includes(row.output));
  const colors=['#42d6b0','#70a7ff','#edbf6e','#f38caa'];
  const points=new Map<number,Row>();
  checked.forEach((run,index)=>(run.validation||[]).forEach((row:Row)=>{
    const point=points.get(row.step)||{step:row.step,stage:0};
    point[`nll_${index}`]=row.nll;
    point[`change_${index}`]=row.val_action_change_accuracy;
    points.set(row.step,point);
  }));
  const chartRows=[...points.values()].sort((a,b)=>a.step-b.step);
  const tables=runs.map(run=>{const val=run.validation?.at(-1)||{};return {...run,
    architecture:'宽 TCN32',
    comparable:run.conditions_hash===currentHash?'条件一致':'条件不同，请查看明细',
    nll:val.nll,top1:val.joint_accuracy,change:val.val_action_change_accuracy,val_step:val.step,
  };});
  const create=async()=>{
    setBusy(true);setMessage('');
    try{
      const result=await command('experiment',{
        name,expected_stage:state.stage,expected_output:state.output,confirm:true,
      });
      setMessage(result.message);
    }catch(reason){setMessage(String(reason));throw reason;}
    finally{setBusy(false);}
  };

  return <><div className="section-intro"><div><h1>宽 TCN32 独立实验</h1><p>固定使用新网络结构；新分支随机初始化，不迁移旧权重。</p></div></div>
    <div className="stats-grid">
      <Stat title="当前状态分支" value="878D → 1024D" note="单层 Linear + LayerNorm + SiLU"/>
      <Stat title="历史状态分支" value="32 帧 → 256D" note="每个残差块含两层因果卷积"/>
      <Stat title="最终融合" value="1536D → 1024D → 432" note="1024 当前 + 256 时序 + 128×2 对象"/>
      <Stat title="前导 / 监督长度" value={`${active.prefix_frames??'—'} / ${active.sequence_length??'—'} 帧`} note="每个监督帧最多读取当前帧及前 31 帧"/>
    </div>
    <Card title="创建独立随机分支" note="原模型先保存；新实验创建后保持暂停。已有目录会被拒绝，不覆盖 last.pt。">
      <div className="inline-controls">
        <label htmlFor="experiment-name">实验名<input id="experiment-name" value={name} maxLength={64} placeholder="wide_tcn32_seed42_a" disabled={busy} onChange={event=>setName(event.target.value)}/></label>
        <ConfirmButton disabled={busy||!state.connected||state.state!=='paused'||!name||!!state.locked_parameters?.length}
          title={`创建宽 TCN32 随机实验 ${name}`}
          description="保存当前模型，在 outputs/temporal_experiments 下创建全新的模型和优化器。不会加载旧权重、修改 NPZ 或自动开始训练。"
          onConfirm={create}>保存当前并创建实验</ConfirmButton>
      </div>
      <p className="muted">实验名只允许英文字母、数字、下划线和连字符。旧分支仍可通过各自的 last.pt 单独保留，但旧架构不能由当前代码续训。</p>
      {(message||error)&&<p role="status" className="feedback">{message||error}</p>}
    </Card>
    <Card title="实验比较" note="默认展示当前组及最近一组；按相同训练步、有效训练帧和验证条件比较。">
      <div className="check-grid">{runs.map(run=><label className="check-label" key={run.output}><input type="checkbox" checked={checked.some(item=>item.output===run.output)} onChange={event=>{const previous=checked.map(item=>item.output);setSelected(event.target.checked?[...previous,run.output].slice(-4):previous.filter(item=>item!==run.output));}}/>{run.name} · 宽 TCN32</label>)}</div>
      <DataTable rows={tables} columns={[
        {key:'name',title:'实验'},{key:'architecture',title:'结构'},{key:'step',title:'训练步'},
        {key:'updates',title:'实际更新'},{key:'samples',title:'累计监督帧'},{key:'val_step',title:'验证步'},
        {key:'nll',title:'验证 NLL ↓',render:value=>number(value,5)},
        {key:'top1',title:'Top-1',render:percent},{key:'change',title:'切换 Top-1',render:percent},
        {key:'samples_per_second',title:'监督帧/秒',render:value=>number(value,0)},{key:'comparable',title:'条件核对'},
      ]}/>
    </Card>
    {checked.some(run=>run.conditions_hash!==currentHash)&&<p className="notice">选中组存在数据、归一化、冻结或超参数差异，不能把效果差异只归因于随机种子。</p>}
    <div className="two-columns">
      <Trend title="同训练步：验证 NLL" rows={chartRows} numericSteps series={checked.map((run,index)=>({key:`nll_${index}`,name:run.name,color:colors[index%4]}))}/>
      <Trend title="同训练步：切换帧 Top-1" rows={chartRows} numericSteps series={checked.map((run,index)=>({key:`change_${index}`,name:run.name,color:colors[index%4]}))}/>
    </div>
    <Card title="条件明细" note="比较速度时还需要使用相同硬件和系统负载。">
      {checked.map(run=><details key={run.output}><summary>{run.name} · 阶段 {run.stage}</summary><pre>{JSON.stringify(run.conditions,null,2)}</pre><p>{run.output}</p></details>)}
    </Card>
  </>;
}

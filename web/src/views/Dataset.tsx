import {useEffect,useState} from 'react';
import {api,type Row} from '../api';
import {Button} from '../components/ui/button';
import {Card,DataTable,Stat} from '../components/common';
import {JointFrequency} from '../components/JointActions';
import {number,percent} from '../lib/utils';

export function Dataset({state}:{state:Row}){
  const [offset,setOffset]=useState(0),[files,setFiles]=useState<Row>({rows:[],total:0}),[error,setError]=useState('');
  useEffect(()=>{let done=false;api(`dataset?offset=${offset}`).then(data=>{if(!done)setFiles(data);}).catch(e=>{if(!done)setError(String(e));});return()=>{done=true;};},[offset,state.data?.split_hash]);
  const train=state.data?.train||{},val=state.data?.validation||{};
  return <><div className="section-intro"><div><h1>数据与覆盖率</h1><p>按整份 REP 分组 8:2，完整重复 NPZ 不跨集合。序列不跨终局或断帧。</p></div></div>
    <div className="stats-grid"><Stat title="训练 REP" value={number(train.replays,0)} note={`${number(train.transitions,0)} 个有效动作帧`}/><Stat title="验证 REP" value={number(val.replays,0)} note={`${number(val.transitions,0)} 个有效动作帧`}/><Stat title="当前缓存" value={`${number(state.latest_train?.cache_gb,2)} GB`} note={`上限 ${state.config?.data?.cache_gb??'—'} GB`}/><Stat title="缓存命中率" value={percent(state.latest_train?.cache_hit_rate)}/></div>
    <Card title="全量固定数据：上一帧复制基线" note="随数据预读一次计算并缓存；不抽样、不运行模型，基线分母排除 START/PAD">
      <DataTable rows={[{name:'训练集',...train},{name:'验证集',...val}]} columns={[{key:'name',title:'集合'},{key:'previous_action_baseline',title:'上一帧复制基线',render:percent},{key:'previous_action_samples',title:'可比较历史帧'},{key:'previous_action_copy_correct',title:'保持同一动作帧'},{key:'action_change_samples',title:'切换帧'},{key:'action_change_fraction',title:'切换帧 / 全部有效帧',render:percent},{key:'valid_samples',title:'全部有效帧'}]}/>
    </Card>
    <Card title="状态输入"><p>人物主体：18 数值 + 5 类别 + 9 战术标志；双方各最多 3 个对象，仍是每对象 8 数值 + action/action_block embedding。</p><p>Current Encoder 仅保留双方各 4 个 Skill 槽的类型 embedding 与有效 mask，共 72D；卡牌全部状态、学习等级及生效等级已移除。展开后的当前状态为 228D，经单层编码得到 256D。</p><p>新增上一帧实际 joint_action_id（145×32 embedding，144 是 START）及上一帧方向组合持续时间（clip 60 / 60）。当前帧标签不进入当前输入；断帧、片段起点重置。</p><p className="muted">max_spirit / hitstop 仅使用实际记录且训练集有覆盖的字段；全部连续归一化仅用训练集拟合，上一帧持续时间使用固定缩放。</p><details><summary>完整字段清单</summary><pre>{state.model_spec?JSON.stringify(state.model_spec.inputs,null,2):'数据准备后显示'}</pre></details></Card>
    <Card title="可选状态记录覆盖" note="旧 NPZ 未记录的字段显示 0 个真实样本，不会当成游戏数值零"><DataTable rows={['self_max_spirit','self_hitstop','opponent_max_spirit','opponent_hitstop'].map((name,i)=>({name,train:train.optional_state_counts?.[i],validation:val.optional_state_counts?.[i]}))} columns={[{key:'name',title:'字段'},{key:'train',title:'训练集有效记录数'},{key:'validation',title:'验证集有效记录数'}]}/></Card>
    <Card title="技能配置识别覆盖率" note="有效动作帧起点中成功识别技能的比例；缺失保留未知"><DataTable rows={['self','opponent'].flatMap(side=>[1,2,3,4].map((slot,i)=>({name:`${side==='self'?'己方':'对方'} skill_slot_${slot}`,train_rate:train.resource_coverage?.[side]&&train.transitions?train.resource_coverage[side][i]/train.transitions:null,val_rate:val.resource_coverage?.[side]&&val.transitions?val.resource_coverage[side][i]/val.transitions:null})))} columns={[{key:'name',title:'技能槽'},{key:'train_rate',title:'训练集',render:percent},{key:'val_rate',title:'验证集',render:percent}]}/></Card>
    <div className="two-columns"><Card title="完整训练集 Joint Action Top-N" note={`Neutral ${percent(train.neutral_fraction)}`}><JointFrequency catalog={state.action_catalog} counts={train.joint_counts} total={train.transitions}/></Card><Card title="验证集 Joint Action Top-N" note={`Neutral ${percent(val.neutral_fraction)}`}><JointFrequency catalog={state.action_catalog} counts={val.joint_counts} total={val.transitions}/></Card></div>
    <Card title="无卡牌动作投影" note="原始 v4 六列按键只取 ABCD；切卡、用卡均不进入 Joint144 标签。">
      <DataTable rows={[{name:'训练集',...train},{name:'验证集',...val}]} columns={[{key:'name',title:'集合'},{key:'card_input_ignored_rows',title:'含已忽略卡键的行数',render:v=>v==null?'未记录':number(v,0)}]}/>
      <p className="muted">保留全部帧，不修改源 NPZ、固定划分或动作对齐。上一帧历史、复制基线及切换帧权重统一基于 Joint144；卡牌独有变化不再算动作切换。此计数包括分片末帧等非监督行，原 REP 中已发生的用卡效果仍然存在。</p>
    </Card>
    <Card title="固定划分清单" note={state.data?.split_hash?`校验 ${state.data.split_hash.slice(0,16)}`:'未准备'}>{error&&<p className="error">{error}</p>}
      <div className="inline-controls"><Button variant="secondary" disabled={!offset} onClick={()=>setOffset(Math.max(0,offset-100))}>上一页</Button><span>{offset+1}–{Math.min(offset+100,files.total)} / {files.total}</span><Button variant="secondary" disabled={offset+100>=files.total} onClick={()=>setOffset(offset+100)}>下一页</Button></div>
      <DataTable rows={files.rows} columns={[{key:'name',title:'NPZ'},{key:'split',title:'集合',render:v=>v==='train'?'训练':'验证'},{key:'frames',title:'状态帧'},{key:'transitions',title:'有效动作帧'},{key:'card_input_ignored_rows',title:'含忽略卡键行',render:v=>v==null?'未记录':number(v,0)}]}/>
    </Card>
  </>;
}

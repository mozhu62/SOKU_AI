import {Card,DataTable,Stat} from '../components/common';
import {number,percent,type Row} from '../lib/utils';

const labels:Record<string,string>={idle:'等待请求',selecting:'正在切卡',await_use_confirmation:'已请求用卡，等待确认',completed:'游戏确认已使用',unavailable:'目标不在手中或观测无效',timeout:'执行超时',target_invalid:'目标已离开手牌',interrupted:'执行中断',missing_card_frame:'卡牌快照未匹配',stale_prediction:'预测过期',control_changed:'控制状态改变',control_not_ready:'控制未就绪'};
export function CardDecision({state}:{state:Row}){
  const c=state.cards,p=state.prediction||{},macro=p.card_macro;
  const catalog:Row[]=c?.catalog||p.spell_catalog||[];
  const name=(id:any)=>id==null?'无':`${catalog.find(card=>card.id===id)?.name||'未知卡'} (${id})`;
  const rows=[{id:null,name:'NONE（不请求用卡）'},...catalog].map((card,index)=>({...card,
    probability:p.spell_probabilities?.[index],logit:p.spell_logits?.[index],chosen:p.spell_class===index,
  })).sort((a,b)=>(b.probability??-1)-(a.probability??-1));
  return <Card title="卡牌决策与宏执行" note="卡牌头选择目标；宏切到目标后请求用卡。按键发送不等于游戏确认使用。">
    {!c?<p className="notice">尚未收到卡牌诊断；请加载模型。已加载仍无数据时，请更新并重启推理后端。</p>:!c.enabled?<p className="notice">当前模型没有卡牌头，不会输出或执行符卡。请选择启用卡牌训练的灵梦 checkpoint。</p>:<>
      <p>{c.status} · 快照帧 {c.frame??'未记录'}</p>
      <div className="stats-grid compact">
        <Stat title="最近卡牌头选择" value={p.spell_class==null?'未推理':p.spell_class===0?'NONE（不请求用卡）':name(p.spell_card_id)} note={percent(p.spell_probabilities?.[p.spell_class])}/>
        <Stat title="当前宏状态" value={labels[c.macro_state]||c.macro_state||'未记录'} note={`锁定目标：${name(c.target_card_id)}`}/>
        <Stat title="当前选中手牌" value={name(c.selected_id)} note={`手牌：${c.hand_ids==null?'未记录':c.hand_ids.map(name).join('、')||'空'}`}/>
      </div>
      <p>DLL 可用卡：{c.available_ids==null?'未记录':c.available_ids.map(name).join('、')||'空'}</p>
      <p className="muted">最近执行：{macro?.kind??'尚未执行'} · {p.execution_reason??'等待推理'}。暂停后保留的模型输出不是当前按键。</p>
      <DataTable rows={rows} columns={[{key:'name',title:'卡牌选项'},{key:'id',title:'Card ID',render:v=>v??'NONE'},{key:'probability',title:'模型概率',render:percent},{key:'logit',title:'原始 logit',render:v=>number(v,4)},{key:'chosen',title:'最近选择',render:v=>v?'已选择':'—'}]}/>
    </>}
  </Card>;
}

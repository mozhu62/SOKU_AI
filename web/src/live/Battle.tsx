import {useState} from 'react';
import {Card,DataTable,Stat} from '../components/common';
import {JointFrequency} from '../components/JointActions';
import {number,type Row} from '../lib/utils';
import {ResourceInputs} from './ResourceInputs';

export const directions=['左下','下','右下','左','无方向','右','左上','上','右上'];
export const buttonNames=['体术','DASH','轻弹幕','重弹幕','切卡','使用符卡'];
export const buttonKeys=['melee','dash','light_projectile','heavy_projectile','change_card','use_spell_card'];

export function Battle({state}:{state:Row}){
  const p=state.prediction||{},game=state.game||{},round=state.current_round||{},summary=state.summary||{};
  const [limit,setLimit]=useState(20);
  const jointRows=(state.action_catalog||[]).map((name:string,id:number)=>({id,name,q:p.joint_q?.[id],selected:p.joint_action_id===id,
    chosen:summary.joint_counts?.[id]??0,current:round.joint_counts?.[id]??0,readback:round.readback_joint_counts?.[id]??0}))
    .sort((a:Row,b:Row)=>(b.q??-Infinity)-(a.q??-Infinity)||a.id-b.id).slice(0,limit);
  const actual=Object.entries(round.actual_actions||{}).map(([id,count])=>({id,count})).sort((a,b)=>Number(b.count)-Number(a.count));
  return <><div className="section-intro"><div><h1>战斗与动作</h1><p>432 个单帧 Controller State 的原始 Q；没有宏动作，不把 Q 当概率。</p></div></div>
    <div className="stats-grid"><Stat title="本局造成 / 受到" value={`${number(round.damage_dealt,0)} / ${number(round.damage_taken,0)}`}/><Stat title="模型指令" value={p.direction?`${p.direction} ${directions[p.direction-1]}`:'等待决策'} note={p.buttons?buttonNames.filter((_,i)=>p.buttons[i]).join(' + ')||'无按钮':'—'}/><Stat title="实际游戏动作 / 动作帧" value={`${number(game.actual_action,0)} / ${number(game.action_frame,0)}`}/><Stat title="已发送按键" value={state.control?.pressed_keys?.join(' + ')||'无'} note={`游戏回读方向 ${game.readback?.[0]??'—'}，按钮 ${game.readback?.[1]?.join(' ')||'—'}`}/></div>
    <Card title="Joint Q 排序" note={`已选 ${p.action??'未记录'}；上一帧实际输入 ${p.previous_joint_action_id===432?'START':state.action_catalog?.[p.previous_joint_action_id]??'未记录'}；上一帧方向持续量 ${number(p.previous_action_duration,3)}（已 /60）`}>
      <label className="inline-controls">展示范围 <select value={limit} onChange={e=>setLimit(Number(e.target.value))}><option value={20}>Top 20 Q</option><option value={50}>Top 50 Q</option><option value={432}>全部 432</option></select></label>
      <DataTable rows={jointRows} columns={[{key:'name',title:'完整 Controller State'},{key:'id',title:'Joint ID'},{key:'q',title:'原始 Q',render:v=>number(v,5)},{key:'selected',title:'本次选择',render:v=>v?'已选择':'—'},{key:'chosen',title:'会话选择数',render:v=>number(v,0)},{key:'current',title:'本局选择数',render:v=>number(v,0)},{key:'readback',title:'本局回读帧数',render:v=>number(v,0)}]}/></Card>
    <div className="two-columns"><Card title="本局模型指令 Top-N"><JointFrequency catalog={state.action_catalog} counts={round.joint_counts} total={round.decisions}/></Card><Card title="本局实际控制回读 Top-N"><JointFrequency catalog={state.action_catalog} counts={round.readback_joint_counts} total={round.observed_frames}/></Card></div>
    <Card title="实际动作进入记录" note="来自游戏 actionId/动作帧变化；不把攻击键输入等同于成功出招或命中"><DataTable rows={actual} columns={[{key:'id',title:'游戏 actionId'},{key:'count',title:'本局进入次数',render:v=>number(v,0)}]}/></Card>
    <ResourceInputs usesResources={state.uses_resources} inputs={p.resource_inputs}/>
    <Card title="原始连接诊断"><pre>{JSON.stringify({game,control:state.control,prediction:p},null,2)}</pre></Card>
  </>;
}

import {Card,Stat,DataTable} from '../components/common';
import {Trend} from '../components/Trend';
import {ActionHistoryPanel} from '../components/ActionHistoryPanel';
import {useHistory,type Row} from '../api';
import {number,percent} from '../lib/utils';

export function Overview({state}:{state:Row}){
  const train=useHistory('train'),validation=useHistory('validation');
  const last=state.latest_train||{},val=state.latest_validation||{};
  const total=state.config?.training?.total_steps;
  const elapsed=Object.values(state.timings||{}).reduce<number>((sum,x)=>sum+(typeof x==='number'?x:0),0);
  const timingLabels:Row={data_wait:'等待 CPU 数据',optimization:'搬运 / 优化 / 诊断',validation:'离线验证',save:'模型保存',paused:'暂停等待'};
  const merged=[...train.rows.map(x=>({...x,train_nll:x.nll})),...validation.rows.map(x=>({...x,val_nll:x.nll}))].sort((a:Row,b:Row)=>a.time-b.time);
  return <>
    <div className="section-intro"><div><h1>行为克隆训练</h1><p>模仿 REP 中的完整单帧按键 · 随机初始化 · 阶段 {state.stage??0}</p></div><span className="run-badge">{state.device||'设备准备中'}{state.amp?' / AMP':''}</span></div>
    <div className="stats-grid">
      <Stat title="训练进度" value={`${number(state.step,0)} / ${number(total,0)}`} note={`实际优化 ${number(state.updates,0)} 次`}/>
      <Stat title="验证 NLL ↓" value={number(val.nll,5)} note={val.samples?`${number(val.samples,0)} 个有效帧 · Step ${val.step}`:'尚未验证'}/>
      <Stat title="验证完整按键一致率 ↑" value={percent(val.joint_accuracy)} note={`Top-5 ${percent(val.joint_top5)} · 多数动作基线 ${percent(val.majority_baseline_accuracy)}`}/>
      <Stat title="有效训练帧 / 秒" value={number(last.samples_per_second,0)} note={`${number(last.steps_per_second,2)} step/s · 最近 ${last.throughput_window_steps??'—'} 步活动时间`}/>
    </div>
    <progress className="run-progress" aria-label="训练完成比例" max={total||1} value={state.step||0}/>
    {(train.error||validation.error)&&<p className="error">{train.error||validation.error}</p>}
    <div className="two-columns">
      <Trend title="训练与验证的模仿误差" rows={merged} series={[{key:'train_nll',name:'训练 NLL',color:'#42d6b0'},{key:'val_nll',name:'验证 NLL',color:'#70a7ff'}]} note="NLL 为未平滑交叉熵；越低表示给予专家动作的概率越高。均匀预测约为 6.07"/>
      <Trend title="验证动作一致率" rows={validation.rows} series={[{key:'joint_accuracy',name:'Top-1',color:'#42d6b0'},{key:'joint_top5',name:'Top-5',color:'#70a7ff'},{key:'majority_baseline_accuracy',name:'多数动作基线',color:'#edbf6e'},{key:'val_previous_action_baseline',name:'上一帧复制基线（全量）',color:'#b7a4ed',connectNulls:false},{key:'val_action_change_accuracy',name:'切换帧 Top-1',color:'#f38caa',connectNulls:false}]} note="总体一致率与真正改变操作的能力分开观察；全量基线和本次验证抽样范围见下方"/>
    </div>
    <ActionHistoryPanel state={state}/>
    <div className="two-columns">
      <Trend title="吞吐趋势" rows={train.rows} series={[{key:'samples_per_second',name:'有效帧 / 秒',color:'#42d6b0'}]}/>
      <Card title="时间花在哪里" note="累计活动时间与暂停时间分开"><div className="time-list">{Object.entries(state.timings||{}).map(([key,value])=><div key={key}><div><span>{timingLabels[key]||key}</span><span>{number(value,1)} s · {percent(elapsed?Number(value)/elapsed:null)}</span></div><progress max={elapsed||1} value={Number(value)}/></div>)}</div><p className="muted">取数时间仅统计实际等待；预取与 GPU 可以重叠。初始化扫描不计入。</p></Card>
    </div>
    <Card title="如何判断有没有学到">
      {val.samples&&val.joint_accuracy<=val.majority_baseline_accuracy&&<p className="notice">验证 Top-1 {percent(val.joint_accuracy)} 尚未超过固定多数动作基线 {percent(val.majority_baseline_accuracy)}；样本 {number(val.samples,0)}。查看各动作召回率，避免把只学会常见输入误判为进步。</p>}
      {last.optimizer_skipped&&<p className="notice">最近记录步因 AMP 梯度溢出跳过更新；查看模块梯度与 logits 最大值。</p>}
      {last.data_wait_seconds>last.optimization_seconds&&<p className="notice">最近记录步取数等待 {number(last.data_wait_seconds,3)} s，大于优化 {number(last.optimization_seconds,3)} s；缓存命中率 {percent(last.cache_hit_rate)}。</p>}
      <p>优先看验证 NLL 是否降低、完整动作一致率及各动作召回率是否提升。只有训练误差下降而验证误差上升，可能在过拟合。</p>
      <p className="muted">BC 学的是高手在相似状态下怎么按键，不优化伤害或胜负；离线模仿更准确不等于实战必然更强。归一化熵是分布诊断，不是在线探索奖励。</p>
    </Card>
    <Card title="最近验证记录" note="固定种子、同阶段相同样本计划；有放回抽样，样本数包含重复帧"><DataTable rows={validation.rows.slice(-12).reverse()} columns={[{key:'step',title:'更新步'},{key:'stage',title:'阶段'},{key:'samples',title:'有效帧'},{key:'nll',title:'NLL',render:v=>number(v,5)},{key:'joint_accuracy',title:'Top-1',render:percent},{key:'joint_top5',title:'Top-5',render:percent},{key:'val_previous_action_baseline',title:'上一帧复制基线',render:percent},{key:'val_action_change_accuracy',title:'切换 Top-1',render:percent},{key:'val_action_change_top5_accuracy',title:'切换 Top-5',render:percent},{key:'val_action_change_samples',title:'切换帧数'},{key:'val_action_change_fraction',title:'切换帧占比',render:percent},{key:'macro_recall',title:'已覆盖动作宏召回',render:percent},{key:'represented_actions',title:'覆盖动作数'}]}/></Card>
  </>;
}

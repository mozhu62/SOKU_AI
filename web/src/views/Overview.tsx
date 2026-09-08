import {Card,Stat,DataTable} from '../components/common';
import {Trend} from '../components/Trend';
import {useHistory,type Row} from '../api';
import {number,percent} from '../lib/utils';

export function Overview({state}:{state:Row}){
  const train=useHistory('train'),validation=useHistory('validation');
  const last=state.latest_train||{},val=state.latest_validation||{};
  const total=state.config?.training?.total_steps;
  const elapsed=Object.values(state.timings||{}).reduce<number>((sum,x)=>sum+(typeof x==='number'?x:0),0);
  const timingLabels:Row={data_wait:'等待 CPU 数据',optimization:'搬运 / 优化 / 诊断',validation:'离线验证',save:'模型保存',paused:'暂停等待'};
  return <>
    <div className="section-intro"><div><h1>离线训练总览</h1><p>固定 REP 数据 · 验证集不参与梯度更新 · 阶段 {state.stage??0}</p></div><span className="run-badge">{state.device||'设备准备中'}{state.amp?' / AMP':''}</span></div>
    <div className="stats-grid">
      <Stat title="训练进度" value={`${number(state.step,0)} / ${number(total,0)}`} note={`实际优化 ${number(state.updates,0)} 次`}/>
      <Stat title="验证 TD 均方误差" value={number(val.td_mse,6)} note={val.samples?`${number(val.samples,0)} 个有效转移 · Step ${val.step}`:'等待固定验证'}/>
      <Stat title="验证解释方差 EV" value={val.samples&&val.target_variance<=1e-8?'不适用':number(val.ev,4)} note="目标方差不足时不计算"/>
      <Stat title="有效训练样本 / 秒" value={number(last.samples_per_second,0)} note={`${number(last.steps_per_second,2)} step/s · 最近记录步`}/>
    </div>
    <progress className="run-progress" aria-label="训练完成比例" max={total||1} value={state.step||0}/>
    {(train.error||validation.error)&&<p className="error">{train.error||validation.error}</p>}
    <div className="two-columns">
      <Trend title="训练目标的组成" rows={train.rows} series={[{key:'loss',name:'总损失',color:'#70a7ff'},{key:'td_loss',name:'TD Huber',color:'#42d6b0'},{key:'cql_gap',name:'CQL 原始差值',color:'#edbf6e'},{key:'expert_imitation_contribution',name:'微量模仿贡献',color:'#c296ef'}]} note="总损失 = TD + α × CQL + 微量模仿贡献（β × CQL / T）；不是额外扣血奖励"/>
      <Trend title="训练与验证的 TD 误差" rows={[...train.rows.map(x=>({...x,train_mse:x.td_mse})),...validation.rows.map(x=>({...x,val_mse:x.td_mse}))].sort((a:Row,b:Row)=>a.time-b.time)} series={[{key:'train_mse',name:'训练 MSE',color:'#42d6b0'},{key:'val_mse',name:'验证 MSE',color:'#70a7ff'}]} note="目标网络会变化；不能等同于实战胜率"/>
    </div>
    <div className="two-columns">
      <Trend title="吞吐趋势" rows={train.rows} series={[{key:'samples_per_second',name:'有效转移 / 秒',color:'#42d6b0'}]}/>
      <Card title="时间花在哪里" note="累计墙钟时间 / 各阶段分开统计"><div className="time-list">{Object.entries(state.timings||{}).map(([key,value])=><div key={key}><div><span>{timingLabels[key]||key}</span><span>{number(value,1)} s · {percent(elapsed?Number(value)/elapsed:null)}</span></div><progress max={elapsed||1} value={Number(value)}/></div>)}</div><p className="muted">后台预取与 GPU 优化可以重叠；这里的取数时间仅统计训练线程实际等待。初始化预读时间不计入。</p></Card>
    </div>
    <Card title="当前诊断">
      {val.ev!=null&&val.ev<0&&<p className="notice">验证 EV 为 {number(val.ev,4)}，低于恒定预测该批目标均值的基准 0。样本量 {number(val.samples,0)}；查看「学习诊断」中的 Q 和目标分布。</p>}
      {last.optimizer_skipped&&<p className="notice">最近记录步因 AMP 梯度溢出跳过优化器更新；查看梯度和 Q 最大值。</p>}
      {last.data_wait_seconds>last.optimization_seconds&&<p className="notice">最近记录步取数等待 {number(last.data_wait_seconds,3)} s，大于优化 {number(last.optimization_seconds,3)} s；缓存命中率 {percent(last.cache_hit_rate)}。</p>}
      <p className="muted">离线误差、专家动作一致率能帮助检查训练；仅凭这些指标无法确定模型在游戏里更强。模型在 432 个完整动作上直接 argmax，不使用在线探索温度。</p>
    </Card>
    <Card title="最近验证记录" note="固定随机种子；相同阶段重复使用同一验证样本计划"><DataTable rows={validation.rows.slice(-12).reverse()} columns={[{key:'step',title:'更新步'},{key:'stage',title:'阶段'},{key:'samples',title:'有效转移'},{key:'td_mse',title:'TD MSE'},{key:'td_mae',title:'TD MAE'},{key:'ev',title:'EV',render:(v,row)=>row.samples&&row.target_variance<=1e-8?'不适用':number(v,4)},{key:'joint_accuracy',title:'完整按键一致',render:percent}]}/></Card>
  </>;
}

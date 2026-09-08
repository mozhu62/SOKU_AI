import {Card,DataTable,Stat} from '../components/common';
import {Trend} from '../components/Trend';
import {useHistory,type Row} from '../api';
import {number,percent} from '../lib/utils';
import {JointFrequency} from '../components/JointActions';

export const moduleLabels:Row={current_encoder:'当前状态（含资源 / 上帧输入）',object_encoder:'弹幕集合编码器',fusion:'特征融合',gru:'GRU 记忆',memory_fusion:'记忆融合',joint_head:'Joint Q 头 · 432'};

export function Diagnostics({state}:{state:Row}){
  const {rows,error}=useHistory('train'),last=state.latest_train||{};
  const n=last.samples;
  const modules=Object.entries(state.parameter_counts||{}).map(([key,count])=>({name:moduleLabels[key]||key,count,gradient:last.module_gradients?.[key],change:last.module_changes?.[key],frozen:state.config?.training?.frozen_modules?.includes(key)?'冻结':'训练'}));
  return <><div className="section-intro"><div><h1>学习诊断</h1><p>最近记录步的更新前 Q；梯度和权重变化来自同一次更新。</p></div><span>Step {number(last.step,0)}</span></div>
    {error&&<p className="error">{error}</p>}
    <div className="stats-grid"><Stat title="训练 TD MSE / MAE" value={`${number(last.td_mse,5)} / ${number(last.td_mae,5)}`}/><Stat title="CQL gap" value={number(last.cql_gap,4)} note={`T·logsumexp(Q/T) − Q_data；α=${number(last.cql_alpha,3)}`}/><Stat title="验证 EV" value={state.latest_validation?.samples&&state.latest_validation.target_variance<=1e-8?'不适用':number(state.latest_validation?.ev,4)} note="目标方差为零时不适用"/><Stat title="完整 Joint expert agreement" value={percent(last.joint_accuracy)} note={`验证 ${percent(state.latest_validation?.joint_accuracy)}`}/></div>
    <div className="stats-grid"><Stat title="Q_data mean / std" value={`${number(last.q_data_mean,4)} / ${number(last.q_data_std,4)}`}/><Stat title="Q_max mean / std" value={`${number(last.q_max_mean,4)} / ${number(last.q_max_std,4)}`}/><Stat title="Q_max − Q_data" value={number(last.q_max_minus_q_data,4)} note={`全部联合动作最大 |Q| ${number(last.q_abs_max,4)}`}/><Stat title="Neutral 数据 / 模型占比" value={`${percent(last.neutral_data_fraction)} / ${percent(last.neutral_pred_fraction)}`} note="仅 5 + 无战斗按钮 + NONE（ID 192）"/></div>
    <div className="two-columns"><Trend title="Q 与 Bellman 目标" rows={rows} series={[{key:'q_data_mean',name:'数据动作 Q',color:'#42d6b0'},{key:'q_max_mean',name:'最大 Q',color:'#edbf6e'},{key:'target_mean',name:'目标均值',color:'#70a7ff'}]}/><Trend title="梯度范数" rows={rows} series={[{key:'grad_norm',name:'裁剪前全局范数',color:'#edbf6e'}]}/></div>
    <div className="two-columns"><Card title="同批数据 Joint Action Top-N" note="每个样本只有一个完整动作标签"><JointFrequency catalog={state.action_catalog} counts={last.joint_data} total={n}/></Card>
    <Card title="模型 argmax Joint Action Top-N" note="同批状态上的最大 Q 动作；这是频率，不是策略概率"><JointFrequency catalog={state.action_catalog} counts={last.joint_pred} total={n}/></Card></div>
    <Card title="模块梯度与实际权重变化" note="冻结参数不接收梯度；变化量为一次记录更新的 L2"><DataTable rows={modules} columns={[{key:'name',title:'模块'},{key:'count',title:'参数量'},{key:'frozen',title:'状态'},{key:'gradient',title:'梯度范数',render:v=>number(v,7)},{key:'change',title:'参数变化 L2',render:v=>number(v,9)}]}/></Card>
  </>;
}

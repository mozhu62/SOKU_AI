import {Card,DataTable,Stat} from '../components/common';
import {Trend} from '../components/Trend';
import {useHistory,type Row} from '../api';
import {number,percent} from '../lib/utils';
import {JointFrequency} from '../components/JointActions';

export const moduleLabels:Row={current_encoder:'当前状态（含资源 / 上帧输入）',object_encoder:'弹幕集合编码器',fusion:'特征融合',gru:'GRU 记忆',tcn:'TCN · 连续 32 帧',memory_fusion:'记忆融合',policy_head:'BC 分类头 · 432'};

export function Diagnostics({state}:{state:Row}){
  const {rows,error}=useHistory('train'),last=state.latest_train||{},val=state.latest_validation||{};
  const modules=Object.entries(state.parameter_counts||{}).map(([key,count])=>({name:moduleLabels[key]||key,count,gradient:last.module_gradients?.[key],change:last.module_changes?.[key],frozen:state.module_status?.[key]?.bypassed?'旁路 · 不参与输出':state.module_status?.[key]?.frozen?'冻结 · 仍参与计算':'训练'}));
  const recall=val.joint_data?(state.action_catalog||[]).map((name:string,id:number)=>({id,name,count:val.joint_data[id],correct:val.joint_correct?.[id],recall:val.joint_data[id]>0?val.joint_correct[id]/val.joint_data[id]:null})).sort((a:Row,b:Row)=>b.count-a.count):[];
  return <><div className="section-intro"><div><h1>模仿学习诊断</h1><p>概率来自记录步更新前的同批状态；梯度和参数变化来自该次更新。</p></div><span>Step {number(last.step,0)}</span></div>
    {error&&<p className="error">{error}</p>}
    <div className="stats-grid">
      <Stat title="训练优化 CE / 未平滑 NLL" value={`${number(last.loss,5)} / ${number(last.nll,5)}`} note={`标签平滑 ${state.config?.training?.label_smoothing??'—'}；0 时两者相同`}/>
      <Stat title="完整动作 Top-1 / Top-5" value={`${percent(last.joint_accuracy)} / ${percent(last.joint_top5)}`} note={`${number(last.samples,0)} 个有效帧`}/>
      <Stat title="专家动作平均概率" value={percent(last.expert_probability)} note="同批每帧真实标签的 softmax 概率平均值"/>
      <Stat title="归一化熵" value={number(last.normalized_entropy,4)} note={`0 集中 / 1 均匀；原始熵 ${number(last.entropy,4)}`}/>
    </div>
    <div className="stats-grid">
      <Stat title="方向 / 战斗掩码一致率" value={`${percent(last.direction_accuracy)} / ${percent(last.combat_accuracy)}`} note="从唯一的完整动作预测解码，不是独立输出头"/>
      <Stat title="卡牌命令一致率" value={percent(last.card_accuracy)}/>
      <Stat title="平均最大概率" value={percent(last.confidence_mean)} note={`最大 |logit| ${number(last.logit_abs_max,4)}；高置信度不等于正确`}/>
      <Stat title="Neutral 数据 / 预测占比" value={`${percent(last.neutral_data_fraction)} / ${percent(last.neutral_pred_fraction)}`} note="5 + 无按钮 + NONE（ID 192）"/>
    </div>
    <Card title="动作保持与切换诊断" note="全量基线不使用模型；切换准确率来自最近记录批的有效切换帧，不是额外训练目标">
      <DataTable rows={[
        {name:'训练（记录步）',baseline:last.train_previous_action_baseline??state.data?.train?.previous_action_baseline,accuracy:last.train_action_change_accuracy,top5:last.train_action_change_top5_accuracy,count:last.train_action_change_samples,fraction:last.train_action_change_fraction},
        {name:'固定验证（抽样）',baseline:val.val_previous_action_baseline??state.data?.validation?.previous_action_baseline,accuracy:val.val_action_change_accuracy,top5:val.val_action_change_top5_accuracy,count:val.val_action_change_samples,fraction:val.val_action_change_fraction},
      ]} columns={[{key:'name',title:'统计范围'},{key:'baseline',title:'全量上一帧复制基线',render:percent},{key:'accuracy',title:'切换 Top-1',render:percent},{key:'top5',title:'切换 Top-5',render:percent},{key:'count',title:'切换帧数'},{key:'fraction',title:'切换帧 / 全部有效帧',render:percent}]}/>
    </Card>
    <Trend title="训练：动作切换准确率" rows={rows} series={[{key:'train_action_change_accuracy',name:'切换帧 Top-1',color:'#f38caa',connectNulls:false},{key:'joint_accuracy',name:'总体 Top-1',color:'#42d6b0'},{key:'train_previous_action_baseline',name:'全量复制基线',color:'#b7a4ed',connectNulls:false}]} note="只记录诊断步，批次内容会变化；缺少切换帧时曲线断开，不显示虚假的 0%"/>
    <div className="two-columns"><Trend title="专家概率与预测置信度" rows={rows} series={[{key:'expert_probability',name:'专家动作概率',color:'#42d6b0'},{key:'confidence_mean',name:'最大动作概率',color:'#70a7ff'}]}/><Trend title="梯度范数" rows={rows} series={[{key:'grad_norm',name:'裁剪前全局范数',color:'#edbf6e'}]}/></div>
    <div className="two-columns"><Card title="同批专家动作分布"><JointFrequency catalog={state.action_catalog} counts={last.joint_data} total={last.samples}/></Card><Card title="同批模型 argmax 动作分布" note="频率不是单帧概率；可以切换全部 432 类"><JointFrequency catalog={state.action_catalog} counts={last.joint_pred} total={last.samples}/></Card></div>
    <Card title="验证：全部动作的召回率" note={`Step ${val.step??'—'} · 已覆盖 ${val.represented_actions??'—'} / 432 · 宏平均 ${percent(val.macro_recall)}；未出现的动作不记零分`}><DataTable rows={recall} columns={[{key:'name',title:'完整 Controller State'},{key:'id',title:'ID'},{key:'count',title:'专家样本数'},{key:'correct',title:'正确预测数'},{key:'recall',title:'召回率',render:percent}]}/></Card>
    <Card title="模块梯度与实际权重变化" note="冻结参数清除梯度；变化量是一次更新的 L2，冻结共享主干后仍可单独训练分类头"><DataTable rows={modules} columns={[{key:'name',title:'模块'},{key:'count',title:'参数量'},{key:'frozen',title:'状态'},{key:'gradient',title:'梯度范数',render:v=>number(v,7)},{key:'change',title:'参数变化 L2',render:v=>number(v,9)}]}/></Card>
  </>;
}

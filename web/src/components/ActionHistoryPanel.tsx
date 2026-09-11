import {Card,DataTable} from './common';
import type {Row} from '../api';
import {number,percent} from '../lib/utils';

export function ActionHistoryPanel({state}:{state:Row}){
  const val=state.latest_validation||{},data=state.data?.validation||{};
  const copy=val.val_previous_action_baseline??data.previous_action_baseline;
  const rows=[
    {name:'多数动作基线',value:val.majority_baseline_accuracy,scope:'本次验证全部有效帧；永远猜训练集最常见动作'},
    {name:'上一帧复制基线',value:copy,scope:`全量固定验证集；${number(data.previous_action_samples,0)} 个具有真实上帧动作的位置`},
    {name:'Validation Top-1',value:val.joint_accuracy,scope:`本次验证全部 ${number(val.samples,0)} 个有效帧`},
    {name:'动作切换帧准确率（Top-1）',value:val.val_action_change_accuracy,scope:`本次验证 ${number(val.val_action_change_samples,0)} 个动作切换帧`},
    {name:'Validation Top-5',value:val.joint_top5,scope:'本次验证全部有效帧；真实动作是否进入最高分 5 个候选'},
  ];
  return <Card title="验证动作一致率 · 基线与切换能力" note={`验证 Step ${val.step??'未记录'}；新指标不参与损失或 best 选优`}>
    <DataTable rows={rows} columns={[{key:'name',title:'指标'},{key:'value',title:'准确率',render:percent},{key:'scope',title:'统计范围'}]}/>
    <p>上一帧复制基线不使用模型，直接将高手上一帧完整 Joint Action 作为当前帧预测。它表示只“继续上一帧操作”就能获得的准确率。</p>
    <p>动作切换帧准确率只看专家操作发生变化的位置，衡量移动、Dash、攻击、弹幕等操作之间的切换时机与新动作选择是否预测正确。</p>
    <p className="muted">切换帧占全部有效验证帧：本次抽样 {percent(val.val_action_change_fraction)} / 全量验证集 {percent(val.val_dataset_action_change_fraction??data.action_change_fraction)}。切换帧 Top-5：{percent(val.val_action_change_top5_accuracy)}。</p>
    <details><summary>分母与可比性说明</summary>
      <p>复制基线排除 START/PAD=144、episode/终局/断帧后的片段起点；不跨断点寻找历史。原有总体 Top-1 口径保持不变，仍包含有效起点帧。</p>
      <p>与模型同批、同历史有效位置比较：复制基线 {percent(val.val_batch_previous_action_baseline)}；模型 Top-1 {percent(val.val_history_joint_accuracy)}；可比较帧 {number(val.val_previous_action_samples,0)}。全量基线不会随训练变化，但固定验证抽样与全量数据的比例可能略有差别。</p>
      <p>更高的总体 Top-1 不单独证明学会了操作切换；应结合切换准确率与样本量。旧日志缺少新增字段时显示“未记录”，不会补成零。</p>
    </details>
  </Card>;
}

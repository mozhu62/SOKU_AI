import type {Row} from '../api';
import {useHistory} from '../api';
import {Card,Stat} from './common';
import {Trend} from './Trend';
import {number} from '../lib/utils';

export function PalrDiagnostics({state,rows}:{state:Row;rows:Row[]}){
  const {rows:validation,error}=useHistory('validation');
  const last=state.latest_train||{},val=state.latest_validation||{},cfg=state.config?.palr;
  return <>
    <Card title="PALR · 历史动作泄漏正则" note="只作用于 TCN 历史表示；不改变推理结构，不参与 best 选优">
      <div className="stats-grid">
        <Stat title="PALR Enabled / Alpha" value={cfg?`${cfg.enabled?'开启':'关闭'} / ${number(cfg.alpha,4)}`:'未记录'} note={`配置抽样上限 ${number(cfg?.sample_size,0)}；通过 YAML 配置，重启后生效`}/>
        <Stat title="PALR 原始 / 加权损失" value={`${number(last.loss_palr,7)} / ${number(last.palr_weighted_loss,7)}`} note="关闭时训练正则为 0；不能据此认为历史表示没有泄漏"/>
        <Stat title="关键帧 CE / 总损失" value={`${number(last.loss_keyframe_bc,5)} / ${number(last.loss_total,5)}`} note="总损失 = 关键帧 CE + Alpha × HSCIC"/>
        <Stat title="Validation HSCIC" value={number(val.validation_hscic,7)} note={`PALR 关闭也测量；抽样 ${number(val.validation_palr_sampled_samples,0)} 帧，跳过 ${number(val.validation_palr_skipped_batches,0)} 批`}/>
      </div>
      <p className="muted">最近记录步：PALR 合法 / 抽样 {number(last.palr_eligible_samples,0)} / {number(last.palr_sampled_samples,0)}；
        {last.palr_skipped===undefined?'未记录':last.palr_skipped?`跳过（${last.palr_skip_reason==='disabled'?'配置关闭':'连续有效专家帧不足'}）`:'已计算'}。
        TCN 特征平均 L2 {number(last.temporal_feature_norm,5)}；TCN 总梯度范数 {number(last.tcn_gradient_norm,7)}（不是 PALR 单独梯度）。</p>
      <p className="muted">HSCIC 是固定条件核与抽样规模下的批次估计；低值不等于实战变强，须结合切换准确率、复制率与实战结果。不同核参数和抽样规模不可直接比较。</p>
    </Card>
    {error&&<p className="error">{error}</p>}
    <div className="two-columns">
      <Trend title="训练损失分解" rows={rows} series={[{key:'loss_keyframe_bc',name:'关键帧 CE',color:'#42d6b0',connectNulls:false},{key:'palr_weighted_loss',name:'加权 PALR',color:'#edbf6e',connectNulls:false},{key:'loss_total',name:'总损失',color:'#70a7ff',connectNulls:false}]}/>
      <Trend title="验证：条件历史泄漏 HSCIC" rows={validation} series={[{key:'validation_hscic',name:'Validation HSCIC',color:'#b7a4ed',connectNulls:false}]} note="每批独立抽样，按参与帧数汇总；无有效样本或旧日志缺字段显示断点"/>
    </div>
  </>;
}

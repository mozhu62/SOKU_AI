import {Card,DataTable} from '../components/common';
import {number,type Row} from '../lib/utils';

export function ResourceInputs({usesResources,inputs}:{usesResources?:boolean,inputs?:Row}){
  if(!inputs) return <Card title="技能与卡牌输入"><p className="muted">{usesResources===false?'当前为旧 BC 模型，不使用技能/卡牌资源。':'等待资源版模型完成一次决策；缺失数据不会显示为 0。'}</p></Card>;
  const skills=['self','opponent'].flatMap(side=>(inputs[side]?.skills||[]).map((skill:Row)=>({
    ...skill,side:side==='self'?'己方':'对方',type:skill.valid?['默认','替换 1','替换 2'][skill.variant]:'未知',
    learned:skill.valid?skill.learned_level:null,effective:skill.valid&&skill.effective_level>=0?skill.effective_level:null,
  })));
  return <Card title="本次决策使用的技能与卡牌" note={`采集序号 ${inputs.sample_serial} / 游戏帧 ${inputs.battle_frame}；以下原始值对应最近已发送决策，暂停后保留显示`}>
    <DataTable rows={skills} columns={[{key:'side',title:'侧别'},{key:'command',title:'指令槽'},{key:'type',title:'安装类型'},{key:'learned',title:'学习等级',render:v=>number(v,0)},{key:'effective',title:'生效等级',render:v=>number(v,0)}]}/>
    <div className="two-columns">{['self','opponent'].map(side=><div key={side}><p>{side==='self'?'己方':'对方'}：符卡能量 {number(inputs[side]?.card_gauge,0)} / 卡牌计数 {number(inputs[side]?.card_count,0)} / 手牌数 {number(inputs[side]?.hand_count,0)}</p>
      <DataTable rows={inputs[side]?.hand_selected_first||[]} columns={[{key:'slot',title:'顺序',render:v=>v+1},{key:'selected',title:'当前卡',render:v=>v?'选中':'—'},{key:'id',title:'卡 ID',render:v=>v==null?'未知':number(v,0)},{key:'cost',title:'费用',render:v=>number(v,0)}]}/></div>)}</div>
  </Card>;
}

"""Chinese report generated exclusively from completed, prespecified outputs."""
from pathlib import Path
import json,html,argparse
import pandas as pd
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor,white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=ROOT/'MAIN');ap.add_argument('--destination',type=Path,default=ROOT.parents[1]/'outputs');args=ap.parse_args()
R=args.results;DEST=args.destination
DEST.mkdir(exist_ok=True)
M=json.loads((R/'run_manifest.json').read_text());assert M['status']=='COMPLETE'
checks=json.loads((R/'prediction_check.json').read_text())
summary=pd.read_csv(R/'summary.csv');comp=pd.read_csv(R/'primary_components.csv');early=pd.read_csv(R/'primary_early.csv');short=pd.read_csv(R/'primary_shortfalls.csv')
pdfmetrics.registerFont(TTFont('CJK','/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc',subfontIndex=0))
styles={
 'title':ParagraphStyle('title',fontName='CJK',fontSize=23,leading=32,textColor=HexColor('#152b47'),spaceAfter=13,wordWrap='CJK'),
 'h':ParagraphStyle('h',fontName='CJK',fontSize=14,leading=21,textColor=HexColor('#174c79'),spaceBefore=12,spaceAfter=7,wordWrap='CJK'),
 'body':ParagraphStyle('body',fontName='CJK',fontSize=10,leading=16,spaceAfter=7,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='CJK',fontSize=8.2,leading=12,spaceAfter=5,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='CJK',fontSize=8.3,leading=12,wordWrap='CJK'),
 'headcell':ParagraphStyle('headcell',fontName='CJK',fontSize=8.3,leading=12,textColor=white,wordWrap='CJK')}
story=[]
def p(s,style='body'):return Paragraph(s,styles[style])
def add(s,style='body'):story.append(p(s,style))
def table(rows,widths):
 t=Table([[p(str(x),'headcell' if i==0 else 'cell') for x in row] for i,row in enumerate(rows)],colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#254b70')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.5,white),('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#f0f4f8'),white]),('LINEBELOW',(0,-1),(-1,-1),.5,HexColor('#b8c5d2'))]))
 story.append(t);story.append(Spacer(1,8))
names={'cifar10_cat_dog':'视觉：猫 / 狗','agnews_business_scitech':'文本：商业 / 科技'}
def fmt(x):return f'{x*1000:.2f}'
def verdict(b):return '成立' if b else '不成立'
def interval(row):return f"{fmt(row['mean'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"

add('深度视觉与文本任务<br/>外部验证报告','title')
add('2026-11-11 | DC-PRISM / supplement-v13 的本机外部适配','small')
add('本轮已完成两类公开任务的完整主动学习曲线。模型为冻结的预训练深度编码器加重新训练的非线性分类头；尚不构成端到端微调或完整多分类标准基准复现。')
add('结论：两个任务的测试复用均明显为正，早期运气对照差也均达到实质水平；但“两个任务都随阶段减弱”的联合预测未通过，也没有证实策略相对事前 oracle 的差距已经小于实质阈值。现有证据足以支持前两项有范围限制的发现，不能称为 v13 全部主张的验证成功。')
add('一、先看登记预测','h')
table([['预测','运行前判据（简写）','结果'],['H1','两个任务的整条曲线测试复用均为正（联合区间下界 &gt; 0）',verdict(checks['H1'])],['H2','两个任务的早期运气为正，且早期大于后期；四个联合区间下界均 &gt; 0',verdict(checks['H2'])],['H3','两个任务的全部策略均实质落后于测试 oracle；且各有至少一策略相对事前 oracle 的联合区间上界 ≤ 0.0005',verdict(checks['H3'])]],[42,407,62])
add('“不成立”表示登记的联合判据没有全部满足，不能自动推导其反面。H3 使用上界证明差距小于阈值；仅仅检验不显著并不算通过。')
add('二、完成规模与本机消耗','h')
add(f"24 条正式轨迹（2 任务 × 12 种子），每条 30 步，6 种方法，共 4,464 个学习曲线评分点。正式运行用时 {M['elapsed_seconds']/60:.1f} 分钟。PyTorch 训练过程显存分配峰值 {M['peak_gpu_memory_bytes']/2**30:.2f} GiB（不含桌面和驱动占用）；本机 RTX 5060 Laptop 总显存约 8 GiB。")
add('五个 Claude 本地提交已按文件树核验导入 GitHub 研究分支，包含 v12/v13 代码、协议、汇总指标、论文版本及已授权公开的内部审稿笔记。v13 的 12 张保存统计表重新计算后全部匹配。原始数据实验没有重新运行。')

story.append(PageBreak())
add('三、实验设计与信息隔离','h')
table([['项目','设定'],['视觉任务','CIFAR-10 猫/狗二分类；MobileNetV3-Small 的冻结图像特征，1,024 维。编码器约 152 万参数，含原分类器的完整模型约 250 万参数。'],['文本任务','AG News Business/Sci-Tech 二分类；冻结 MiniLM-L6，约 2,271 万参数，384 维均值池化特征，文本最多 128 token。'],['统一学习器','只用辅助集拟合 PCA/白化至 32 维；32 隐单元 ReLU 分类头，每次从共同初始化重新拟合 60 步全批 Adam。所有 oracle 的搜索和训练预算相同。'],['曲线预算','68 个初始标签；每步 40 个新候选中取 10 个，30 步后 368 标签；事前 oracle 使用 32 份标签复本。'],['每条轨迹的数据角色','官方训练集内：辅助集 2,000、验证集 512、参考集 256、初始历史 68、候选流 1,200，全部互斥；官方测试集抽 512 个记录评分。'],['结果模型与基线','GBDT 只用辅助集标签拟合；Random、Uncertainty、CoreSet 三个策略。基线集合与 v13 的五策略集合不同。']],[86,425])
add('三种 oracle 的选择规则','h')
add('测试 oracle：用候选真实标签，在测试集上贪心选样。拆分 oracle：同样使用候选真实标签，但在独立验证集上选样。事前 oracle：候选标签由辅助集结果模型产生的 32 个伯努利复本替代，在验证集上按平均损失选样，选定后才加入真实标签。')
add('各方法沿自己的轨迹累计标签，所有曲线用同一测试集的真实标签评分。整条曲线损失为第 1-30 步 Brier 的均值；早期为第 1-10 步，后期为第 21-30 步。')
add('隔离核验：各角色索引互斥，预处理和结果模型不读取验证/测试标签；视觉二分类子集无精确像素重复，文本提取前移除了 2 条精确重复文本。未系统排查近重复和预训练语料污染。')
add('协议与源代码在本地正式运行前冻结。GitHub 归档发生在启动后、检查正式端点前；这不属于前瞻性的公开预注册。所有 24 条轨迹按固定种子完成，无端点窥视、无追加种子。','small')

story.append(PageBreak())
add('四、完整学习曲线与整段平均结果','h')
story.append(Image(str(R/'learning_curves.png'),width=511,height=171))
add('图中阴影是跨种子的标准误，仅用于描述；正式检验使用下表的多重比较校正区间。分数越低越好。','small')
rows=[['任务','测试 oracle','拆分 oracle','事前 oracle','Random','Uncertainty','CoreSet']]
for _,r in summary[summary.window=='all'].iterrows():rows.append([names[r.dataset]]+[fmt(r[k]) for k in ['oracle-test','oracle-split','oracle-exante','Random','Uncertainty','CoreSet']])
table(rows,[95,68,68,68,68,76,68])
add('表内所有数字的单位均为 Brier × 1,000。对照差值大于零，表示后者的测试损失更高。','small')
rows=[['任务','整段分量','均值 [联合 95% 区间]','实质为正？']]
for _,r in comp.iterrows():rows.append([names[r.dataset],'复用' if r.endpoint=='reuse' else '运气对照差',interval(r),'是' if r.material_positive else '未证实'])
table(rows,[120,88,220,83])
add('复用 = 拆分 oracle - 测试 oracle；运气对照差 = 事前 oracle - 拆分 oracle。两者之和加上“策略 - 事前 oracle”，恰好等于“策略 - 测试 oracle”。这是沿各自轨迹的算术分解，不能直接视为已识别的总体因果分解。')

story.append(PageBreak())
add('五、早晚阶段与策略差距','h')
story.append(Image(str(R/'components_over_curve.png'),width=511,height=171))
add('图中剩余差距固定以 Random 为锚，避免按结果选择最佳策略。','small')
rows=[['任务','对照量','均值 [联合 95% 区间]','为正？']]
for _,r in early.iterrows():rows.append([names[r.dataset],'早期运气' if r.endpoint=='early_luck' else '早期 - 后期运气',interval(r),'是' if r.positive else '未证实'])
table(rows,[120,130,200,61])
rows=[['任务','oracle','全部策略实质落后？','至少一策略已证实差距 ≤ 阈值？']]
for ds in names:
 for o,label in zip(['oracle-test','oracle-split','oracle-exante'],['测试','拆分','事前']):
  c=checks[ds];rows.append([names[ds],label,'是' if c['all_materially_short'][o] else '未证实','是' if c['any_certified_not_materially_short'][o] else '未证实'])
table(rows,[115,60,146,190])
add('正式检验：以轨迹种子为配对单位；整段分量 4 个端点、早期及早晚差 4 个端点、策略差距 18 个端点，分别做 Bonferroni 双侧联合区间。登记判定以这些区间为准；另存的 Holm 单侧 p 值是补充结果，不能替代主判据。实质阈值 0.0005 Brier。','small')

story.append(PageBreak())
add('六、证据边界与可复现材料','h')
add('本轮把 v13 的三 oracle、完整曲线和信息隔离设计移到了两个视觉/文本任务。它提供的是“冻结深度表示 + 非线性头”的外部适配证据，不能替代主干网络端到端微调、完整多分类任务或 CDALBench 原配置的复现。')
add('事前 oracle 依赖用额外辅助标签拟合的结果模型。所谓运气对照差受该模型误差、有限标签复本、贪心搜索以及后续轨迹变化共同影响；没有证明任何策略都无法获得这部分收益。负值或区间包含零均如实保留。')
add('两个任务无法检验跨数据集噪声相关规律。置信区间主要描述给定数据集、预训练表示、学习器及划分规则下的种子随机性，不能升级为对总体或任意深度模型的覆盖保证。没有用本轮结果重写或撤回原论文中的主张。')
add('保存材料','h')
add('GitHub 包含：原始五提交到发布提交的映射；v12/v13 冻结协议、代码和结果；压缩指标内容审计；v13 统计重算匹配记录；本轮协议与冻结哈希、模型/数据版本、逐轨迹指标与划分哈希、联合区间、预测核对、完整曲线和复现说明。')
add('原工作目录及未提交论文修改保持原样。GitHub API 导入会产生新的提交编号；每个对应快照的 Git 文件树均与原提交相同。原始个体数据、预训练权重与深度特征缓存未上传。')
add('模型与数据来源（Hugging Face）','h')
for title,url in [('MobileNetV3-Small','https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k'),('MiniLM-L6','https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2'),('CIFAR-10','https://huggingface.co/datasets/uoft-cs/cifar10'),('AG News','https://huggingface.co/datasets/fancyzhx/ag_news')]:
 add(f'<link href="{url}" color="#174c79">{title}</link>：{html.escape(url)}','small')
add('仓库：<link href="https://github.com/Lanceashire/DC-PRISM" color="#174c79">Lanceashire/DC-PRISM</link>。实验目录 experiments/external_v14；结果目录 results/external_v14。','small')

def footer(canvas,doc):
 canvas.setStrokeColor(HexColor('#c3cfda'));canvas.line(42,36,553,36)
 canvas.setFont('CJK',8);canvas.setFillColor(HexColor('#64758a'));canvas.drawString(42,24,'DC-PRISM | 外部验证 | 2026-11-11');canvas.drawRightString(553,24,str(doc.page))

path=DEST/'v14_视觉与文本外部验证报告_20261111.pdf'
doc=SimpleDocTemplate(str(path),pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=48,title='v14 视觉与文本外部验证报告',author='Research artifact')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(path)

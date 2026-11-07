from pathlib import Path
import json,re
P=Path(__file__).parent
for lang in ['en','zh']:
 b=json.loads((P/(lang+'_blocks.json')).read_text())
 if lang=='en':
  abstract='''Oracle selectors in label-acquisition benchmarks can exploit both realized training outcomes and the evaluation set. We distinguish the value of perfect information about hidden training labels from test-set reuse and from the performance of a specified ex-ante selector. For a fixed feasible family, we derive distribution-free population inequalities and their exact empirical analogues; a Gaussian two-set expression is an approximation rather than a universal bound. Across 624 simulated states, the empirical inequalities hold throughout. Eight cohorts and three downstream learners reveal substantial variation in hindsight opportunity. Within-cohort label-flip interventions increase the empirical family floor monotonically in all three tested cohorts, while larger candidate pools show heterogeneous effects. Exact enumeration on 12-candidate pools checks a greedy-oracle approximation, and probability-model perturbations expose sensitivity beyond held-out calibration checks. We retain cross-fitted greedy performance as an operational benchmark, with its optimization and Monte Carlo errors stated explicitly. Delayed replay shows that DC-PRISM can improve on quantity-oriented references while losing to information-greedy rules. The resulting protocol audits which parts of an oracle gap are supported by an evaluation design, rather than treating the entire gap as attainable improvement.'''
  b[0]=re.sub(r'\\begin\{abstract\}.*?\\keywords',lambda m:'\\begin{abstract}\n'+abstract+'\n\\keywords',b[0],flags=re.S)
  b[3]=r'''\paragraph{Contributions.}
\begin{itemize}
\item A protocol distinguishing fixed-family perfect-information value, evaluation-set reuse, and the gain of a specified cross-fitted selector. These quantities have different identification and estimation requirements.
\item Distribution-free inequalities for the fixed-family floor, checked on 624 states, together with exact small-pool enumeration, probability-model sensitivity, and within-cohort label-noise interventions.
\item An empirical audit across eight cohorts and three learners, with delayed replay as a separate outcome check. The evidence documents where hindsight opportunity persists and where practical selection rules fail to exploit it.
\end{itemize}'''
  b[8]=r'''\paragraph{Greedy benchmark diagnostics.} Benchmarks select a batch greedily using realized labels and an evaluation set~\cite{werner2023comparable,werner2024cdalbench}. We split test records into $T_1,T_2$. Let $G_{\rm rep}$ be the greedy selector's mean gain when selecting and scoring on $T_1$, $G_{\rm hold}$ its gain when scoring on $T_2$, and $G_{\rm cf}$ the gain of a greedy selector choosing by mean $T_1$ loss on one bank of training-label replicates and evaluated on $T_2$ using an independent bank, with banks exchanged and averaged. Then
\[
G_{\rm rep}=(G_{\rm rep}-G_{\rm hold})+(G_{\rm hold}-G_{\rm cf})+G_{\rm cf}.
\]
This is an accounting identity. The first contrast diagnoses evaluation-set reuse; the second is a \emph{greedy residual}, which also contains search and finite-simulation effects. It equals the fixed-family floor only when the respective selectors optimize a common family and their objectives and evaluation distributions agree. We therefore do not identify this residual as an irreducible floor or call $G_{\rm cf}$ the optimal deployable value.'''
  b[9]=r'''\paragraph{Valid fixed-family inequalities.} Write $d_S=\bar L(M;Y)-\bar L(S;Y)$, $\mu_S=\mathbb E d_S$, and let $b$ maximize $\mu_S$; let $r$ be another set. The family floor is $F=\mathbb E\max_S d_S-\max_S\mu_S$. Pointwise comparison with $b,r$, and centering each gain, give
\begin{equation}
\mathbb E(d_r-d_b)_+\ \leq F\ \leq\mathbb E\max_S(d_S-\mu_S).\label{eq:two}
\end{equation}
Indeed, $\max_S d_S-d_b\ge(d_r-d_b)_+$, whereas $\max_S d_S\le\max_S\mu_S+\max_S(d_S-\mu_S)$. Replacing expectations and means by the same empirical distribution gives exact sample inequalities, including when $b,r$ are selected from that sample. These empirical statements do not provide confidence coverage for population $F$. If the centered gains are jointly Gaussian, arbitrary dependence is allowed and $F\le\max_S\sigma_S\sqrt{2\log|\mathcal F|}$. If $d_b-d_r$ is normal with mean $\Delta$ and standard deviation $s$, the lower expression is $s\varphi(\Delta/s)-\Delta\Phi(-\Delta/s)$. Normality is an additional modeling assumption. An independent-normal construction is not a universal upper bound.'''
  b[10]=r'''\paragraph{Estimation and model substitution.} With known $p$, common random numbers generate training-label replicates. For a fixed family, maximizing estimated mean gain gives an upward-biased estimate of $H_{\rm pre}$ in expectation. Selecting on one label bank and evaluating on another yields the expected value of that finite-bank selector, at most $H_{\rm pre}$. Neither expectation statement is a deterministic finite-sample bracket. On real data, a boosting model fitted on disjoint auxiliary records supplies $\hat p$; recorded test labels remain fixed. This evaluates opportunity conditional on that test set and probability model. Comparing simulated and recorded-outcome hindsight gaps tests one functional of the model, not full identification of the floor. Taking the per-state maximum of independently noisy policy-value estimates can itself introduce selection optimism; such post-hoc maxima are sensitivity summaries, not certified lower bounds on achievable value.'''
  b[14]=r'''\paragraph{Recorded-outcome checks and sensitivity.} A confidence interval containing zero does not establish agreement. In the revision we use equivalence tests with margin $0.5$ in the displayed units. Pooled checks for logistic and boosting learners support equivalence, while only two of 16 individual cohort--learner cells establish it; the neural learner does not establish pooled equivalence. In 192 additional greedy-calibration states, the pooled reported-gap difference has 90\% interval $[-0.089,0.416]$, within the margin, whereas the held-out-gap interval $[-0.138,0.513]$ crosses it. Known-probability perturbations give mean floor bias about $-0.04$ at the unperturbed model, but changing probability sharpness shifts bias by approximately $0.5$--$0.8$. Disjoint fitting is essential but does not establish robustness to probability-model misspecification.'''
  b[15]=b[15].replace('The decomposition does not depend on the family.','Family sensitivity.').replace('leaves the deployable share','in these experiments leaves the estimated deployable share')
  b[16]=r'''\section{Findings}
\label{sec:findings}
\paragraph{The greedy benchmark gap shrinks under held-out selection.} Table~\ref{tab:known} retains the original cross-fitted greedy estimates. Its columns implement the accounting identity in Section~\ref{sec:protocol}; the residual column includes approximation error and is not the fixed-family $F$. On semi-synthetic states the reported lead over the best fixed strategy is $6.9$, while the cross-fitted greedy lead is approximately zero. On recorded-outcome states, correction reduces the apparent lead substantially. This demonstrates sensitivity to evaluation design, without proving that the entire removed component is irreducible.'''
  b[18]=r'''\paragraph{Noise ordering and a direct intervention.} The original cross-cohort study registered a relationship between the noise index and the greedy residual share (Table~\ref{tab:gradient}). Its interpretation depends on the ex-ante comparator: using a post-hoc strengthened comparator changes the logistic rank correlation to $0.93$ and the neural correlation to $0.52$ ($p=0.15$). The latter no longer supplies clear cross-cohort evidence. A distinct registered intervention flips labels with probability $0,0.1,0.2,0.3$ within HMEQ, Bank Marketing and USPS, using 16 states per dose and cohort. The empirical fixed-family floor increases monotonically in all three cohorts (Figure~\ref{fig:floor}b). This intervention supports a response to randomized label corruption within the specified simulation; it does not establish a universal monotonic relationship for natural label noise.'''
  b[20]=r'''\paragraph{Bounds, search, and scale checks.} The corrected empirical lower and upper inequalities hold in all 624 revision states. Their validity is algebraic; this check verifies implementation, not a new population confidence guarantee. Stress tests with negative dependence, skewness and heavy tails invalidate the former independent-normal shortcut. In 12-candidate pools, exact enumeration of all feasible sets (about 730 on average) yields similar aggregate greedy diagnostics: the mean residual difference is approximately $-0.02$ and the ex-ante gain difference $0.01$ in displayed units. This checks approximation at that scale. Candidate pools of 24, 48 and 96 produce heterogeneous floor changes, so no large-pool approximation guarantee follows.'''
  b[22]=r'''\paragraph{Budget sensitivity is empirical.} Increasing a budget can enlarge a feasible family, but $F=H_{\rm post}-H_{\rm pre}$ is a difference of two changing quantities and need not increase. In the registered budget experiment, varying the target count from 3 to 6 to 12 increased the measured family floor in all eight cohorts, with paired mean increase $0.74$ $[0.67,0.82]$ in displayed units (Figure~\ref{fig:budget}). This is a result for the tested distributions and learner, rather than a consequence of~\eqref{eq:two}.'''
  b[32]=r'''\section{Discussion and Conclusion}
\label{sec:conclusion}
Oracle gaps depend on what the selector may observe and how its performance is evaluated. For a fixed family, the value of observing hidden training outcomes is distinct from ex-ante acquisition value. Greedy benchmark comparisons additionally mix evaluation-set reuse, search error and finite-simulation selection. The corrected inequalities, noise interventions and approximation checks support an evaluation audit with explicit assumptions. A useful report pairs held-out oracle performance with fixed-family diagnostics and sensitivity to the probability model. DC-PRISM's mixed replay performance illustrates why substantial hindsight opportunity alone cannot establish the effectiveness of an acquisition strategy.'''
  b[33]=r'''\paragraph{Limitations.} Probability-model substitution does not identify the real conditional outcome law, and test-label conditioning is weaker than population predictive-risk evaluation. Cross-fitted greedy performance includes optimization and Monte Carlo error; post-hoc maxima over policy estimates are optimistic. Exact enumeration covers 12-candidate pools only. Random label corruption is a controlled intervention rather than a complete model of natural uncertainty. Pools and learners remain modest; larger pretrained models and verified calendar-time cohorts were not studied. Baselines are explicit adaptations. The risk-envelope probability interpretation additionally requires conditional independence and valid upper risk scores, which the empirical scores do not establish. Public source and local results have different coverage; anonymous artifact access must be prepared and tested before submission.'''
 else:
  abstract='''标签获取基准中的 oracle 可能同时利用已实现训练结果与评价集。本文区分隐藏训练标签的完全信息价值、测试集复用，以及指定事前选择器的表现。对于固定可行集合族，我们给出不依赖分布的总体不等式及其严格的经验版本；两集合高斯表达式仅在相应假设下适用。624 个模拟状态均通过经验不等式检查。八个数据集与三类下游学习器呈现不同程度的事后机会。在三个数据集内实施随机标签翻转干预时，经验族下限均随翻转强度单调增大，而扩大候选池的效果并不一致。12 候选池的精确枚举检验了贪心近似，概率模型扰动则揭示了留出校准检验之外的敏感性。我们将交叉拟合贪心收益作为操作性基准，并明确其搜索误差与蒙特卡洛误差。延迟回放显示，DC-PRISM 可改善数量导向基准，也可能落后于信息贪心规则。由此，本文审核 oracle 差距的各部分由何种评价设计支持，而不将整个差距都解释为可实现的提升。'''
  b[0]=re.sub(r'\\begin\{abstract\}.*?\\keywords',lambda m:'\\begin{abstract}\n'+abstract+'\n\\keywords',b[0],flags=re.S)
  b[3]=r'''\paragraph{贡献。}
\begin{itemize}
\item 建立区分固定族完全信息价值、评价集复用与指定交叉拟合选择器收益的协议，并明确它们不同的识别与估计条件。
\item 给出固定族下限的不依赖分布的不等式，在 624 个状态上检查实现，并结合小池精确枚举、概率模型敏感性及数据集内标签噪声干预。
\item 在八个数据集与三类学习器上审核事后机会，将延迟回放作为独立结果核验，记录实际策略未能利用机会的情形。
\end{itemize}'''
  b[8]=r'''\paragraph{贪心基准诊断。}基准利用已实现标签与评价集贪心选择批次~\cite{werner2023comparable,werner2024cdalbench}。将测试记录拆为 $T_1,T_2$。令 $G_{\rm rep}$ 为在 $T_1$ 选择并评分的平均收益，$G_{\rm hold}$ 为同一选择器在 $T_2$ 的收益；$G_{\rm cf}$ 为在一个训练标签重复样本库上按平均 $T_1$ 损失选择、在另一独立样本库上用 $T_2$ 评分的贪心规则收益，交换两库后平均。则
\[
G_{\rm rep}=(G_{\rm rep}-G_{\rm hold})+(G_{\rm hold}-G_{\rm cf})+G_{\rm cf}.
\]
这是一条记账恒等式。第一项诊断评价集复用，第二项是\emph{贪心残差}，同时包含搜索与有限模拟效应。只有在选择器优化同一候选族，且目标与评价分布一致等条件下，它才对应固定族下限。因此，本文不将该残差认定为不可消除下限，也不将 $G_{\rm cf}$ 称为最优可部署价值。'''
  b[9]=r'''\paragraph{有效的固定族不等式。}记 $d_S=\bar L(M;Y)-\bar L(S;Y)$、$\mu_S=\mathbb E d_S$，令 $b$ 最大化 $\mu_S$，$r$ 为另一集合。族下限为 $F=\mathbb E\max_S d_S-\max_S\mu_S$。逐点比较及中心化可得
\begin{equation}
\mathbb E(d_r-d_b)_+\ \leq F\ \leq\mathbb E\max_S(d_S-\mu_S).\label{eq:two}
\end{equation}
因为 $\max_S d_S-d_b\ge(d_r-d_b)_+$，且 $\max_S d_S\le\max_S\mu_S+\max_S(d_S-\mu_S)$。以同一经验分布替换期望与均值后，样本不等式严格成立，即使 $b,r$ 由该样本选择亦然；但这不提供总体 $F$ 的置信覆盖。若中心化收益联合高斯，则允许任意相关性，并有 $F\le\max_S\sigma_S\sqrt{2\log|\mathcal F|}$。若 $d_b-d_r$ 为均值 $\Delta$、标准差 $s$ 的正态变量，下界表达式为 $s\varphi(\Delta/s)-\Delta\Phi(-\Delta/s)$。正态性是额外建模假设；独立正态构造不是普适上界。'''
  b[10]=r'''\paragraph{估计与模型替代。}已知 $p$ 时，以公共随机数生成训练标签重复样本。在固定族中，最大化估计平均收益所得 $H_{\rm pre}$ 估计在期望上偏高；在一库选择、另一库评估，则估计有限样本选择器的期望价值，该值不超过 $H_{\rm pre}$。两者均不构成单次有限样本的确定性夹逼。真实数据用不相交辅助记录拟合的梯度提升模型提供 $\hat p$，记录的测试标签保持固定，因此评价以该测试集与概率模型为条件。模拟与记录结果的事后差距比较仅检验模型的一个泛函，不能完全识别下限。逐状态对带噪声的策略收益估计取最大值，会进一步引入选择乐观性；此类事后最大值属于敏感性汇总，不是可实现价值的认证下界。'''
  b[14]=r'''\paragraph{记录结果检查与敏感性。}置信区间包含零并不证明一致。返修采用显示单位下 $0.5$ 的等价界限。逻辑回归与梯度提升学习器的汇总检查支持等价，但 16 个单独数据集与学习器格子中仅两格证明等价，神经学习器汇总也未通过。在另 192 个贪心校准状态中，报告差距的汇总差异 90\% 区间为 $[-0.089,0.416]$，落在等价界内；留出差距区间 $[-0.138,0.513]$ 则越界。已知概率的扰动检查中，未扰动模型的平均下限偏差约 $-0.04$，但调整概率锐度使偏差改变约 $0.5$--$0.8$。不相交拟合很重要，却不能证明对概率模型误设稳健。'''
  b[15]=b[15].replace('分解不依赖候选族。','候选族敏感性。').replace('可部署占比','估计的可部署占比')
  b[16]=r'''\section{发现}
\label{sec:findings}
\paragraph{留出选择使贪心基准差距缩小。}表~\ref{tab:known} 保留原始交叉拟合贪心估计，各列实现第~\ref{sec:protocol} 节的记账恒等式；其中残差包含近似误差，不是固定族 $F$。半合成状态中，相对最好固定策略的报告领先为 $6.9$，交叉拟合贪心领先约为零。记录结果状态中的表观领先也明显缩小。这证明评价设计会影响差距，但不能证明被扣除的整个部分都不可消除。'''
  b[18]=r'''\paragraph{噪声排序与直接干预。}原跨数据集研究登记了噪声指标与贪心残差占比的关系（表~\ref{tab:gradient}）。其解释依赖事前基准：采用事后增强的比较器后，逻辑回归秩相关变为 $0.93$，神经学习器为 $0.52$（$p=0.15$），后者不再构成明确的跨数据集证据。另一项登记干预在 HMEQ、Bank Marketing 和 USPS 内，以 $0,0.1,0.2,0.3$ 的概率随机翻转标签，每个剂量、每个数据集有 16 个状态。三个数据集的经验固定族下限均单调增加（图~\ref{fig:floor}b）。这支持指定模拟中对随机标签污染的响应，不构成自然标签噪声下的普适单调规律。'''
  b[20]=r'''\paragraph{界限、搜索与规模检查。}修正后的经验上下界在全部 624 个返修状态中成立。其有效性来自代数，因此这项检查验证实现，不是新的总体置信保证。负相关、偏态与重尾压力测试否定了原独立正态捷径。12 候选池中穷举全部可行集合（平均约 730 个）得到与贪心近似接近的汇总诊断：平均残差差异约 $-0.02$，事前收益差异约 $0.01$，单位同表格。这仅检查相应规模的近似。候选池为 24、48、96 时，下限变化具有异质性，不能据此宣称大池近似保证。'''
  b[22]=r'''\paragraph{预算敏感性是经验结果。}增大预算可以扩大可行族，但 $F=H_{\rm post}-H_{\rm pre}$ 是两个变化量之差，不必增大。在登记预算实验中，目标数量由 3 变为 6、12 时，八个数据集的经验族下限均增大，配对平均增加 $0.74$，区间为 $[0.67,0.82]$，单位同表格（图~\ref{fig:budget}）。这是所测分布与学习器的结果，并非式~\eqref{eq:two} 的推论。'''
  b[32]=r'''\section{讨论与结论}
\label{sec:conclusion}
Oracle 差距依赖选择器可观察的信息及评价方式。固定族中，获知隐藏训练结果的价值与事前获取价值不同；贪心基准比较还混合评价集复用、搜索误差与有限模拟选择。修正不等式、噪声干预和近似检查支持具有明确假设的评价审核。报告应将留出 oracle 表现、固定族诊断与概率模型敏感性结合。DC-PRISM 在回放中的混合表现说明，较大的事后机会本身不能确立获取策略的有效性。'''
  b[33]=r'''\paragraph{局限。}概率模型替代不能识别真实条件结果分布，条件于测试标签的评价也弱于总体预测风险评价。交叉拟合贪心表现含搜索和蒙特卡洛误差，策略估计的事后最大值存在乐观偏差。精确枚举仅覆盖 12 候选池。随机标签污染是受控干预，不是自然不确定性的完整模型。候选池与学习器仍较小，未研究大型预训练模型及经核实的日历时间队列。基线是明确的适配版本。风险包络的概率解释还要求条件独立和有效上界风险分数，经验评分未确立这些条件。公开源码与本地结果覆盖不同，投稿前仍需准备并验证匿名材料访问。'''
 # Retain original table numbers; relabel their estimator, rather than substituting post-hoc maxima.
 for i in [17,19]:
  b[i]=b[i].replace('Aleatoric floor','Greedy residual').replace('aleatoric floor','greedy residual').replace('Floor','Residual').replace('Deployable','CF greedy').replace('deployable gap','cross-fitted greedy gain').replace('偶然性下限','贪心残差').replace('下限','残差').replace('可部署差距','交叉拟合贪心收益').replace('Deployable','CF greedy')
 b[25]=b[25].replace('Deployable gap $H_{\\mathrm{pre}}$ (cross-fitted)','Cross-fitted finite-family selector gain').replace('$H_{\\mathrm{pre}}$','CF gain').replace('可部署差距 $H_{\\mathrm{pre}}$（交叉拟合）','交叉拟合有限族选择器收益')
 # Replace obsolete figure with corrected diagnostics generated from v11.
 b[21]=r'''\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{figures/fig_predict.pdf}
\caption{'''+('Corrected diagnostics. (a) Empirical lower bound, floor and upper bound by group, covering 624 states; this verifies the empirical identity, not population coverage. (b) Within-cohort randomized label flips: empirical family floor at four doses, 16 states per dose and cohort. Lines connect descriptive means.' if lang=='en' else '修正诊断。(a) 各组经验下界、下限与上界，共覆盖 624 个状态；验证经验恒等关系，不提供总体覆盖。(b) 数据集内随机标签翻转：四种剂量下的经验族下限，每个剂量、每个数据集 16 个状态；连线连接描述性均值。')+r'''}
\label{fig:floor}
\end{figure}'''
 # Conceptual framework was quantitatively and theoretically stale; avoid embedding it.
 b[4]=''
 b[3]=b[3].replace('Fig.~\\ref{fig:framework}','Section~\\ref{sec:protocol}')
 # Correct the cardinality certificate claim without assuming an old upper bound was attained.
 b[6]=b[6].replace('and was off by one label otherwise','; exact follow-up search certified all remaining 88 reference counts').replace('其余状态差一个标签','其余 88 个状态经后续精确搜索也证明数量最优')
 # State independence required by the concentration interpretation.
 b[6]=b[6].replace('if $p_i\\le q_i$','under conditional independence and $p_i\\le q_i$')
 # Avoid identifying a finite-bank selector with the population optimum in narrative.
 for i in [24,26]:
  b[i]=b[i].replace('deployable gap','estimated ex-ante selector gain').replace('deployable','estimated as ex-ante value').replace('可部署差距','事前选择器估计收益').replace('可部署','估计为事前价值')
 b[2]=b[2].replace('It has not been measured','We investigate its magnitude').replace('for oracle bounds in label acquisition, where it decides','for oracle bounds in label acquisition, where it affects').replace('这一差距尚未被测量过','本文研究这一差距的大小')
 b[27]=b[27].replace('shows the ordering the protocol predicts','reports a separate dynamic comparison').replace('显示了协议预测的排序','报告独立的动态比较')
 (P/lang/'main.tex').write_text('\n\n'.join(b))

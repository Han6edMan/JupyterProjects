# 概率

## 基本概念

- **定义（概率）**<br>  
  对于样本空间 S 的每个可测事件 E 定义一个函数 P(E)，其满足
  1. $0 \le P(E) \le 1$
  2. P(S) = 1
  3. 对于事件序列 $\{E_i\}$ ，其满足 $E_iE_j = \empty\,,\,i \ne j$，有
  	$$P(\sum_{i=1}^{\infty}E_i) = \sum_{i=1}^{\infty} P(E_i)$$
  则称 P(E) 为事件 E 的概率



- **定理**<br>
  概率函数 P 是连续的，即对于递增或递减事件序列 $\{E_i\}$，有
	$$\lim_{n \rightarrow \infty} P(E_n) = P(\lim_{n \rightarrow \infty}E_n)$$



- **定义（随机变量）**<br>
  随机变量 X 是样本空间 S 到 $\R$ 上的函数：
	$$\begin{aligned}X: &S \rightarrow \R \\ &s \mapsto x \end{aligned}$$
	对任意 $x \in \R$，随机变量 X 的分布函数 F 定义为
	$$F(x) = P\{X \le x\} = P\{X \in (-\infty, x]\}$$

期望、方差、矩母函数、特征函数、拉普拉斯变换、条件期望……

## 随机过程
- **定义（随机过程）：**<br>
  随机变量族 $\underline{X} = \{X(t), t \in T\}$ 被称为随机过程，其中 T 为指标集，t 通常解释为时间。$\underline X$ 的实现被称为样本路径。<br>
  若 T 为可数集，则称 $\underline X$ 为离散时间的随机过程，随机事件 $X(t)$ 也通常记为 $X_t$；而若 T 为连续统，则称 $\underline X$ 为连续时间过程。<br>
  对于连续时间的随机过程，若对一切 $t_0<t_1< \cdots<t_n$，随机变量 $X(t_k) - X(t_{k-1})$ 都是独立的，则称该过程为独立增量过程。若 $X(t+s) - X(t)$ 对一切 t 有相同的分布，则称为平稳增量过程


## Markov 链
- **定义**<br>
  考虑一个取值于有限或可数个值的随机过程 $\{ X_n, n = 0, 1, 2, \cdots\}$，若对于当前状态 $X_n$，下一时刻任意状态 $X_{n+1}$ 独立于过去所有状态而只依赖于现在状态，即
  $$P(X_{n+1}|X_n, \cdots, X_1, X_0) = P(X_{n+1}|X_n)$$
  则称该随机过程具有 Markov 性，并称该随机过程为 Markov 链。<br>
  若以 $P_{ij}$ 表示当前所处状态为 i 且下一时刻状态为 j 的概率，并记 P 为单步转移概率 $P_{ij}$ 的矩阵，则
  $$P = \begin{bmatrix}P_{00} & P_{01} &\cdots\\ P_{10} & P_{11} & \cdots\\ \vdots&\vdots&\ddots \\\end{bmatrix}, \quad \sum_{j=1}^{\infty}P_{ij} = 1 $$

- **Chapman-Kolmogorov 方程及推论**<br>
  记 $P_{ij}^n$ 为当前状态为 i 且 n 步后状态为 j 的概率，$P^{(n)}$ 为相应的概率转移矩阵，则不难得出
  $$P_{ij}^{n+m} = \sum_{k=1}^{\infty} P_{ik}^n P_{kj}^m \qquad \mathrm{i.e.} \qquad P^{(m+n)} = P^{(n)} \cdot P^{(m)} $$
  进而 $P^{(n)} = P^{(n-1)} \cdot P =\cdots= P^n$

- **定义**<br>
  如果状态 j 可由状态 i 到达，即 $$P_{ij}^n >0$。可互相到达的两个状态称为连通的，记为 $i\leftrightarrow j$，互通的状态被称为在同一个类中。Markov 链称为不可约的，如果它只有一个类

- **定义**<br>对于任一状态 i，d 为集合 $\{n∣n ≥ 1,P^n_{ii}>0\}$ 的最大公约数，如果 d=1 ，则该状态为非周期的，否则称状态 i 具有周期 d。

下面不加证明的给出非周期连通 Markov 链收敛的结论和性质
- **定理**<br>
  若一个非周期 Markov 链有状态转移矩阵 P, 且它的任何两个状态是连通的，则$\lim_{n \rightarrow \infty} P_{ij}^n$与 i 无关，记为 $\pi (j)$，则
  1. <br>
  $$\lim_{n \rightarrow \infty} P^n = \begin{bmatrix}\pi(1) & \pi(2) &\cdots & \pi(j) &\cdots\\\pi(1) & \pi(2) &\cdots & \pi(j) &\cdots\\\cdots&\cdots&\cdots&\cdots&\cdots\\\pi(1) & \pi(2) &\cdots & \pi(j) &\cdots\\ \cdots&\cdots&\cdots&\cdots&\cdots\end{bmatrix}$$
  2. <br>
  $$\pi(j)=\sum_{i=0}^{\infty} \pi(i) P_{ij} $$
  3. 记 $\pi = [\pi(1),\pi(2),\cdots,\pi(j),\cdots]$，则 $\pi$ 是方程 $\pi = P \pi$ 的唯一非负解，且 $\sum_{i=0}^{\infty} \pi(i) =1$，$\pi$ 称为 Markov 链的平稳分布。

### 基于 Markov 链采样
给定某平稳分布对应的 Markov 链的转移矩阵 P，我们可以很容易地采样出服从该平稳分布的样本集：<br>
任意初始化一个概率分布 $\pi_0(x)$，再基于条件概率分布 $P(x|x_0)$ 采样状态值 $x_1$，并一直进行下去；我们假设状态转移进行到一定的次数（例如 $n_1$ 次）后，分布 $\pi_n(x)$ 已经接近收敛于平稳分布了，此时我们可以利用采样集 $(x_{n_1+1}, x_{n_1+2}, \cdots x_{n_2})$ 来进行蒙特卡罗模拟；总结以上过程如下
1. 声明状态转移矩阵 P、状态转移次数阈值 $n_1$、采样样本个数 $n_2$
2. 初始状态 $x_0$
3. 根据条件概率分布 $P(x|x_t)$ 采样得到样本 $x_{t+1}, (t = 1, 2, \cdots, n_1+n_2-1)$
4. $(x_{n1}, x_{n_1+1}, \cdots, x_{n_1+n_2−1})$ 即为平稳分布对应的样本集。

### MCMC 采样
对于平稳分布 $\pi$ 的转移矩阵未知的情况，注意到在上面 Markov 过程中，最后收敛得到的平稳分布满足 $\sum_{i=1}^{\infty}\pi(i) P(i, j)=\pi(j) = \sum_{i=1}^{\infty}\pi(j) P(j, i)$，进而如果我们能够满足
$$\pi(i) P(i, j)= \pi(j) P(j, i)$$
就可以利用 Markov 过程进行采样了。
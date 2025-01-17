# 3 多元线性回归
&emsp;&emsp;在2.3.1中，我们证得

\[\hat{\beta_1}=\beta_1+\frac{\sum_{i=1}^{n}(X_i-\bar{X})u_i}{\sum_{i=1}^{n}(X_i-\bar{X})^2}\]

因此有

\[\hat{\beta_1}\stackrel{d}{\rightarrow}\beta_1+\frac{\mathrm{cov}(X,u)}{\mathrm{var}(X)}=\beta_1+\rho_{Xu}\frac{\sigma_u}{\sigma_x}\]

如果模型具有内生性，即$\mathrm{cov}(X,u)\neq0$，$\hat{\beta_1}$不是$\beta_1$的无偏和一致估计，此时简单线性回归出现了遗漏变量偏差. 为了解决这个问题，需要采用多元线性回归.

## 3.1 回归系数的估计
**定义3.1.1** 定义

\[Y_i=X_i^\prime\beta+u_i\]

为**多元线性回归模型**.

&emsp;&emsp;注意到样本多元回归中，

\[X_i=\left(\begin{array}{c}
	1\\
	X_{i1}\\
	X_{i2}\\
	\vdots\\
	X_{ik}
\end{array}\right)\]

而总体多元回归中，

\[X_{n\times(k+1)}=\left(\begin{array}{cccc}
	1&X_{11}&\ldots&X_{1k}\\
	1&X_{21}&\ldots&X_{2k}\\
	1&X_{31}&\ldots&X_{3k}\\
	\vdots&\vdots&\,&\vdots\\
	1&X_{n1}&\ldots&X_{nk}
\end{array}\right)\]

其中，$k$表示解释变量的个数，$n$表示样本数.

&emsp;&emsp;类似地，我们也可得到多元线性回归的正规方程组. 目标函数为

\[\min\sum_{i=1}^{n}(Y_i-X_i^\prime\hat{\beta})^2\]

对$\hat{\beta}$进行求导，得

\[\nabla=-2\sum_{i=1}^{n}X_i^\prime\hat{u_i}=0\]

即

\[\sum_{i=1}^{n}X_i^\prime\hat{u_i}=0\]

将正规方程组写成标量形式，可得

\[\begin{cases}
	\sum_{i=1}^n\hat{u_i}=0\\
	\sum_{i=1}^nX_{i1}\hat{u_i}=0\\
	\sum_{i=1}^nX_{i2}\hat{u_i}=0\\
	\ldots\\
	\sum_{i=1}^nX_{ik}\hat{u_i}=0
\end{cases}\]

现对正规方程组进行求解：

\[X^\prime(Y-X\hat{\beta})=0\]

即

\[\hat{\beta}=(X^\prime X)^{-1}X^\prime Y\]

### 3.1.1 回归系数的期望和方差

&emsp;&emsp;现验证$\hat{\beta}$的无偏性和一致性.

\[\mathbb{E}[\hat{\beta}|X]=\mathbb{E}[(X^\prime X)^{-1}X^\prime Y|X]=(X^\prime X)^{-1}X^\prime\mathbb{E}[Y|X]=(X^\prime X)^{-1}X^\prime X\beta=\beta\]

&emsp;&emsp;由定理2.2.1可知，$\beta=(\mathbb{E}[X^\prime X])^{-1}\mathbb{E}[X^\prime Y]$，根据连续映射定理，可得

\[\hat{\beta}=(X^\prime X)^{-1}X^\prime Y\stackrel{p}{\rightarrow}(\mathbb{E}[X_i X_i^\prime])^{-1}\mathbb{E}[X_i Y]=\beta\]

&emsp;&emsp;接下来求$\hat{\beta}$的条件方差. 设误差方差矩阵为$D(d_{ii}=\sigma_i^2,d_{ij}=0,i\neq j)$，则

\[\mathrm{var}(\hat{\beta}|X)=\mathrm{var}((X^\prime X)^{-1}X^\prime Y|X)=(X^\prime X)^{-1}X^\prime DX(X^\prime X)^{-1}\]

在同方差条件下，由于$D=I_n\sigma^2$，则

\[\mathrm{var}(\hat{\beta})=\sigma^2(X^\prime X)^{-1}\]

### 3.1.2 回归系数的大样本抽样分布
&emsp;&emsp;注意到

\[\hat{\beta}=(X^\prime X)^{-1}X^\prime Y=(X^\prime X)^{-1}X^\prime (X\beta+u)=\beta+(X^\prime X)^{-1}X^\prime u=\beta+\left(\sum_{i=1}^{n}X_iX_i^\prime\right)^{-1}\left(\sum_{i=1}^{n}X_iu_i\right)\]

由于

\[\mathbb{E}\left[\sum_{i=1}^{n}X_iu_i|X_i\right]=0\]

\[\mathrm{var}(X_i u_i)=\mathbb{E}[X_i^\prime X_iu_i^2]\]

则

\[\sum_{i=1}^{n}X_iu_i\stackrel{d}{\rightarrow}N(0,n\mathbb{E}[X_i^\prime X_iu_i^2])\]

因为

\[\left(\sum_{i=1}^{n}X_iX_i^\prime\right)^{-1}\stackrel{p}{\rightarrow}\mathbb{E}[n(X_iX_i^\prime)]^{-1}=\frac{1}{n}\mathbb{E}[(X_iX_i^\prime)]^{-1}\]

则

\[(X^\prime X)^{-1}X^\prime u\stackrel{d}{\rightarrow}N\left( 0,\frac{1}{n}\mathbb{E}[(X_iX_i^\prime)]^{-1}\mathbb{E}[X_i^\prime X_iu_i^2]\mathbb{E}[(X_iX_i^\prime)]^{-1}\right) \]

即可得到$\hat{\beta}$的大样本抽样分布为

\[\hat{\beta}\stackrel{d}{\rightarrow}N\left( \beta,\frac{1}{n}\mathbb{E}[(X_iX_i^\prime)]^{-1}\mathbb{E}[X_i^\prime X_iu_i^2]\mathbb{E}[(X_iX_i^\prime)]^{-1}\right)\]

## 3.2 分块回归和偏回归
**定义3.2.1** 定义矩阵

\[P=X(X^\prime X)^{-1}X^\prime\]

为**投影矩阵**.

&emsp;&emsp;注意到$\hat{Y}=X\hat{\beta}=X(X^\prime X)^{-1}X^\prime Y=PY$，用$P$左乘任何向量可得该向量在超平面$X$上的投影.

**定理3.2.1** （投影矩阵的性质）(1)$P^\prime=P$；
	
(2)$P^2=P$；
	
(3)$PX=X$；
	
(4)$\mathrm{r}(P)=\mathrm{tr}(P)=k+1$.

证明：

\[P^\prime=X((X^\prime X)^\prime)^{-1}X^\prime=X^\prime(X^\prime X)^{-1}X=P\]

\[P^2=X(X^\prime X)^{-1}X^\prime X(X^\prime X)^{-1}X^\prime=X(X^\prime X)^{-1}X^\prime=P\]

\[PX=X(X^\prime X)^{-1}X^\prime X=X\]
	
由于$P$是实对称阵，可进行相似对角化，即

\[P=Q^\prime I_{k+1}Q\]

则有$\mathrm{r}(P)=k+1$. 由幂等矩阵的性质可知，$\mathrm{tr}(P)=\mathrm{r}(P)=k+1$.

**定义3.2.2** 定义矩阵

\[M=I_n-P\]

为**消灭矩阵**.

&emsp;&emsp;注意到$\hat{u}=Y-\hat{Y}=Y-PY=MY$，用$M$左乘任何向量可得该向量投影后的残差向量.

**定理3.2.2** （消灭矩阵的性质）(1)$M^\prime=M$；
	
(2)$M^2=M$；
	
(3)$MX=0$；
	
(4)$PM=0$；
	
(5)$\mathrm{r}(M)=\mathrm{tr}(M)=n-k-1$.

证明：

\[M^\prime=I_n^\prime-P^\prime=I_n-P\]

\[M^2=(I_n-P)(I_n-P)=I_n-2P+P^2=I_n-P=M\]

\[MX=X-PX=X-X=0\]

\[PM=P(I_n-P)=P-P^2=0\]

由$PM=0$，可得$\mathrm{r}(P)+\mathrm{r}(M)\leq n$；又由$P+M=I_n$，可得$n=\mathrm{r}(I_n)\leq \mathrm{r}(P)+\mathrm{r}(M)$，所以

\[\mathrm{r}(P)+\mathrm{r}(M)=n\]

则有$\mathrm{r}(M)=n-k-1$. 由幂等矩阵的性质可知，$\mathrm{tr}(M)=\mathrm{r}(M)=n-k-1$.

&emsp;&emsp;利用消灭矩阵的性质，我们可以将残差写成总体误差项的函数

\[\hat{u}=MY=M(X\beta+u)=Mu.\]

进而将残差平方和也写成总体误差项的函数

\[SSR=\hat{u}^\prime\hat{u}=u^\prime MMu=u^\prime Mu.\]

&emsp;&emsp;在多元线性回归中，只要增加一个变量就会对所有的回归系数产生影响，然而仅从$\hat{\beta}=(X^\prime X)^{-1}X^\prime Y$这一表达式中很难看出不同变量的影响.为此，我们将$X$进行分块，即$X=(X_1\quad X_2)$，分别对应于两组解释变量，此时多元线性回归模型可以改写为

\[Y=X_1\beta_1+X_2\beta_2+u\]

$\hat{\beta}=(X^\prime X)^{-1}X^\prime Y$可改写为

$$\begin{align*}
	\left(\begin{array}{c}
		\hat{\beta_1}\\
		\hat{\beta_2}
	\end{array}\right)&=\left(\left(\begin{array}{c}
		X_1^\prime\\
		X_2^\prime
	\end{array}\right)(X_1\quad X_2)\right)^{-1}\left(\begin{array}{c}
		X_1^\prime\\
		X_2^\prime
	\end{array}\right)Y\\
	&=\left(\begin{array}{cc}
		X_1^\prime X_1&	X_1^\prime X_2\\
		X_2^\prime X_1&	X_2^\prime X_2
	\end{array}\right)^{-1}\left(\begin{array}{c}
		X_1\\
		X_2
	\end{array}\right)Y\\
	&=\left(\begin{array}{c}
		(X_1^\prime M_2X_1)^{-1}X_1^\prime M_2\\
		(X_2^\prime M_1X_2)^{-1}X_2^\prime M_1
	\end{array}\right)Y
\end{align*}$$

令$r_1=M_1X_2$，$\tilde{u}_1=M_1Y$，则

\[\hat{\beta_2}=(\hat{r_1}^\prime\hat{r_1})^{-1}\hat{r_1}^\prime\tilde{u}_1\]

这里可以看出，$\hat{r_1}$是$X_2$对$X_1$回归得到的残差矩阵，$\tilde{u}_1$是$Y$对$X_1$回归得到的残差矩阵，而$\hat{\beta_2}$恰恰是$\tilde{u}_1$对$\hat{r_1}$进行回归所得的回归系数.这就是F**risch-Waugh-Lovell定理**的内容.

&emsp;&emsp;根据这一定理，我们就可对**偏回归系数**$\beta_j$进行估计.设$X_j$是解释变量$X_2$的样本组成的一维行向量. 首先由$X_j$对其他所有解释变量进行回归，得到残差列向量$\hat{r}_j$；再由$Y$对其他所有解释变量进行回归，得到残差列向量$\tilde{u}_j$，则

\[\hat{\beta_j}=\frac{\sum_{i=1}^{n}\hat{r}_{ij}\tilde{u}_{ij}}{\sum_{i=1}^{n}\hat{r}_{ij}^2}\]

由正规方程组易知

\[\sum_{i=1}^{n}\hat{r}_{ij}\tilde{u}_{ij}=\sum_{i=1}^{n}\hat{r}_{ij}(Y_i-X^*\tilde{\beta})\sum_{i=1}^{n}\hat{r}_{ij}Y_i\]

则有

\[\hat{\beta_j}=\frac{\sum_{i=1}^{n}\hat{r}_{ij}Y_i}{\sum_{i=1}^{n}\hat{r}_{ij}^2}\]

$\beta_j$是剔除了其他所有解释变量后的残差$\hat{r}_{j}$与$Y$的样本协方差和$\hat{r}_{j}$方差之比.

**注3.2.1** $X^*$为除$X_j$以外的其他解释变量组成的行向量，$\tilde{\beta}$为$Y$对除$X_j$以外的其他解释变量的回归系数列向量，显然$\hat{r}_{ij}$与$X^*$无关.

### 3.2.1 解释变量个数不同时偏回归系数之间的关系
&emsp;&emsp;现在，我们用上述结论探究解释变量个数不同时偏回归系数之间的关系.

&emsp;&emsp;先考察简单线性回归$Y=\gamma_0+\gamma_1X_1+e$和多元线性回归$Y=\beta_0+\beta_1X_1+\cdots+\beta_kX_k+u$.

$$\begin{align*}
	\hat{\gamma_1}&=\frac{\sum_{i=1}^n(X_{i1}-\bar{X}_1)Y_i}{\sum_{i=1}^n(X_{i1}-\bar{X}_1)^2}\\
	&=\frac{\sum_{i=1}^n(X_{i1}-\bar{X}_1)(\hat{\beta_0}+\hat{\beta_1}X_{i1}+\cdots+\hat{\beta_k}X_{ik}+\hat{u_i})}{\sum_{i=1}^n(X_{i1}-\bar{X}_1)^2}\\
	&=\hat{\beta_1}+\hat{\beta_2}\hat{\gamma_{2,1}}+\cdots+\hat{\beta_k}\hat{\gamma_{k,1}}
\end{align*}$$

$\hat{\gamma_{j,1}}$表示$X_j$对$X_1$的回归系数.

&emsp;&emsp;再将条件放宽，考察短回归$Y=\gamma_0+\gamma_1X_1+\cdots+\gamma_pX_p+e$和长回归$Y=\beta_0+\beta_1X_1+\cdots+\beta_pX_p+\beta_{p+1}X_{p+1}+\cdots+\beta_kX_k+u$.

$$\begin{align*}
	\hat{\gamma_1}&=\frac{\sum_{i=1}^n\hat{r_i}Y_i}{\sum_{i=1}^n\hat{r_i}^2}\\
	&=\frac{\sum_{i=1}^n\hat{r_i}(\hat{\beta_0}+\hat{\beta_1}X_{i1}+\cdots+\hat{\beta_k}X_{ik}+\hat{u_i})}{\sum_{i=1}^n\hat{r_i}^2}\\	&=\hat{\beta_1}+\hat{\beta}_{p+1}\frac{\sum_{i=1}^n\hat{r_i}X_{ip+1}}{\sum_{i=1}^n\hat{r_i}^2}+\cdots+\hat{\beta_k}\frac{\sum_{i=1}^n\hat{r_i}X_{ik}}{\sum_{i=1}^n\hat{r_i}^2}\\
	&=\hat{\beta_1}+\hat{\beta}_{p+1}\hat{\delta}_{p+1,1}+\cdots+\hat{\beta_k}\hat{\delta}_{k,1}
\end{align*}$$

$\hat{\delta}_{j,1}(p+1\leq j\leq k)$表示$X_j$对$X_1$的回归系数.

### 3.2.2 偏回归系数的期望和方差
&emsp;&emsp;现验证$\hat{\beta_j}$的无偏性.

\[\hat{\beta_j}=\frac{\sum_{i=1}^{n}\hat{r}_{ij}Y_i}{\sum_{i=1}^{n}\hat{r}_{ij}^2}=\beta_j+\frac{\sum_{i=1}^{n}\hat{r}_{ij}u_i}{\sum_{i=1}^{n}\hat{r}_{ij}^2}\]

对$\hat{\beta_j}-\beta_j$取条件期望，得

\[\mathbb{E}[\hat{\beta_j}-\beta_j|X_{1j},X_{2j},\cdots,X_{nj}]=\frac{\sum_{i=1}^{n}\hat{r}_{ij}\mathbb{E}[u_i|X_{1j},X_{2j},\cdots,X_{nj}]}{\sum_{i=1}^{n}\hat{r}_{ij}^2}=0\]

则

\[\mathbb{E}[\hat{\beta_j}]=\beta_j\]

即$\hat{\beta_j}$是$\beta_j$的无偏估计量.

&emsp;&emsp;在异方差条件下，

\[\mathrm{var}(\hat{\beta_j}|X_{1j},X_{2j},\cdots,X_{nj})=\frac{\sum_{i=1}^{n}\hat{r}_{ij}^2\mathbb{E}[u_i^2|X_{1j},X_{2j},\cdots,X_{nj}]}{\sum_{i=1}^{n}\hat{r}_{ij}^2}=\frac{\sum_{i=1}^n\hat{r}_{ij}^2\sigma_i^2}{(\sum_{i=1}^n\hat{r}_{ij}^2)^2}\]

&emsp;&emsp;在同方差条件下，

\[\mathrm{var}(\hat{\beta_j})=\frac{\sigma^2}{\sum_{i=1}^n\hat{r}_{ij}^2}=\frac{\sigma^2}{TSS_j(1-R_j^2)}\]

其中，$TSS_j=\sum_{i=1}^{n}(X_{ij}-\bar{X_j})^2$是$X_j$的总样本变异，$R_j^2$则是将$X_j$对所有其他自变量（包括截距项）进行回归得到的$R^2$.

### 3.2.3 偏回归系数的大样本抽样分布
&emsp;&emsp;易知

\[\sum_{i=1}^{n}\hat{r}_{ij}u_i\stackrel{d}\rightarrow N\left(0,n\mathrm{var}(\hat{r}_{ij}u_i)\right)\]

设$\frac{1}{n}\sum_{i=1}^n\hat{r}_{ij}^2\stackrel{p}\rightarrow a_j^2$，

则

\[\hat{\beta_j}\stackrel{d}\rightarrow N\left(\beta_j,\frac{\mathrm{var}(\hat{r}_{ij}u_i)}{n(a_j^2)^2}\right)\]

在同方差假设下，

\[\hat{\beta_j}\stackrel{d}\rightarrow N\left(\beta_j,\frac{\sigma^2}{na_j^2}\right)\]

## 3.3 拟合优度和误差方差的估计
### 3.3.1 拟合优度
&emsp;&emsp;在简单线性回归中，我们在一元条件下证明了定理2.4.1，现在我们将该定理推广到多元情况.

$$\begin{align*}
	TSS&=Y^\prime Y\\
	&=(\hat{Y}+\hat{u})^\prime(\hat{Y}+\hat{u})\\
	&=\hat{Y}^\prime\hat{Y}+2\hat{Y}^\prime\hat{u}+\hat{u}^\prime\hat{u}
\end{align*}$$

由于

\[\hat{Y}^\prime\hat{u}=Y^\prime PMY=0\]

则有

\[TSS=\hat{Y}^\prime\hat{Y}+\hat{u}^\prime\hat{u}=SSR+ESS.\]

### 3.3.2 误差方差的估计
&emsp;&emsp;考察残差平方和

\[SSR=u^\prime Mu=\mathrm{tr}(u^\prime Mu)=\mathrm{tr}(Muu^\prime)\]

则有

\[\mathbb{E}[SSR|X]=\mathrm{tr}(\mathbb{E}[Muu^\prime|X])=\mathrm{tr}(M\mathbb{E}[uu^\prime|X])=\mathrm{tr}(MD)\]

在同方差假设下，$D=I_n\sigma^2$，则

\[\mathbb{E}[SSR]=\sigma^2(n-k-1)\]

整理可得

\[\mathbb{E}\left[\frac{\hat{u}^\prime\hat{u}}{n-k-1}\right]=\sigma^2\]

则

\[\hat{\sigma}^2=\frac{1}{n-k-1}\sum_{i=1}^n\hat{u_i}^2\]

\[SER=\sqrt{\frac{1}{n-k-1}\sum_{i=1}^n\hat{u_i}^2}=\sqrt{\frac{SSR}{n-k-1}}\]

## 3.4 正态回归
**定义3.4.1** 当$u_i|X_i\sim N(0,\sigma^2)$时，定义

\[Y_i=X_i^\prime\beta+\mu_i\]

为**正态回归模型**.

&emsp;&emsp;该模型在零条件均值、独立同分布、有限峰度、无完全共线假设的基础上，加入了正态假设，即$\mathbb{E}[u|X]=0$的假设加强为$u|X\sim N(0,\sigma^2I_n)$. 由此，我们可以得到$\hat{\beta}$和$\hat{u}$的条件分布：

&emsp;&emsp;因为$\hat{\beta}=(X^\prime X)^{-1}X^\prime Y=\beta+(X^\prime X)^{-1}X^\prime u$，则有$\mathrm{var}(\hat{\beta}|X)=\sigma^2(X^\prime X)^{-1}X^\prime((X^\prime X)^{-1}X^\prime)^\prime=\sigma^2(X^\prime X)^{-1}$，那么

\[\hat{\beta}|X\sim N(\beta,\sigma^2(X^\prime X)^{-1})\]

&emsp;&emsp;类似地，由于$\hat{u}=Mu$，则

\[\hat{u}|X\sim N(0,\sigma^2M)\]

**定理3.4.1** 在正态回归模型中，$\hat{\beta}$和$\hat{u}$相互独立.

证明：考察$\hat{\beta}$和$\hat{u}$的联合正态分布.

\[\left(\begin{array}{c}\hat{\beta}-\beta\\\hat{u}\end{array}\right)=\left(\begin{array}{c}(X^\prime X)^{-1}X^\prime\\M\end{array}\right)u\]

\[\mathrm{var}\left(\left.\left(\begin{array}{c}
		\hat{\beta}-\beta\\
		\hat{u}
	\end{array}\right)\right|X\right)=\sigma^2\left(\begin{array}{c}
		(X^\prime X)^{-1}X^\prime\\
		M
	\end{array}\right)(X(X^\prime X)^{-1}\quad M)=\left(\begin{array}{cc}
		\sigma^2(X^\prime X)^{-1}&0\\
		0&\sigma^2M
	\end{array}\right) \]

由于$\mathrm{cov}(\hat{\beta}-\beta,\hat{u})=0$，则$\mathrm{cov}(\hat{\beta},\hat{u})=0$，在正态条件下$\hat{\beta}$和$\hat{u}$相互独立.

**定理3.4.2**

\[\frac{(n-k-1)\hat{\sigma}^2}{\sigma^2}\sim\chi^2(n-k-1)\]

证明：

$$\begin{align*}
		\frac{(n-k-1)\hat{\sigma}^2}{\sigma^2}&=\frac{\hat{u}^\prime\hat{u}}{\sigma^2}\\
		&=\left(\frac{u}{\sigma}\right)^\prime M\left(\frac{u}{\sigma}\right)
	\end{align*}$$

由定理3.2.2(5)可知$\mathrm{r}(M)=n-k-1$，则

\[\frac{(n-k-1)\hat{\sigma}^2}{\sigma^2}\sim\chi^2(n-k-1)\]

### 3.4.1 置信区间

#### $\beta_j$的置信区间
	
构造$t$统计量

\[t=\frac{\hat{\beta}_j-\beta_j}{\hat{\sigma}\sqrt{[(X^\prime X)^{-1}]_{jj}}}\sim t(n-k-1)\]

则$\beta_j$的置信水平为$\alpha$的置信区间为

\[\left(\hat{\beta}_j\pm t_{\frac{\alpha}{2}}(n-k-1)\hat{\sigma}\sqrt{[(X^\prime X)^{-1}]_{jj}}\right)\]
	
#### $\sigma^2$的置信区间
	
构造$\chi^2$统计量

\[\chi^2=\frac{\hat{u}^\prime\hat{u}}{\sigma^2}\sim\chi^2(n-k-1)\]

则$\sigma^2$的置信水平为$\alpha$的置信区间为

\[\left(\frac{\hat{u}^\prime\hat{u}}{\chi_{\frac{\alpha}{2}}^2(n-k-1)},\frac{\hat{u}^\prime\hat{u}}{\chi_{1-\frac{\alpha}{2}}^2(n-k-1)}\right) \]

###  3.4.2 假设检验
#### 偏回归系数$\beta_j$的$t$检验

\[H_0: \beta_j=b,\,H_1: \beta_j\neq b\]

构造$t$统计量

\[t=\frac{\hat{\beta_j}-b}{SE(\hat{\beta_j})}\]

拒绝域为

\[|t|\geq t_{\frac{\alpha}{2}}(n-k-1)\]
	
#### 对参数线性组合的$t$检验

\[H_0: \beta_1=\beta_2,\,H_1: \beta_1\neq\beta_2\]

构造$t$统计量

\[t=\frac{\hat{\beta_1}-\hat{\beta_2}}{SE(\hat{\beta_1}-\hat{\beta_2})}\]

其中

\[\hat{\beta_1}-\hat{\beta_2}=a^\prime\hat{\beta},\,a=\left( \begin{array}{c}
		0\\
		1\\
		-1\\
		O
	\end{array}\right),\,SE(\hat{\beta_1}-\hat{\beta_2})=a^\prime\hat{\sigma}^2(X^\prime X)^{-1}a\]

拒绝域为

\[|t|\geq t_{\frac{\alpha}{2}}(n-k-1)\]

#### 回归系数$\beta$的$F$检验

\[H_0: A\beta=b,\,H_1: A\beta\neq b\]

设$\mathrm{r}(A)=m$，构造$F$统计量

\[F=\frac{(A\hat{\beta}-b)^\prime(\hat{\sigma}^2A(X^\prime X)^{-1}A^\prime)^{-1}(A\hat{\beta}-b)}{m}\sim F(m,n-k-1)\]

拒绝域为

\[F\geq F_{\alpha}(m,n-k-1)\]
	
#### 对冗余变量的$F$检验

\[H_0: \beta_{k-q+1}=\cdots=\beta_k=0,\,H_1: H_0\,\mbox{为假}\]

无约束回归$u$：$Y=\beta_0+\beta_1X_1+\cdots+\beta_{k-q}X_{k-q}+\cdots+\beta_kX_k+u$
	
有约束回归$r$：$Y=\beta_0+\beta_1X_1+\cdots+\beta_{k-q}X_{k-q}+u$
	
基于残差平方和$SSR$的$F$检验：

\[F=\frac{(SSR_r-SSR_u)/q}{SSR_u/(n-k-1)}\sim F(q,n-k-1)\]
	
基于$R^2$的$F$检验：

\[F=\frac{(R_u^2-R_r^2)/q}{(1-R_u^2)/(n-k-1)}\sim F(q,n-k-1)\]
	
拒绝域为

\[F\geq F_{\alpha}(q,n-k-1)\]
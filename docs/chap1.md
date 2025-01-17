# 1 概率论与数理统计基础

## 1.1 参数估计量的性质

**定义1.1.1** 定义

$$\mathrm{bias}(\hat{\theta})=\mathbb{E}[\hat{\theta}]-\theta$$

为估计量的**偏差**.

&emsp;&emsp;当$\mathrm{bias}(\hat{\theta})=0$时，$\hat{\theta}$是$\theta$的**无偏估计**；

&emsp;&emsp;当${\lim_{n \to +\infty}\mathrm{bias}(\hat{\theta})}=0$时，$\hat{\theta}$是$\theta$的**渐进无偏估计**；

&emsp;&emsp;当${\lim_{n \to +\infty}P\{|\hat{\theta}-\theta|\geq\epsilon\}}=0$时，$\hat{\theta}$是$\theta$的**一致估计**.

**定义1.1.2** 定义

$$\mathrm{var}(\hat{\theta})=\mathbb{E}[(\hat{\theta}-\mathbb{E}[\theta])^2]$$

为估计量的**方差**.

&emsp;&emsp;对于$\hat{\theta}$的两个无偏估计量$\hat{\theta_1}$和$\hat{\theta_2}$，当$\mathrm{Var}(\hat{\theta_1})<\mathrm{Var}(\hat{\theta_2})$，则称$\hat{\theta_1}$相对于$\hat{\theta_2}$是**有效**的；若$\hat{\theta_0}$是所有$\hat{\theta}$的无偏估计量中方差最小的，则称$\hat{\theta_0}$是$\hat{\theta}$的**最优无偏估计量**.

**定义1.1.3** 定义

$$\mathrm{mse}(\hat{\theta})=\mathbb{E}[(\hat{\theta}-\theta)^2]$$

为估计量的**均方误**.

**定理1.1.1** （偏差-方差权衡）

$$\mathrm{mse}(\hat{\theta})=\mathrm{var}(\hat{\theta})+\mathrm{bias}^2(\hat{\theta})$$

证明：

$$\begin{align*}
\mathrm{var}(\hat{\theta})+\mathrm{bias}^2(\hat{\theta})&=\mathbb{E}[\hat{\theta}^2]-(\mathbb{E}[\theta])^2+(\mathbb{E}[\theta])^2-2\theta\mathbb{E}[\hat{\theta}]+\theta^2\\
&=\mathbb{E}[\hat{\theta}^2-2\theta\hat{\theta}+\theta^2]\\
&=\mathrm{mse}(\hat{\theta})
\end{align*}$$

## 1.2 多维随机变量的分布
### 1.2.1 多维随机向量及其数字特征
**定义1.2.1** 如果$X_1,X_2,...,X_n$都是随机变量，则称$\pmb{X}=(X_1,X_2,...X_n)^\mathrm{T}$为$n$维随机向量，简称**随机向量**.

**定义1.2.2** (1)设$\pmb{X}=(X_1,X_2,...X_n)^\mathrm{T}$是离散型随机向量，则它的分布函数为

$$F(\pmb{x})=F(x_1,x_2,...,x_n)=P\{X_1\leq x_1,X_2\leq x_2,...,X_n\leq x_n\}$$

式中，$\pmb{x}=(x_1,x_2,...,x_n)\in\mathbb{R}^n$，并记成$\pmb{X}\sim F$；

(2)设$\pmb{X}\sim F(\pmb{x})=F(x_1,x_2,...x_n)$，且$\pmb{X}$为连续型随机向量，若存在一个非负的函数$f(\cdot)$，使得

$$F(\pmb{x})=\int_{-\infty}^{x_1}\int_{-\infty}^{x_2}...\int_{-\infty}^{x_n}f(t_1,t_2,...,t_n)\mathrm{d}t_1\mathrm{d}t_2...\mathrm{d}t_n$$

对一切$\pmb{x}\in\mathbb{R}^n$成立，则称$\symbfit{X}$有分布密度函数$f(\cdot).$

**定义1.2.3** 设$\pmb{X}=(X_1,X_2,...X_n)^\mathrm{T}$有$n$个分量，若为$\mathbb{E}[X_i]=\mu_i(i=1,2,...,n)$存在，定义随机向量$\pmb{X}$的均值为

$$\mathbb{E}[\symbfit{X}]=\left(
	\begin{array}{cccc}
		\mathbb{E}[X_1]\\
		\mathbb{E}[X_2]\\
		\vdots\\
		\mathbb{E}[X_n]
	\end{array}\right)
	=\left(
	\begin{array}{cccc}
		\mu_1\\
		\mu_2\\
		\vdots\\
		\mu_n
	\end{array}\right)=\symbfit{\mu}$$

$\pmb{\mu}$是一个$n$维向量，称为均值向量.

**推论1.2.1** 当$\pmb{A}$，$\pmb{B}$为常数矩阵时，由定义1.2.3可推出如下性质：

(1)$\mathbb{E}[\pmb{AX}]=\pmb{A}\mathbb{E}[\pmb{X}]$

(2)$\mathbb{E}[\pmb{AXB}]=\pmb{A}\mathbb{E}[\pmb{X}]\pmb{B}$

证明：设

$$\pmb{A}=\left(
	\begin{array}{cccc}
		a_{11} & a_{12} & \ldots & a_{1n}\\
		a_{21} & a_{22} & \ldots & a_{2n}\\
		\vdots & \vdots & \, & \vdots\\
		a_{m1} & a_{m2} & \ldots & a_{mn} 
	\end{array}\right),\pmb{B}=(b_1,b_2,...,b_s),$$

则

$$\mathbb{E}[\symbfit{AX}]=\left(
	\begin{array}{cccc}
		\sum_{i=1}^{n}\mathbb{E}[a_{1i}X_i]\\
		\sum_{i=1}^{n}\mathbb{E}[a_{2i}X_i]\\
		\vdots\\
		\sum_{i=1}^{n}\mathbb{E}[a_{mi}X_i]
	\end{array}\right)=\left(
	\begin{array}{cccc}
		\sum_{i=1}^{n}a_{1i}\mathbb{E}[X_i]\\
		\sum_{i=1}^{n}a_{2i}\mathbb{E}[X_i]\\
		\vdots\\
		\sum_{i=1}^{n}a_{mi}\mathbb{E}[X_i]
	\end{array}\right)=\symbfit{A}\mathbb{E}[\symbfit{X}]$$

$$\begin{align*}
\mathbb{E}[\symbfit{AXB}]&=\left(\begin{array}{cccc}
\sum_{i=1}^{n}\mathbb{E}[a_{1i}X_ib_1] & \sum_{i=1}^{n}\mathbb{E}[a_{1i}X_ib_2] & \ldots & \sum_{i=1}^{n}\mathbb{E}[a_{1i}X_ib_s]\\
\sum_{i=1}^{n}\mathbb{E}[a_{2i}X_ib_1] & \sum_{i=1}^{n}\mathbb{E}[a_{2i}X_ib_2] & \ldots & \sum_{i=1}^{n}\mathbb{E}[a_{2i}X_ib_s]\\
\vdots & \vdots & \, & \vdots\\
\sum_{i=1}^{n}\mathbb{E}[a_{mi}X_ib_1] & \sum_{i=1}^{n}\mathbb{E}[a_{mi}X_ib_2] & \ldots & \sum_{i=1}^{n}\mathbb{E}[a_{mi}X_ib_s]
		\end{array}\right)\\
		&=\left(\begin{array}{cccc}
			b_1\sum_{i=1}^{n}a_{1i}\mathbb{E}[X_i] & b_2\sum_{i=1}^{n}a_{1i}\mathbb{E}[X_i] & \ldots & b_s\sum_{i=1}^{n}a_{1i}\mathbb{E}[X_i]\\
			b_1\sum_{i=1}^{n}a_{2i}\mathbb{E}[X_i] & b_2\sum_{i=1}^{n}a_{2i}\mathbb{E}[X_i] & \ldots & b_s\sum_{i=1}^{n}a_{2i}\mathbb{E}[X_i]\\
			\vdots & \vdots & \, & \vdots\\
			b_1\sum_{i=1}^{n}a_{mi}\mathbb{E}[X_i] & b_2\sum_{i=1}^{n}a_{mi}\mathbb{E}[X_i] & \ldots & b_s\sum_{i=1}^{n}a_{mi}\mathbb{E}[X_i]
		\end{array}\right)\\
		&=\symbfit{A}\mathbb{E}[\symbfit{X}]\symbfit{B}
	\end{align*}$$

**定义1.2.4** (1)对于随机向量$\symbfit{X}=(X_1,X_2,...X_n)^\mathrm{T}$，

$$\begin{align*}
		\symbfit{\Sigma}&=D(\symbfit{X})=\mathbb{E}[(\symbfit{X}-\mathbb{E}[\symbfit{X}])(\symbfit{X}-\mathbb{E}[\symbfit{X}])^\mathrm{T}]\\
		&=\left(\begin{array}{cccc}
			D(X_1) & \mathrm{cov}(X_1,X_2) & \ldots & \mathrm{cov}(X_1,X_n)\\
			\mathrm{cov}(X_2,X_1) & D(X_2) & \ldots & \mathrm{cov}(X_2,X_n)\\
			\vdots & \vdots & \, & \vdots\\
			\mathrm{cov}(X_n,X_1) & \mathrm{cov}(X_n,X_2) & \ldots & D(X_n)
		\end{array}\right)\\
		&=(\sigma_{ij})_{n\times n}
\end{align*}$$

为$n$维随机向量$\symbfit{X}$的**协方差矩阵**，$|\symbfit{\Sigma}|$为$\symbfit{X}$的**广义方差**.
	
(2)对于随机向量$\symbfit{X}=(X_1,X_2,...X_m)^\mathrm{T}$和$\symbfit{Y}=(Y_1,Y_2,...Y_n)^\mathrm{T}$，它们之间的协方差矩阵为

$$\symbfit{\Sigma}_{m\times n}=\mathbb{E}[(\symbfit{X}-\mathbb{E}[\symbfit{X}])(\symbfit{Y}-\mathbb{E}[\symbfit{Y}])^\mathrm{T}]=\mathrm{cov}(\symbfit{X},\symbfit{Y})=(\mathrm{cov}(X_i,Y_j)), i=1,2,...,m; j=1,2,...,n.$$

若$\mathrm{cov}(\symbfit{X},\symbfit{Y})=\textbf{0}$，则称$\symbfit{X}$和$\symbfit{Y}$不相关.

**推论1.2.2** 当$\symbfit{A}$，$\symbfit{B}$为常数矩阵时，由定义1.2.4可推出如下性质：

(1)$D(\symbfit{AX})=\symbfit{A}D(\symbfit{X})\symbfit{A}^\mathrm{T}=\symbfit{A}\symbfit{\Sigma}\symbfit{A}^\mathrm{T}$
	
(2)$\mathrm{cov}(\symbfit{AX},\symbfit{BY})=\symbfit{A}\mathrm{cov}(\symbfit{X},\symbfit{Y})\symbfit{B}^\mathrm{T}$

证明：

$$D(\symbfit{AX})=\mathbb{E}[(\symbfit{AX}-\mathbb{E}[\symbfit{AX}])(\symbfit{AX}-\mathbb{E}[\symbfit{AX}])^\mathrm{T}]=\mathbb{E}[\symbfit{A}(\symbfit{X}-\mathbb{E}[\symbfit{X}])(\symbfit{X}-\mathbb{E}[\symbfit{X}])^\mathrm{T}\symbfit{A}^\mathrm{T}]=\symbfit{A}D(\symbfit{X})\symbfit{A}^\mathrm{T}$$

$$\begin{align*}\mathrm{cov}(\symbfit{AX},\symbfit{BY})&=\mathbb{E}[(\symbfit{AX}-\mathbb{E}[\symbfit{AX}])(\symbfit{BY}-\mathbb{E}[\symbfit{BY}])^\mathrm{T}]=\mathbb{E}[\symbfit{A}(\symbfit{X}-\mathbb{E}[\symbfit{X}])(\symbfit{Y}-\mathbb{E}[\symbfit{Y}])^\mathrm{T}\symbfit{B}^\mathrm{T}]\\
		&=\symbfit{A}\mathrm{cov}(\symbfit{X},\symbfit{Y})\symbfit{B}^\mathrm{T}
	\end{align*}$$

**定义1.2.5** 若随机向量$\symbfit{X}=(X_1,X_2,...X_n)^\mathrm{T}$的协方差矩阵存在，且每个分量的方差大于零，则

$$\symbfit{R}=(\mathrm{corr}(X_i,Y_j))=(r_{ij})_{n\times n}$$

为$\symbfit{X}$的**相关矩阵**，其中

$$r_{ij}=\frac{\mathrm{cov}(X_i,Y_j)}{\sqrt{\vphantom{D(X_j)}D(X_i)}\sqrt{\vphantom{D(X_i)}D(X_j)}}, i,j=1,2,...,n$$

### 1.2.2 多元正态分布
**定义1.2.6** 	若随机向量$\symbfit{X}=(X_1,X_2,...X_p)^\mathrm{T}$的概率密度函数为

$$f(x_1,x_2,...x_n)=\frac{1}{(2\pi)^{\frac{p}{2}}|\symbfit{\Sigma}|^{\frac{1}{2}}}\mathrm{exp}\left\{-\frac{1}{2}(\symbfit{x-\mu})^\mathrm{T}\symbfit{\Sigma}^{-1}(\symbfit{x-\mu})\right\}, \symbfit{\Sigma}>\textbf{0}$$

则称$\symbfit{X}=(X_1,X_2,...X_p)^\mathrm{T}$服从$p$元正态分布，也称$\symbfit{X}$为$p$元正态变量，记为：

$$\symbfit{X}\sim N_p(\symbfit{\mu},\symbfit{\Sigma})$$

&emsp;&emsp;当$p=2$时，可以得到二元正态分布的概率密度函数.

&emsp;&emsp;设$\symbfit{X}=(X_1,X_2)^\mathrm{T}$服从二元正态分布，则

$$\symbfit{\Sigma}=\left(\begin{array}{cccc}
	\sigma_1^2 & \sigma_1\sigma_2r\\
	\sigma_2\sigma_1r & \sigma_2^2
\end{array}\right)$$

则

$$|\symbfit{\Sigma}|=\sigma_1^2\sigma_2^2(1-r^2)$$

$$\symbfit{\Sigma}^{-1}=\frac{1}{\sigma_1^2\sigma_2^2(1-r^2)}\left(\begin{array}{cccc}
	\sigma_2^2 & -\sigma_1\sigma_2r\\
	-\sigma_2\sigma_1r & \sigma_1^2
\end{array}\right)$$

则二元正态变量$\symbfit{X}$的概率密度函数为

$$f(x_1,x_2)=\frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-r^2}}\mathrm{exp}\left\{-\frac{1}{2(1-r^2)}\left[\frac{(x_1-\mu_1)^2}{\sigma_1^2}-2r\frac{(x_1-\mu_1)(x_2-\mu_2)}{\sigma_1\sigma_2}+\frac{(x_2-\mu_2)^2}{\sigma_2^2}\right]\right\}$$

当$r=0$时，

$$\begin{align*}
	f(x_1,x_2)&=\frac{1}{2\pi\sigma_1\sigma_2}\mathrm{exp}\left\{-\frac{1}{2}\left[\frac{(x_1-\mu_1)^2}{\sigma_1^2}+\frac{(x_2-\mu_2)^2}{\sigma_2^2}\right]\right\}\\
	&=\frac{1}{\sqrt{2\pi}\sigma_1}\mathrm{exp}\left[-\frac{(x_1-\mu_1)^2}{2\sigma_1^2}\right]\frac{1}{\sqrt{2\pi}\sigma_2}\mathrm{exp}\left[-\frac{(x_2-\mu_2)^2}{2\sigma_2^2}\right]\\
	&=f(x_1)f(x_2)
\end{align*}$$

此时$X_1$，$X_2$相互独立.

**推论1.2.3** （多元正态分布的性质）设$\symbfit{X}\sim N_p(\symbfit{\mu},\symbfit{\Sigma})$，
	
(1)$\mathbb{E}[\symbfit{X}]=\symbfit{\mu},D(\symbfit{X})=\symbfit{\Sigma}$；
	
(2)设$m$维随机向量$\symbfit{Z}_{m\times1}=\symbfit{AX}+\symbfit{b}$，其中$\symbfit{A}_{m\times p}$为常数矩阵，$\symbfit{b}$为$m$维常向量，则$\symbfit{Z}\sim N_m(\symbfit{A\mu}+\symbfit{b},\symbfit{A\Sigma A}^\mathrm{T})$.

### 1.2.3 $\chi^2$分布
**定义1.2.7** 当$\symbfit{X}\sim N_n(\textbf{0},\symbfit{I}_n)$时，$\symbfit{X}^\mathrm{T}\symbfit{X}\sim \chi^2(n)$.

**推论1.2.4** 若$\symbfit{A}_{n\times n}$为对称幂等矩阵，$r(\symbfit{A})=r$，当$\symbfit{X}\sim N_n(\textbf{0},\symbfit{I}_n)$时，$\symbfit{X}^\mathrm{T}\symbfit{AX}\sim \chi^2(r)$.

证明：因为$\symbfit{A}$为对称幂等矩阵，则一定存在可逆矩阵$\symbfit{P}_{n\times n}$，使得

$$\symbfit{A}=\symbfit{P}^\mathrm{T}\left(\begin{array}{cccc}
		\symbfit{I}_r & \symbfit{O}\\
		\symbfit{O} & \symbfit{O}
	\end{array}\right)\symbfit{P}$$

令$\symbfit{Y}=\symbfit{PX}$，则

$$\symbfit{X}^\mathrm{T}\symbfit{AX}=\symbfit{X}^\mathrm{T}\symbfit{P}^\mathrm{T}\left(\begin{array}{cccc}
		\symbfit{I}_r & \symbfit{O}\\
		\symbfit{O} & \symbfit{O}
	\end{array}\right)\symbfit{PX}=\symbfit{Y}^\mathrm{T}\left(\begin{array}{cccc}
		\symbfit{I}_r & \symbfit{O}\\
		\symbfit{O} & \symbfit{O}
	\end{array}\right)\symbfit{Y}\sim \chi^2(r).$$

**推论1.2.5** 当$\symbfit{X}\sim N_n(\symbfit{\mu},\symbfit{\Sigma})$时，$(\symbfit{X-\mu})^\mathrm{T}\symbfit{\Sigma}^{-1}(\symbfit{X-\mu})\sim \chi^2(n)$.
## 1.3 条件期望
### 1.3.1 条件期望和条件期望误差
**定义1.3.1** (1)设$X$和$Y$的联合分布为离散分布，对于$P\{Y=y\}>0$的$y$值，$X$在给定$Y=y$之下的条件期望为
	
$$\mathbb{E}[X|Y=y]=\sum_{x}xP\{X=x|Y=y\}=\sum_{x}x\frac{P\{X=x,Y=y\}}{P\{Y=y\}}$$

(2)设$X$和$Y$有连续型联合分布，当$f_Y(y)>0$时，$X$在给定$Y=y$之下的条件期望为

$$\mathbb{E}[X|Y=y]=\int_{-\infty}^{+\infty}xf_{X|Y}(x|y)\mathrm{d}x=\int_{-\infty}^{+\infty}x\frac{f_{X,Y}(x,y)}{f_Y(y)}\mathrm{d}x$$

&emsp;&emsp;给定一组随机变量$Y_i(i=1,2,...,n)$，当$Y_i$取遍所有的可能值后，得到的$X_i$的期望值便可以看成$X_i$关于$Y_i$的函数，此时称$\mathbb{E}[X_i|Y_i]$为$X_i$关于$Y_i$的**条件期望函数**(conditional expectation function，简称CEF).

&emsp;&emsp;类似地，可以定义条件方差：

$$\mathrm{var}(X|Y=y)=\mathbb{E}[(X-\mathbb{E}[X|Y])^2|Y=y].$$

**推论1.3.1** 

$$\mathbb{E}[g(X,Y)|Y=y]=
	\begin{cases}
		\sum_{x}g(x,y)f_{X|Y}(x|y), X\mbox{是离散型变量}\\
		\int_{-\infty}^{+\infty}g(x,y)f_{X|Y}(x|y)\mathrm{d}x, X\mbox{是连续型变量}
	\end{cases}$$

**推论1.3.2**

$$\mathbb{E}[g(X)h(Y)|Y]=h(Y)\mathbb{E}[g(X)|Y]$$

证明：	考虑连续情形：

$$\begin{align*}
		\mathbb{E}[g(X)h(Y)|Y=y]&=\int_{-\infty}^{+\infty}g(x)h(y)f_{X|Y}(x|y)\mathrm{d}x\\
		&=h(y)\int_{-\infty}^{+\infty}g(x)f_{X|Y}(x|y)\mathrm{d}x\\
		&=h(y)\mathbb{E}[g(X)|Y=y].
	\end{align*}$$

&emsp;&emsp;因为$\mathbb{E}[g(X)h(Y)|Y]$的每一个实现值$(\mathbb{E}[g(X)h(Y)|Y=y])$与$h(Y)\mathbb{E}[g(X)|Y]$的每一个实现值$(h(y)\mathbb{E}[g(X)|Y=y])$总是相等的，所以有

$$\mathbb{E}[g(X)h(Y)|Y]=h(Y)	\mathbb{E}[g(X)|Y]$$

**定理1.3.1** （迭代期望法则）

$$\mathbb{E}[\mathbb{E}[X|Y]]=\mathbb{E}[X]$$

证明：当$X$和$Y$均为离散型变量时，

$$\begin{align*}
		\mathbb{E}[\mathbb{E}[X|Y]]&=\sum_{y}\sum_{x}x\frac{P\{X=x,Y=y\}}{P\{Y=y\}}P\{Y=y\}
		\\	&=\sum_{y}\sum_{x}xP\{X=x,Y=y\}\\
		&=\sum_{x}x\sum_{y}P\{X=x,Y=y\}\\
		&=\sum_{x}xP\{X=x\}\\
		&=\mathbb{E}[X]
	\end{align*}$$
	
当$X$和$Y$均为连续型变量时，

$$\begin{align*}
		\mathbb{E}[\mathbb{E}[X|Y]]&=\int_{-\infty}^{+\infty}\left(\int_{-\infty}^{+\infty}x\frac{f_{X,Y}(x,y)}{f_Y(y)}\mathrm{d}x\right)f_Y(y)\mathrm{d}y
		\\	&=\int_{-\infty}^{+\infty}x\left(\int_{-\infty}^{+\infty}f_{X,Y}(x,y)\mathrm{d}y\right)\mathrm{d}x\\
		&=\int_{-\infty}^{+\infty}xf_X(x)\mathrm{d}x\\
		&=\mathbb{E}[X]
	\end{align*}$$

**定理1.3.2** （全方差法则）

$$\mathrm{var}(X)=\mathrm{var}(\mathbb{E}[X|Y])+\mathbb{E}[\mathrm{var}(X|Y)]$$

证明：注意到

$$\mathrm{var}(\mathbb{E}[X|Y])=\mathbb{E}[(\mathbb{E}[X|Y])^2]-(\mathbb{E}[\mathbb{E}[X|Y]])^2=\mathbb{E}[(\mathbb{E}[X|Y])^2]-(\mathbb{E}[X])^2$$

$$\mathbb{E}[\mathrm{var}(X|Y)]=\mathbb{E}[\mathbb{E}[X^2|Y]-(\mathbb{E}[X|Y])^2]=\mathbb{E}[X^2]-\mathbb{E}[(\mathbb{E}[X|Y])^2]$$

因此有

$$\mathrm{var}(\mathbb{E}[X|Y])+\mathbb{E}[\mathrm{var}(X|Y)]=-(\mathbb{E}[X])^2+\mathbb{E}[X^2]=\mathrm{var}(X)$$

**定义1.3.2** 	定义

$$e=X-\mathbb{E}[X|Y]$$

为**条件期望误差**(CEF error).

这里，$e$是由$(X,Y)$的联合分布决定的随机变量.同时，上式可以看作$X$的条件期望的分解，即

$$X=\mathbb{E}[X|Y]+e.$$

**推论1.3.3** (1)$\mathbb{E}[e|Y]=\mathbb{E}[e]=0$
	
(2)$\mathrm{cov}(e,g(Y))=\mathbb{E}[e g(Y)]=\mathrm{cov}(e,\mathbb{E}[X|Y])=0$
	
(3)$\mathrm{var}(e)=\mathbb{E}[\mathrm{var}(X|Y)]$

证明：(1)$\mathbb{E}[e|Y]=\mathbb{E}[X-\mathbb{E}[X|Y]|Y]=\mathbb{E}[X|Y]-\mathbb{E}[X|Y]=0.$ $\mathbb{E}[e]=\mathbb{E}[\mathbb{E}[e|Y]]=0.$
	
(2)$\mathrm{cov}(e,g(Y))=\mathbb{E}[e g(Y)]=\mathbb{E}[\mathbb{E}[e g(Y)]|Y]=\mathbb{E}[g(Y)\mathbb{E}[e]|Y]=0.$

因为$\mathbb{E}[X|Y]$是关于$Y$的一个函数，满足$g(Y)$的形式，则有$\mathrm{cov}(e,\mathbb{E}[X|Y])=0.$
	
(3)$\mathrm{var}(e)=\mathbb{E}[e^2]=\mathbb{E}[\mathbb{E}[e^2|Y]]=\mathbb{E}[\mathbb{E}[(X-\mathbb{E}[X|Y])^2|Y]]=\mathbb{E}[\mathrm{var}(X|Y)].$

**定理1.3.3** （条件期望函数的预测性质）对于给定的$X$，当$\mathbb{E}[Y^2]$存在时，$\mathbb{E}[Y|X]$是对$Y$的最小均方误预测.

证明：令$g(X)$为任意一个对$Y$的预测函数,则

$$\begin{align*}
		\mathrm{mse}(g(X))&=\mathbb{E}[(g(X)-Y)^2]\\
		&=\mathbb{E}[[g(X)-\mathbb{E}[Y|X]-e]^2]\\
		&=\mathbb{E}[[g(X)-\mathbb{E}[Y|X]]^2]+2\mathbb{E}[e[g(X)-\mathbb{E}[Y|X]]]+\mathbb{E}[e^2]\\
		&=\mathbb{E}[[g(X)-\mathbb{E}[Y|X]]^2]+\mathbb{E}[e^2]\\
		&\geq\mathbb{E}[e^2]
	\end{align*}$$
	
当且仅当$g(X)=\mathbb{E}[Y|X]$时，“$=$”成立. 因此有

$$\mathbb{E}[Y|X]=\arg\min\mathbb{E}[(g(X)-Y)^2]=\arg\min\mathrm{mse}(g(X)).$$

### 1.3.2 条件期望函数
**定义1.3.3** 当条件期望函数$m(x)=\mathbb{E}[Y|X=x]$满足

$$m(x)=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_kx_k=x^\mathrm{T}\beta,\mbox{其中}x=\left(\begin{array}{c}
		1\\
		x_1\\
		x_2\\
		\vdots\\
		x_k
	\end{array}\right),\beta=\left(\begin{array}{c}
		\beta_0\\
		\beta_1\\
		\beta_2\\
		\vdots\\
		\beta_k
	\end{array}\right)$$

时，则称$m(x)$为线性条件期望函数（Linear CEF）.

&emsp;&emsp;如果关于$X$的条件期望函数是线性的，对$Y$进行条件期望分解，可得

$$Y=X^\mathrm{T}\beta+e.$$

&emsp;&emsp;现在，我们介绍Linear CEF的一个特例.

&emsp;&emsp;若$X$为分类数据，在进行预测时，我们需要对$X$进行赋值，使$X$成为**虚拟变量**（Dummy Variables）.这里，$X$是离散型随机变量，如果对于$X$所取的所有可能值，$\beta$中均有相应的参数与其对应，我们称模型达到**饱和**.此时，可构造线性饱和虚拟变量模型，即

$$m(\tilde{X})=X^\mathrm{T}\beta+e$$

其中，设$\tilde{X}_i(i=1,2,\cdots,k)$有$n_i$个取值，则虚拟随机变量有$N=\prod_{i=1}^kn_i$个，$X$是一个$N$维向量.这里，$Y=X^\mathrm{T}\beta+e$称为**饱和虚拟变量回归模型**.

&emsp;&emsp;注意，无论$Y$的分布如何，上述模型都能完美地拟合Linear CEF，证明极其复杂，这里不另加赘述.

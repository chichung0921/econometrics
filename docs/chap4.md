# 4 工具变量回归
## 4.1 内生性
&emsp;&emsp;在线性回归模型中，我们有很重要的外生性假设，即满足$\mathbb{E}[Xu]=0$，然而现实中常常会发生\textbf{内生性}问题，即$\mathbb{E}[Xu]\neq 0$，这时线性回归模型便失效了. 这便是我们在第三章引言部分介绍的遗漏变量偏误. 在这种情况下，为了将$Y=X^\prime\beta+u$与线性回归模型的情况进行区分，我们将该式称为**结构方程**.

&emsp;&emsp;以下是内生性出现的一些实例：

### 4.1.1 度量误差
&emsp;&emsp;假设随机变量$Y,Z$满足$\mathbb{E}[Y|Z]=Z^\prime\beta$，然而$Z$无法观测，于是我们用$X=Z+e$代替$Z$进行回归，其中$e$是测量误差，且满足$\mathbb{E}[Ze]=0$，则有

\[Y=Z^\prime\beta+u=(X-e)^\prime\beta+u=X^\prime\beta+v\]

则$v=u-e^\prime\beta$. 此时$\mathbb{E}[Xv]=\mathbb{E}[(Z+e)(u-e^\prime\beta)]=-\mathbb{E}[ee^\prime]\beta$. 用$Y$对$X$进行偏回归，得到

\[\beta^*=\beta+(\mathbb{E}[XX^\prime])^{-1}\mathbb{E}[Xv]=(1-\mathbb{E}[XX^\prime]^{-1}\mathbb{E}[ee^\prime])\beta=(1-\mathbb{E}[ZZ^\prime+ee^\prime]^{-1}\mathbb{E}[ee^\prime])\beta\]

当测量误差较大，即$\mathbb{E}[ee^\prime]$较大时，$\beta^*\rightarrow0$.

### 4.1.2 供求曲线
	
&emsp;&emsp;设需求曲线方程为$Q=-\beta_1P+u_1$，供给曲线方程为$Q=\beta_2Q+u_2$，写成矩阵形式为

\[\left(\begin{array}{cc}
		1&\beta_1\\
		1&-\beta_2
	\end{array}\right)\left(\begin{array}{c}
		Q\\
		P
	\end{array}\right)=\left(\begin{array}{c}
		u_1\\
		u_2
	\end{array}\right) \]
	
解得

\[\left(\begin{array}{c}
		Q\\
		P
	\end{array}\right)=\frac{1}{\beta_1+\beta_2}\left(\begin{array}{c}
		\beta_2u_1+\beta_1u_2\\
		u_1-u_2
	\end{array}\right)\]

设$Q$对$P$的回归方程为$Q=\beta^*P+u^*$则

\[\beta^*=\frac{\mathbb{E}[PQ]}{\mathbb{E}[P^2]}=\frac{\beta_2-\beta_1}{2}\]

可以发现，$\beta^*$既不等于$\beta_1$，也不等于$\beta_2$.

## 4.2 工具变量估计
**定义4.2.1** 对于结构方程$Y=X^\prime\beta+u$，其中$X$中含$k$个解释变量，若能找到一个变量$Z_{l\times1}$，满足
	
(1)**相关性**：$r(\mathbb{E}[ZX^\prime])=k$（秩条件）；
	
(2)**外生性**：$\mathbb{E}(Zu)=0$，
	
则称$Z$是$X$的**工具变量**（Instrumental Variable）.

**注4.2.1** $l=k$是矩阵$\mathbb{E}[ZX^\prime]$可逆的必要条件.

&emsp;&emsp;相关性的这一条件我们又称为**秩条件**，要满足该条件须有$l\geq k$，即**阶条件**. 根据是否满足阶条件可分为三种情况：**不可识别**（unidentified）：$l<k$；**恰好识别**（just or exactly identified）：$l=k$；**过度识别**（overidentified）：$l>k$.

&emsp;&emsp;当$l=k$时，注意到式子

\[\mathbb{E}[ZY]=\mathbb{E}[ZX^\prime\beta]=\mathbb{E}[ZX^\prime]\beta\]

则

\[\beta=(\mathbb{E}[ZX^\prime])^{-1}\mathbb{E}[ZY]\]

则$\beta$的IV估计为

\[\hat{\beta}^{IV}=\left(\sum_{i=1}^{n}Z_iX_i^\prime\right)^{-1}\left(\sum_{i=1}^{n}Z_iY\right)=(Z^\prime X)^{-1}Z^\prime Y\]

显然，工具变量估计仅适用于恰好识别的情况.

## 4.3 两阶段最小二乘
&emsp;&emsp;当$l\geq k$时，我们作两阶段最小二乘：

&emsp;&emsp;第一阶段回归：$X$对$Z$进行回归，建立结构方程$X=Z^\prime\gamma+u_1$，得到

\[\hat{\gamma}=(Z^\prime Z)^{-1}Z^\prime X\]

\[\hat{X}=PX=Z(Z^\prime Z)^{-1}Z^\prime X\]

将$X$的结构方程代入$Y$的结构方程，得到

\[Y=X^\prime\beta+u=(\hat{X}+\hat{u}_1)^\prime\beta+u=\hat{X}^\prime\beta+u_2\]

&emsp;&emsp;第二阶段回归：$Y$对$\hat{X}$进行回归，建立结构方程$Y=\hat{X}^\prime\beta+u_2$，得到

\[\hat{\beta}^{TSLS}=(\hat{X}^\prime\hat{X})^{-1}\hat{X}^\prime Y=(X^\prime Z(Z^\prime Z)^{-1}Z^\prime X)^{-1}X^\prime Z(Z^\prime Z)^{-1}Z^\prime Y\]

&emsp;&emsp;由于$Y=X^\prime\beta+u=\hat{X}^\prime\beta+(u+(X-\hat{X})^\prime\beta)$，则有

\[\mathbb{E}[\hat{X}u_2]=\mathbb{E}[\hat{X}(u+(X-\hat{X})^\prime\beta)]=\mathbb{E}[Z^\prime\hat{\gamma}(u+u_2^\prime\beta)]=\mathbb{E}[Z^\prime\hat{\gamma}u]+\mathbb{E}[(u_2Z)^\prime\beta]=0\]

此时$\hat{X}$和扰动项正交.

&emsp;&emsp;当$l=k$时，矩阵$X^\prime Z$可逆，则

\[\hat{\beta}^{TSLS}=(Z^\prime X)^{-1}Z^\prime Y\]

&emsp;&emsp;两阶段最小二乘的实质便是：将内生解释变量$X$分成两个部分，即由工具变量$Z$所造成的外生部分$\hat{X}$和与扰动项相关的其余部分$X-\hat{X}$；然后，被解释变量对外生部分进行回归，从而满足OLS对前定变量的要求而得到一致估计.

## 4.4 工具变量检验
### 4.4.1 相关性检验
&emsp;&emsp;易知当$l=k$时，

\[\hat{\beta}^{TSLS}=(Z^\prime X)^{-1}Z^\prime Y=\beta+(Z^\prime X)^{-1}Z^\prime u\]

当$Z$和$X$相关性微弱，即$Z$为**弱工具变量**时，$(Z^\prime X)^{-1}Z^\prime u$无法依概率收敛到0，此时$\hat{\beta}^{TSLS}$不是$\beta$的一致性估计.

&emsp;&emsp;此时，我们用$F$检验进行弱工具变量的检验，可以证明

\[\mathbb{E}[\hat{\beta}^{TSLS}]-\beta\approx\frac{\hat{\beta}^{OLS}-\beta}{\mathbb{E}[F]-1}\]

其中$F$是两阶段最小二乘第一阶段回归中$\gamma=0$这一检验的$F$统计量（回顾3.4中回归系数的$F$检验）.

&emsp;&emsp;经验法则：当仅有一元内生回归变量时，若$F<10$，$TSLS$估计量有偏，且$TSLS$估计的$t$统计量和置信区间均不可靠.

### 4.4.2 外生性检验
&emsp;&emsp;当外生性不满足时，$\mathbb{E}[Z^\prime u]\neq0$，此时$TSLS$估计量不一致. 此时，我们用**过度识别约束检验**（J检验）对外生性进行检验.

&emsp;&emsp;建立$\hat{u}$对工具变量的回归方程

\[\hat{u}=\delta_0+Z^\prime\delta+W^\prime\xi+e\]

令$F$为检验$\delta=0$的同方差适用$F$统计量，构造$J$统计量

\[J=lF\]

可以证明，当所有工具变量都为外生的原假设和同方差假设下，大样本条件下$J\sim\chi^2(l-k)$.
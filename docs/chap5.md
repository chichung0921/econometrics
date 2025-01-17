# 5 面板数据回归
## 5.1 固定效应模型
### 5.1.1 个体固定效应
**定义5.1.1** 定义

\[Y_{it}=X_{it}^\prime\beta+Z_i^\prime\delta+u_i+\epsilon_{it}\]

为**个体固定效应模型**，其中$X_{it}$随时间和个体而变，$Z_i$不随时间变化但随个体变化，$u_i$为**不可观测**的代表个体异质性且与某个解释变量相关的截距项（称为**个体固定效应**），$\epsilon_{it}$为随时间和个体而变的扰动项.

&emsp;&emsp;若将个体虚拟变量引入该模型，则上式可写成

\[Y_{it}=X_{it}^\prime\beta+Z_i^\prime\delta+\gamma_2D2_i+\gamma_3D3_i+\cdots+\gamma_NDN_i+\epsilon_{it}\]

注意，引入的虚拟变量个数应为$N-1$，若引入$N$个虚拟变量，则会引起完全多重共线性.

#### 固定效应估计
&emsp;&emsp;对于固定效应模型，给定个体$i$，将回归方程两边取时间上的平均，得

\[\bar{Y}_i=\bar{X}_i^\prime\beta+Z_i^\prime\delta+u_i+\bar{\epsilon_{i}}\]

两式相减，得

\[Y_{it}-\bar{Y}_i=(X_{it}-\bar{X}_i)^\prime\beta+\epsilon_{it}-\bar{\epsilon}_{i}\]

此时，可以得到$\beta$的固定效应估计，记为$\hat{\beta}_{FE}$，即

\[\bar{Y}_{it}=\bar{X}_{it}^\prime\beta+\bar{\epsilon}_{it}\]

&emsp;&emsp;在该估计中，扰动项必须与**各期**解释变量均不相关，满足严格外生性，即保证$X_{it}-\bar{X}_i$和$\epsilon_{it}-\bar{\epsilon}_{i}$独立.

#### 一阶差分估计
&emsp;&emsp;对固定效应模型两边进行一阶差分，得

\[Y_{it}-Y_{i,t-1}=(X_{it}-X_{i,t-1})^\prime\beta+\epsilon_{it}-\epsilon_{i,t-1}\]

此时，可以得到$\beta$的一阶差分估计，记为$\hat{\beta}_{FD}$.

&emsp;&emsp;在该估计中，只要求满足$X_{it}-X_{i,t-1}$和$\epsilon_{it}-\epsilon_{i,t-1}$独立这一假设，此一致性条件比固定效应估计的一致性条件更弱. 可以证明，当$T=2$时，$\hat{\beta}_{FE}=\hat{\beta}_{FD}$；当$T>2$时，若$\{\epsilon_{it}\}$独立同分布，$\hat{\beta}_{FE}$比$\hat{\beta}_{FD}$更有效率.

### 5.1.2 时间固定效应
**定义5.1.2** 定义

\[Y_{it}=X_{it}^\prime\beta+\gamma S_t+\epsilon_{it}=X_{it}^\prime\beta+\lambda_t+\epsilon_{it}\]

为**时间固定效应模型**，其中$X_{it}$随时间和个体而变，$\lambda_t$为第$t$期独有的截距项（称为**时间固定效应**），$\epsilon_{it}$为随时间和个体而变的扰动项.

&emsp;&emsp;若将时间虚拟变量引入该模型，则上式可写成

\[Y_{it}=X_{it}^\prime\beta+\lambda_2B2_t+\lambda_3B3_t+\cdots+\lambda_TBT_t+\epsilon_{it}\]

类似地，引入的虚拟变量个数应为$T-1$.

&emsp;&emsp;若将个体固定效应和时间固定效应混合，可建立模型

\[Y_{it}=X_{it}^\prime\beta+Z_i^\prime\delta+\lambda t+u_i+\epsilon_{it}\]

引入所有虚拟变量，可得

\[Y_{it}=X_{it}^\prime\beta+Z_i^\prime\delta+\gamma_2D2_i+\gamma_3D3_i+\cdots+\gamma_NDN_i+\lambda_2B2_t+\lambda_3B3_t+\cdots+\lambda_TBT_t+\epsilon_{it}\]

**注5.1.1** 我们称$\lambda t$为**时间趋势项**，若该项写成虚拟变量的形式，则为**时间固定效应**的一种. 时间趋势项不一定为一次，可以为高次，它可以看成是时间固定效应的线性组合，因此时间固定效应项往往能吸收时间趋势项.

## 5.2 随机效应模型
**定义5.2.1** 定义

\[Y_{it}=X_{it}^\prime\beta+Z_i^\prime\delta+u_i+\epsilon_{it}\]

为**随机效应模型**，其中$u_i$和$X_{it}$、$Z_i$均不相关.

&emsp;&emsp;此时我们可以得到$\beta$的OLS一致性估计. 然而，由于扰动项由$u_i+\epsilon_{it}$组成，不是球形扰动项，因此OLS不是最有效率的. 假设不同个体之间的扰动项互不相关，由于$u_i$的存在，同一个体不同时期的扰动项之间存在自相关，即

\[\mathrm{cov}(u_i+\epsilon_{it},u_i+\epsilon_{is})=\begin{cases}
	\sigma_u^2,\,\mbox{若}t\neq s\\
	\sigma_u^2+\sigma_\epsilon^2,\,\mbox{若}t=s
\end{cases}\]

当$t\neq s$时，自相关系数满足

\[\rho=\frac{\sigma_u^2}{\sigma_u^2+\sigma_\epsilon^2}\]

**注5.2.1** 由于面板数据的特点，虽然通常可以假设不同个体之间的扰动项相互独立，但同一个体在不同时期的扰动项之间往往存在自相关，因此对标准误的估计应使用**聚类稳健标准误**. 所谓聚类，就是由每个个体不同时期的所有观测值所组成.

&emsp;&emsp;因此，可用OLS的残差估计$\sigma_u^2+\sigma_\epsilon^2$，用FE的残差估计$\epsilon^2$，用广义最小二乘法来估计，得到随机效应估计量$\hat{\beta}_{RE}$，即

\[Y_{it}-\hat{\theta}\bar{Y_i}=(X_{it}-\hat{\theta}\bar{X}_i)^\prime\beta+(1-\hat{\theta})Z_i^\prime\delta+[(1-\hat{\theta})u_i+(\epsilon_{it}-\hat{\theta}\bar{\epsilon}_i)]\]

其中$\hat{\theta}$是$\theta$的一致估计量，$\theta$满足

\[\theta=1-\frac{\sigma_\epsilon}{\sqrt{T\sigma_u^2+\sigma_\epsilon^2}},\,0\leq\theta\leq1\]

可以证明，新的扰动项是同方差且无序列相关的.

## 5.3 双重差分法
&emsp;&emsp;现考虑两期面板模型

\[Y_{it}=\alpha+\gamma D_t+\beta X_{it}+u_i+\epsilon_{it}\]

其中，$D_t$为实验期虚拟变量，$u_i$为不可测的个体特征，$X_{it}$为政策虚拟变量，且有

\[D_t=\begin{cases}
	1,\,t=2\\
	0,\,t=1
\end{cases},\,X_{it}=\begin{cases}
	1,\,\mbox{若}i\in\mbox{处理组},\,\mbox{且}t=2\\
	0,\,\mbox{其他}
\end{cases}\]

在2.1中，我们介绍了潜在结果框架，了解到在随机实验的条件下，零条件均值假设的成立. 然而，如果实验未能完全随机化，则$X_{it}$可能与$u_i$有关，从而导致OLS估计不一致.

&emsp;&emsp;因此，我们用一阶差分对上式进行处理，用第二期减去第一期，消去$u_i$，得

\[\Delta Y_{it}=\gamma+\beta\Delta X_{it}+\Delta\epsilon_{it}\]

根据潜在结果框架，

\[\hat{\beta}_{ols}=\Delta\bar{Y}_{T}-\Delta\bar{Y}_{C}\]

这种估计方法称为**双重差分估计**（DD），估计量记为$\hat{\beta}_{DD}$.

&emsp;&emsp;显然，以$\Delta Y_{it}$为被解释变量的双重差分法不适用于多期数据，因此需回到以$Y_{it}$为被解释变量的面板模型.建立与原模型等价的方程

\[Y_{it}=\beta_0+\beta_1 G_i\cdot D_t+\beta_2G_i+\gamma D_t+\epsilon_{it}\]

其中，$G_i$为分组虚拟变量，刻画处理组和控制组本身的差异；$D_t$为时间虚拟变量，刻画实验前后两期本身的差异；互动项$G_i\cdot D_t=X_{it}$刻画处理组的政策效应，且

\[G_{i}=\begin{cases}
	1,\,\mbox{若}i\in\mbox{处理组}\\
	0,\,\mbox{若}i\in\mbox{控制组}
\end{cases},\,D_t=\begin{cases}
	1,\,t=2\\
	0,\,t=1
\end{cases}\]

&emsp;&emsp;下面证明上式中的$\hat{\beta}_1$就是$\hat{\beta}_{DD}$，即两个方程等价.

&emsp;&emsp;令$t=1$，则有

\[Y_{i1}=\beta_0+\beta_2G_i+\epsilon_{i1}\]

令$t=2$，则有

\[Y_{i2}=\beta_0+\beta_1 G_i\cdot D_2+\beta_2G_i+\gamma+\epsilon_{i2}\]

两式相减，得

\[\Delta Y_i=\beta_1 G_i\cdot D_2+\gamma+\Delta\epsilon_{i}=\beta_1 X_{i2}+\gamma+\Delta\epsilon_{i}\]

因此$\hat{\beta}_1=\hat{\beta}_{DD}$.

&emsp;&emsp;双重差分法的优点在于，它同时控制了分组效应$G_i$和时间效应$D_t$. 其隐含假设是，即使没有政策变化，控制组和处理组的时间趋势也一样，在方程中表现为共同的$\gamma D_t$这一项. 

&emsp;&emsp;如果处理组和控制组没有共同的时间趋势，上述模型需要进行调整，我们需要在模型中加入时间趋势项或时间固定效应来控制二者的时间趋势. 如果加入时间趋势项，则引入变量初始值与时间的交乘项；如果加入时间固定效应，则引入变量初始值与时间固定效应的交乘项. 另外，三重差分法也可以控制时间趋势，但设置的交乘项更为复杂.
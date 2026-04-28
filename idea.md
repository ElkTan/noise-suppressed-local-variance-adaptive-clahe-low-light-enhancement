## **修正后的选题：更小、更稳、更像课程论文**

建议把原题改成：

**一种噪声抑制的局部方差自适应 CLAHE 低照度图像增强方法**

英文：

**Noise-Suppressed Local-Variance Adaptive CLAHE for Low-Light Image Enhancement**

核心仍然只做一件事：

**改进 CLAHE 的 clip limit 设置方式。**

但把原来的“直接用局部方差调 clip limit”改成：

**先估计结构方差，再平滑 clip limit 参数图，最后只在亮度通道增强并做轻量色彩补偿。**

这样切入点仍然小，但漏洞被补上了。

------

# **0. 实现修正说明（根据首轮实验反馈）**

首轮实验发现，原始实现里有两个关键问题会让消融实验和参数敏感性实验几乎不变，因此需要对算法说明做进一步修正。

## **问题 1：噪声估计与局部方差量纲不一致**

如果直接用 Laplacian 高频响应的 MAD 去估计噪声，再把该噪声方差直接从亮度块方差中扣除，会出现：

- 局部方差是在亮度域统计得到；
- Laplacian 响应是在导数域统计得到；
- 两者量纲不一致。

这样会导致：

s_{i,j}=\max(v_{i,j}-\lambda\sigma_n^2,0)

中的 \sigma_n^2 过大，最终把很多 tile 的结构强度直接压成 0，使得噪声抑制模块失效。

因此本文实现改为：

1. 先对亮度图像做高斯平滑，得到 \tilde{Y}；
2. 用残差

R=Y-\tilde{Y}

估计噪声强度；
3. 再用 MAD 估计残差域噪声标准差：

\sigma_n = \frac{\operatorname{median}(|R-\operatorname{median}(R)|)}{0.6745}

这样得到的 \sigma_n^2 与局部亮度方差处于同一灰度域，更适合做噪声扣除。

## **问题 2：逐块 CLAHE 时 clip limit 作用不明显**

首轮实现中，对每个 tile 再调用一次 OpenCV CLAHE，并且内部仍使用较密的子块划分。这会导致不同 tile 的 clip limit 虽然数值不同，但实际映射曲线变化很小，进而出现：

- \beta 改变后输出不变；
- C_{\max} 改变后输出不变；
- clip map 平滑前后结果差异很小。

因此本文实现修正为：

- 每个外部 tile 只对应一个内部 CLAHE 区域；
- 对该 tile 仅使用一个局部直方图执行对比度限制。

换言之，tile 级自适应参数必须直接作用于 tile 级直方图映射，而不能再被过细的内部网格结构稀释。

## **问题 3：原始有理函数映射动态范围偏小**

原始映射

C_{i,j}=C_{\min}+(C_{\max}-C_{\min})\frac{s_{i,j}}{s_{i,j}+k}

在当前数据范围下容易让大多数 tile 的 clip limit 落在较窄区间，导致不同区域增强强度接近。

因此修正为：

1. 先对结构强度图做分位数归一化：

\hat{s}_{i,j}=\operatorname{clip}\left(\frac{s_{i,j}-P_{10}(s)}{P_{90}(s)-P_{10}(s)+\epsilon},0,1\right)

2. 再用幂函数映射得到 clip limit：

C_{i,j}=C_{\min}+(C_{\max}-C_{\min})\hat{s}_{i,j}^{\gamma}

其中：

- P_{10}(s),P_{90}(s) 分别表示结构强度的 10% 和 90% 分位数；
- \gamma 用于控制增强分配曲线；
- 可令 \gamma 与 \beta 关联，使 \beta 控制高结构区域的增强力度。

这样做的好处是：

- 可以扩大不同 tile 之间的 clip limit 差异；
- 能避免少数极端值主导全图参数范围；
- 参数敏感性实验中更容易观察到 \beta 与 C_{\max} 的实际影响。

## **问题 4：色度补偿更适合作为可选后处理**

首轮实验还表明，如果在合成数据定量评价中直接对色度通道施加较强补偿，虽然视觉上可能更鲜艳，但会明显改变 RGB 颜色分布，从而拉低基于 RGB 的 SSIM。

因此本文进一步将色度补偿重新定位为：

- **核心算法之外的可选后处理模块**；
- 在合成数据定量实验中，优先评价亮度增强核心算法本身；
- 在真实低照度图像展示中，再使用轻量色度补偿改善主观视觉观感。

对应地，本文的核心定量增强输出为：

I'_{core} = \mathcal{F}(Y) \rightarrow Y'

然后仅在需要视觉展示时，才进一步构造：

I'_{vis} = \mathcal{G}(Y', C_r, C_b)

其中 \mathcal{G} 表示轻量色度补偿模块。

这样的拆分更有利于：

- 单独验证亮度增强与噪声抑制机制是否有效；
- 避免色彩后处理对定量指标造成额外干扰；
- 让论文中的“定量结果”和“主观视觉结果”各自对应明确的目标。

------

# **1. 修正后的核心问题**

传统 CLAHE 能增强局部对比度，但在低照度图像中容易放大噪声。Pizer 等人在 AHE/CLAHE 的早期工作中已经明确指出，AHE 的两个主要问题之一就是在相对均匀区域可能产生 noise overenhancement，因此需要 contrast limiting。 

原方案的问题是：

\sigma_{i,j}^{2}

既可能来自真实纹理，也可能来自噪声。

所以不能直接写：

C_{i,j}=C_{\min}+(C_{\max}-C_{\min})\frac{\sigma_{i,j}^{2}}{\sigma_{i,j}^{2}+k}

而应该改成：

C_{i,j}=C_{\min}+(C_{\max}-C_{\min})\frac{s_{i,j}}{s_{i,j}+k}

其中 s_{i,j} 不再是原始局部方差，而是**噪声抑制后的结构强度估计**。

------

# **2. 修正后的算法主线**

## **Step 1：转换到亮度-色度空间**

输入 RGB 图像 I，转换到 YCbCr：

I_{RGB} \rightarrow (Y, C_b, C_r)

只对亮度通道 Y 做 CLAHE 增强，避免 RGB 三通道分别增强导致明显色偏。

------

## **Step 2：构造“结构方差”而不是原始方差**

原始方案直接计算每个块的局部方差：

\sigma_{i,j}^{2}=\operatorname{Var}(Y_{i,j})

现在改成先对亮度图像做轻量平滑：

\tilde{Y}=G_{\sigma} * Y

其中 G_{\sigma} 是小尺度高斯核，比如 3\times3 或 5\times5。

然后在平滑后的亮度图 \tilde{Y} 上计算局部方差：

v_{i,j}=\operatorname{Var}(\tilde{Y}_{i,j})

这样 v_{i,j} 更接近结构变化，而不是噪声波动。

为了进一步稳健，可以估计一个全局噪声方差 \sigma_n^2，然后做扣除：

s_{i,j}=\max(v_{i,j}-\lambda\sigma_n^2,0)

其中：

- v_{i,j}：平滑后图像块的方差；
- \sigma_n^2：全局噪声方差估计；
- \lambda：噪声扣除系数，建议取 0.5\sim1.0；
- s_{i,j}：最终用于调节 clip limit 的结构强度。

这样可以避免“暗区噪声方差大，所以被误判成纹理区”的问题。

------

# **3. 噪声方差怎么估计？**

为了保持课程作业复杂度，不建议引入复杂噪声模型。可以用一个简单稳健的估计方法：

\sigma_n = \frac{\operatorname{median}(|H|)}{0.6745}

其中 H 是高通滤波或小波 HH 子带系数。

如果不想引入小波，也可以用 Laplacian 高频响应估计：

H = \Delta Y

\sigma_n = \frac{\operatorname{median}(|H-\operatorname{median}(H)|)}{0.6745}

论文里可以说：使用 MAD，即 median absolute deviation，对高频噪声进行稳健估计。

这个步骤足够简单，而且和数字图像处理课程中的空间滤波、噪声估计、频率成分分析都相关。

------

# **4. 自适应 clip limit 公式**

最终建议公式写成：

C_{i,j}^{raw} = C_{\min} + (C_{\max}-C_{\min}) \cdot \frac{s_{i,j}}{s_{i,j}+k}

其中：

s_{i,j}=\max(\operatorname{Var}(\tilde{Y}_{i,j})-\lambda\sigma_n^2,0)

这里 k 不要设成固定常数，而应设成和图像内容相关的自适应参数：

k=\beta \cdot \operatorname{mean}(s_{i,j})+\epsilon

其中：

- \beta 控制曲线陡峭程度；
- \epsilon 防止分母为 0；
- 建议 \beta \in \{0.5,1,2,4\} 做参数敏感性实验。

这样论文里可以避免一个严重质疑：

你的 k 换一张图是不是就失效？

你的回答是：

k 随整幅图像的平均结构强度自适应变化，因此不同图像之间具有更好的尺度一致性。

------

# **5. 平滑 clip limit 参数图，抑制块效应**

计算得到所有块的 C_{i,j}^{raw} 后，不要直接使用。先对 clip limit 矩阵做二维高斯平滑：

C_{i,j} = G_{\rho} * C_{i,j}^{raw}

其中 G_{\rho} 作用在块级参数图上，而不是作用在原图上。

例如图像被分成 8\times8 个 tile，那么 C^{raw} 就是一个 8\times8 的矩阵。对这个小矩阵做一次高斯平滑即可。

这样能降低相邻 tile 的增强强度突变，减少 halo 和 block artifacts。

这一点非常重要，因为它让你的方法从“一个公式”变成“一个考虑工程可实现性的算法”。

------

# **6. 色彩补偿：不要让图像发灰**

只增强 Y 通道后，图像可能变亮但饱和度下降。可以加入一个非常轻量的色彩补偿。

设增强前后亮度分别为：

Y,\quad Y'

定义亮度增益：

r(x)=\frac{Y'(x)+\epsilon}{Y(x)+\epsilon}

然后对色度偏移做补偿：

C_b'(x)=128+\eta \cdot r(x)^\delta \cdot (C_b(x)-128)

C_r'(x)=128+\eta \cdot r(x)^\delta \cdot (C_r(x)-128)

其中：

- 128 是 8-bit YCbCr 中性色度中心；
- \eta 是饱和度补偿强度，建议 0.8\sim1.1；
- \delta 是非线性压缩系数，建议 0.3\sim0.6。

为什么不用你反馈里那个最直接的：

C_{new}=C_{old}\times\frac{Y_{enhanced}}{Y_{original}}

因为这个式子可能在暗区把色度放得过大，造成彩色噪声。用 r^\delta 可以压缩过大的亮度增益。

如果想更简单，也可以在 HSV 空间中做：

S'(x)=\operatorname{clip}(S(x)\cdot r(x)^\delta,0,1)

但我更建议用 YCbCr，因为你的主算法已经在 Y 通道上做增强，叙述更统一。

------

# **7. 修正后的完整算法**

可以把方法命名为：

**NS-LVA-CLAHE**
 Noise-Suppressed Local-Variance Adaptive CLAHE

算法流程如下：

1. 输入 RGB 低照度图像 I；
2. 转换到 YCbCr，得到 Y,C_b,C_r；
3. 对 Y 做轻量高斯平滑，得到 \tilde{Y}；
4. 将 \tilde{Y} 分成 m\times n 个 tile；
5. 计算每个 tile 的平滑局部方差 v_{i,j}；
6. 估计全局噪声方差 \sigma_n^2；
7. 得到结构强度：

s_{i,j}=\max(v_{i,j}-\lambda\sigma_n^2,0)

1. 计算自适应参数：

k=\beta \operatorname{mean}(s_{i,j})+\epsilon

1. 计算原始 clip limit：

C_{i,j}^{raw} = C_{\min} + (C_{\max}-C_{\min}) \frac{s_{i,j}}{s_{i,j}+k}

1. 对 C^{raw} 做块级高斯平滑，得到 C；
2. 对每个 tile 使用对应 C_{i,j} 执行 CLAHE；
3. 使用双线性插值得到增强亮度 Y'；
4. 根据亮度增益 r(x) 做色度补偿；
5. 转回 RGB，得到输出图像。

------

# **8. 修正后的论文贡献**

建议写成三点：

## **贡献 1：噪声抑制的结构方差估计**

不是直接用局部方差，而是用：

s_{i,j}=\max(\operatorname{Var}(\tilde{Y}_{i,j})-\lambda\sigma_n^2,0)

区分真实纹理和低照度噪声。

## **贡献 2：空间平滑的自适应 clip limit**

对 C_{i,j} 参数图做高斯平滑，使相邻图像块增强强度连续，降低块效应和 halo。

## **贡献 3：亮度增强后的轻量色度补偿**

只增强亮度通道以避免色偏，同时根据亮度增益对色度通道做压缩式补偿，缓解图像发灰问题。

这三个贡献都很小，但它们正好对应三个实际问题：

| **问题**           | **对应改进**          |
| ------------------ | --------------------- |
| 噪声被误判成纹理   | 噪声抑制结构方差      |
| 相邻块增强强度突变 | clip limit 参数图平滑 |
| 只增强亮度导致发灰 | 色度补偿              |

------

# **9. 修正后的实验设计**

## **9.1 合成低照度数据**

原公式：

I_{low}=I^\gamma+n

建议改成：

I_{low}=(\alpha I)^\gamma+n

其中：

- I\in[0,1]；
- \alpha\in\{0.2,0.3,0.4\}；
- \gamma\in\{1.5,2.0,2.2\}；
- n 可设为高斯噪声：

n\sim \mathcal{N}(0,\sigma^2)

也可以模拟信号相关的泊松-高斯噪声：

I_{low}=\operatorname{Poisson}((\alpha I)^\gamma \cdot q)/q+n_g

但为了课程作业，建议先用高斯噪声，容易控制变量。

------

## **9.2 对比方法**

建议 baseline 保持少而有力：

1. Low-light input；
2. HE；
3. 固定参数 CLAHE；
4. Gaussian/Bilateral denoising + 固定参数 CLAHE；
5. BM3D + 固定参数 CLAHE；
6. 你的 NS-LVA-CLAHE。

BM3D 是经典图像去噪方法，由 Dabov、Foi、Katkovnik 和 Egiazarian 提出，核心思想是把相似 2D 图像块分组成 3D 数组，在变换域进行协同滤波。把 BM3D+CLAHE 加入 baseline，可以回应“先去噪再 CLAHE 不就行了”的质疑。 

如果嫌 BM3D 实现麻烦，可以保留：

- Bilateral filtering + CLAHE；
- Median filtering + CLAHE。

但最好至少有一个“去噪 + CLAHE” baseline。

------

## **9.3 消融实验**

必须做，且不用太多。

| **方法**                 | **验证目的**               |
| ------------------------ | -------------------------- |
| 固定 CLAHE               | 基准                       |
| 原始方差自适应 CLAHE     | 证明直接用方差会被噪声误导 |
| 结构方差自适应 CLAHE     | 验证噪声扣除有效           |
| 结构方差 + clip map 平滑 | 验证块效应降低             |
| 完整方法 + 色度补偿      | 验证视觉色彩改善           |

这组消融非常关键，因为它能直接回应反馈中指出的所有漏洞。

------

## **9.4 指标**

有参考图像时：

- PSNR；
- SSIM；
- 局部噪声放大指标；
- 暗区方差变化。

SSIM 原始论文是 Wang、Bovik、Sheikh 和 Simoncelli 2004 年的 *Image Quality Assessment: From Error Visibility to Structural Similarity*，其核心思想是用结构信息保持程度评价图像质量，而不是只看误差可见性。 

除了 PSNR/SSIM，我建议你自己定义一个简单指标，专门服务于本文论点。

例如选择原图中低亮度且低梯度区域：

\Omega=\{x\mid Y(x)<T_l,\ |\nabla Y(x)|<T_g\}

然后计算增强前后的暗区噪声放大比：

NAR=\frac{\operatorname{Var}(Y'_{\Omega})}{\operatorname{Var}(Y_{\Omega})+\epsilon}

其中 NAR 是 Noise Amplification Ratio。

这比单纯 PSNR/SSIM 更能证明你的方法确实抑制了暗区噪声。

------

# **10. 参数设置建议**

为了避免参数太多，建议固定如下：

| **参数**   | **建议值**           | **说明**       |
| ---------- | -------------------- | -------------- |
| tile grid  | 8\times8             | CLAHE 常用设置 |
| C_{\min}   | 1.0                  | 最小增强       |
| C_{\max}   | 4.0                  | 最大增强       |
| 高斯平滑核 | 3\times3 或 5\times5 | 用于结构方差   |
| \lambda    | 0.5 或 1.0           | 噪声扣除强度   |
| \beta      | 1.0                  | k 的尺度参数   |
| \eta       | 1.0                  | 色度补偿强度   |
| \delta     | 0.5                  | 压缩亮度增益   |

参数敏感性实验只做两个：

1. \beta\in\{0.5,1,2,4\}
2. C_{\max}\in\{2,3,4,5\}

不要把所有参数都扫一遍，否则作业失控。

------

# **11. 最终论文题目与摘要版本**

## **题目**

**一种噪声抑制的局部方差自适应 CLAHE 低照度图像增强方法**

## **摘要草稿**

针对传统 CLAHE 在低照度图像增强中采用固定对比度限制参数，容易在平坦暗区域放大噪声的问题，本文提出一种噪声抑制的局部方差自适应 CLAHE 方法。该方法首先将 RGB 图像转换到 YCbCr 空间，仅在亮度通道进行局部对比度增强；然后通过平滑亮度图和全局噪声方差估计，构造噪声抑制的局部结构强度，并据此自适应调节各图像块的 clip limit。为降低相邻图像块增强强度突变导致的块效应，本文进一步对 clip limit 参数图进行空间平滑。最后，根据亮度增强比例对色度通道进行轻量补偿，以缓解亮度增强后的图像发灰问题。实验在合成低照度图像和真实低照度图像上进行，并与 HE、固定参数 CLAHE、去噪后 CLAHE 等方法比较。结果表明，所提方法在提升局部对比度的同时能够有效抑制暗区噪声放大，并在 PSNR、SSIM、暗区噪声放大比和主观视觉质量上取得更稳定的表现。

------

# **12. 这版 idea 的定位**

这版不是复杂算法，而是一个很标准的“小改进型课程论文”：

\text{CLAHE} \quad\rightarrow\quad \text{局部方差自适应 CLAHE} \quad\rightarrow\quad \text{噪声抑制的局部方差自适应 CLAHE}

它的优点是：

1. **切入点小**：只改 clip limit；
2. **问题明确**：低照度 CLAHE 噪声放大；
3. **理论可解释**：结构方差控制增强强度；
4. **实验可落地**：不需要训练网络；
5. **反馈闭环完整**：噪声、块效应、颜色、合成数据、baseline 都能回应。

最终建议你就用这个版本，不要再加入 Retinex、小波、暗通道或深度学习。否则题目会重新膨胀。

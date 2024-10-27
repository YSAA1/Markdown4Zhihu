# TimeXer模型结构分解

TimeXer模型的核心在于同时对内生变量（即目标变量）和外生变量进行建模，通过自注意力和交叉注意力机制在不同层次上捕捉时间序列数据中的时间和变量间依赖关系。以下将TimeXer模型分解为五个部分，并配以必要的公式和解释。

## 1. 问题设定

TimeXer主要用于带有外生变量的时间序列预测，给定一个内生时间序列


<img src="https://www.zhihu.com/equation?tex=\mathbf{x}_{1:T} = \{x_1, x_2, \dots, x_T\} \in \mathbb{R}^{T \times 1}
" alt="\mathbf{x}_{1:T} = \{x_1, x_2, \dots, x_T\} \in \mathbb{R}^{T \times 1}
" class="ee_img tr_noresize" eeimg="1">

和多个外生时间序列


<img src="https://www.zhihu.com/equation?tex=\mathbf{z}_{1:T_{ex}} = \{z^{(1)}_{1:T_{ex}}, z^{(2)}_{1:T_{ex}}, \dots, z^{(C)}_{1:T_{ex}}\} \in \mathbb{R}^{T_{ex} \times C}
" alt="\mathbf{z}_{1:T_{ex}} = \{z^{(1)}_{1:T_{ex}}, z^{(2)}_{1:T_{ex}}, \dots, z^{(C)}_{1:T_{ex}}\} \in \mathbb{R}^{T_{ex} \times C}
" class="ee_img tr_noresize" eeimg="1">

其中  <img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1">  表示外生变量的数量， <img src="https://www.zhihu.com/equation?tex=T" alt="T" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=T_{ex}" alt="T_{ex}" class="ee_img tr_noresize" eeimg="1">  为内生和外生变量的历史长度。

目标是利用历史数据  <img src="https://www.zhihu.com/equation?tex=\mathbf{x}_{1:T}" alt="\mathbf{x}_{1:T}" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\mathbf{z}_{1:T_{ex}}" alt="\mathbf{z}_{1:T_{ex}}" class="ee_img tr_noresize" eeimg="1">  预测未来  <img src="https://www.zhihu.com/equation?tex=S" alt="S" class="ee_img tr_noresize" eeimg="1">  个时间步长内的内生变量：


<img src="https://www.zhihu.com/equation?tex=\hat{\mathbf{x}}_{T+1:T+S} = F_{\theta} (\mathbf{x}_{1:T}, \mathbf{z}_{1:T_{ex}})
" alt="\hat{\mathbf{x}}_{T+1:T+S} = F_{\theta} (\mathbf{x}_{1:T}, \mathbf{z}_{1:T_{ex}})
" class="ee_img tr_noresize" eeimg="1">

## 2. 内生变量嵌入（Endogenous Embedding）

TimeXer将内生时间序列分割成多个时间片段（patch），并对每个时间片段生成一个时间标记（token），此外引入了一个全局标记，用于汇总全局信息。

- **时间片段生成**：将内生序列  <img src="https://www.zhihu.com/equation?tex=\mathbf{x}_{1:T}" alt="\mathbf{x}_{1:T}" class="ee_img tr_noresize" eeimg="1">  划分为不重叠的时间片段  <img src="https://www.zhihu.com/equation?tex=\{s_1, s_2, \dots, s_N\}" alt="\{s_1, s_2, \dots, s_N\}" class="ee_img tr_noresize" eeimg="1"> ，其中每个时间片段长度为  <img src="https://www.zhihu.com/equation?tex=P" alt="P" class="ee_img tr_noresize" eeimg="1"> ，总共有  <img src="https://www.zhihu.com/equation?tex=N = \lfloor \frac{T}{P} \rfloor" alt="N = \lfloor \frac{T}{P} \rfloor" class="ee_img tr_noresize" eeimg="1">  个片段。


<img src="https://www.zhihu.com/equation?tex=\{s_1, s_2, \dots, s_N\} = \text{Patchify}(\mathbf{x}_{1:T})
  " alt="\{s_1, s_2, \dots, s_N\} = \text{Patchify}(\mathbf{x}_{1:T})
  " class="ee_img tr_noresize" eeimg="1">

- **片段嵌入**：通过线性映射，将每个时间片段投影到高维向量空间形成时间标记：


<img src="https://www.zhihu.com/equation?tex=\mathbf{P}_{\text{en}} = \text{PatchEmbed}(s_1, s_2, \dots, s_N)
  " alt="\mathbf{P}_{\text{en}} = \text{PatchEmbed}(s_1, s_2, \dots, s_N)
  " class="ee_img tr_noresize" eeimg="1">

  其中  <img src="https://www.zhihu.com/equation?tex=\text{PatchEmbed}(\cdot)" alt="\text{PatchEmbed}(\cdot)" class="ee_img tr_noresize" eeimg="1">  表示一个可训练的线性映射，用于生成每个时间片段的表示。

- **全局标记**：引入可学习的全局标记  <img src="https://www.zhihu.com/equation?tex=\mathbf{G}_{\text{en}}" alt="\mathbf{G}_{\text{en}}" class="ee_img tr_noresize" eeimg="1">  来捕捉整个序列的宏观信息：


<img src="https://www.zhihu.com/equation?tex=\mathbf{G}_{\text{en}} = \text{Learnable}(\mathbf{x}_{1:T})
  " alt="\mathbf{G}_{\text{en}} = \text{Learnable}(\mathbf{x}_{1:T})
  " class="ee_img tr_noresize" eeimg="1">

最终，内生变量的嵌入包括  <img src="https://www.zhihu.com/equation?tex=N" alt="N" class="ee_img tr_noresize" eeimg="1">  个时间标记  <img src="https://www.zhihu.com/equation?tex=\mathbf{P}_{\text{en}}" alt="\mathbf{P}_{\text{en}}" class="ee_img tr_noresize" eeimg="1">  和一个全局标记  <img src="https://www.zhihu.com/equation?tex=\mathbf{G}_{\text{en}}" alt="\mathbf{G}_{\text{en}}" class="ee_img tr_noresize" eeimg="1"> ，它们共同作为输入进入 TimeXer 的 Transformer 编码器。

## 3. 外生变量嵌入（Exogenous Embedding）

与内生变量不同，外生变量以变量级别进行嵌入，每个外生变量被视为一个单独的标记，以便更自然地处理频率不匹配或数据缺失问题。

- **变量嵌入**：通过线性映射，将每个外生变量的时间序列表示为一个独立的变量标记：


<img src="https://www.zhihu.com/equation?tex=\mathbf{V}_{\text{ex}, i} = \text{VariateEmbed}(z^{(i)}_{1:T_{ex}}), \quad i \in \{1, \dots, C\}
  " alt="\mathbf{V}_{\text{ex}, i} = \text{VariateEmbed}(z^{(i)}_{1:T_{ex}}), \quad i \in \{1, \dots, C\}
  " class="ee_img tr_noresize" eeimg="1">

  其中  <img src="https://www.zhihu.com/equation?tex=\text{VariateEmbed}(\cdot)" alt="\text{VariateEmbed}(\cdot)" class="ee_img tr_noresize" eeimg="1">  是一个可训练的线性投影。得到的外生变量表示为  <img src="https://www.zhihu.com/equation?tex=\mathbf{V}_{\text{ex}} = \{\mathbf{V}_{\text{ex}, 1}, \dots, \mathbf{V}_{\text{ex}, C}\}" alt="\mathbf{V}_{\text{ex}} = \{\mathbf{V}_{\text{ex}, 1}, \dots, \mathbf{V}_{\text{ex}, C}\}" class="ee_img tr_noresize" eeimg="1"> 。

## 4. 内生自注意力机制（Endogenous Self-Attention）

TimeXer的内生自注意力机制在时间片段和全局标记之间捕捉时间依赖关系，分以下三步实现信息的双向交互：

- **片段到片段**：自注意力机制在时间片段之间捕捉局部的时间依赖关系：


<img src="https://www.zhihu.com/equation?tex=\mathbf{P}_{\text{en}}^{(l,1)} = \text{LayerNorm} \left( \mathbf{P}_{\text{en}}^{(l)} + \text{Self-Attention}(\mathbf{P}_{\text{en}}^{(l)}) \right)
  " alt="\mathbf{P}_{\text{en}}^{(l,1)} = \text{LayerNorm} \left( \mathbf{P}_{\text{en}}^{(l)} + \text{Self-Attention}(\mathbf{P}_{\text{en}}^{(l)}) \right)
  " class="ee_img tr_noresize" eeimg="1">

- **全局到片段**：通过全局标记，时间片段接收序列整体的宏观信息：


<img src="https://www.zhihu.com/equation?tex=\mathbf{P}_{\text{en}}^{(l,2)} = \text{LayerNorm} \left( \mathbf{P}_{\text{en}}^{(l)} + \text{Cross-Attention}(\mathbf{P}_{\text{en}}^{(l)}, \mathbf{G}_{\text{en}}^{(l)}) \right)
  " alt="\mathbf{P}_{\text{en}}^{(l,2)} = \text{LayerNorm} \left( \mathbf{P}_{\text{en}}^{(l)} + \text{Cross-Attention}(\mathbf{P}_{\text{en}}^{(l)}, \mathbf{G}_{\text{en}}^{(l)}) \right)
  " class="ee_img tr_noresize" eeimg="1">

- **片段到全局**：全局标记通过关注所有时间片段，从而捕捉序列的整体特性：


<img src="https://www.zhihu.com/equation?tex=\mathbf{G}_{\text{en}}^{(l+1)} = \text{LayerNorm} \left( \mathbf{G}_{\text{en}}^{(l)} + \text{Cross-Attention}(\mathbf{G}_{\text{en}}^{(l)}, \mathbf{P}_{\text{en}}^{(l)}) \right)
  " alt="\mathbf{G}_{\text{en}}^{(l+1)} = \text{LayerNorm} \left( \mathbf{G}_{\text{en}}^{(l)} + \text{Cross-Attention}(\mathbf{G}_{\text{en}}^{(l)}, \mathbf{P}_{\text{en}}^{(l)}) \right)
  " class="ee_img tr_noresize" eeimg="1">

最终，内生自注意力的计算表示为：


<img src="https://www.zhihu.com/equation?tex=\mathbf{P}_{\text{en}}^{(l+1)}, \mathbf{G}_{\text{en}}^{(l+1)} = \text{LayerNorm} \left( \left[ \mathbf{P}_{\text{en}}^{(l)}, \mathbf{G}_{\text{en}}^{(l)} \right] + \text{Self-Attention} \left( \left[ \mathbf{P}_{\text{en}}^{(l)}, \mathbf{G}_{\text{en}}^{(l)} \right] \right) \right)
" alt="\mathbf{P}_{\text{en}}^{(l+1)}, \mathbf{G}_{\text{en}}^{(l+1)} = \text{LayerNorm} \left( \left[ \mathbf{P}_{\text{en}}^{(l)}, \mathbf{G}_{\text{en}}^{(l)} \right] + \text{Self-Attention} \left( \left[ \mathbf{P}_{\text{en}}^{(l)}, \mathbf{G}_{\text{en}}^{(l)} \right] \right) \right)
" class="ee_img tr_noresize" eeimg="1">

## 5. 外生-内生交叉注意力（Exogenous-to-Endogenous Cross-Attention）

TimeXer的交叉注意力机制在内生和外生变量之间建立联系，外生变量作为键和值，内生变量的全局标记作为查询。

- **变量到全局**：交叉注意力使得内生变量的全局标记可以从外生变量中获得信息：


<img src="https://www.zhihu.com/equation?tex=\mathbf{G}_{\text{en}}^{(l+1)} = \text{LayerNorm} \left( \mathbf{G}_{\text{en}}^{(l)} + \text{Cross-Attention}(\mathbf{G}_{\text{en}}^{(l)}, \mathbf{V}_{\text{ex}}) \right)
  " alt="\mathbf{G}_{\text{en}}^{(l+1)} = \text{LayerNorm} \left( \mathbf{G}_{\text{en}}^{(l)} + \text{Cross-Attention}(\mathbf{G}_{\text{en}}^{(l)}, \mathbf{V}_{\text{ex}}) \right)
  " class="ee_img tr_noresize" eeimg="1">

这种机制允许外生信息向内生变量流动，从而加强对预测目标的理解。

## 6. 预测层与损失函数

TimeXer通过线性投影将内生变量的嵌入生成预测值：


<img src="https://www.zhihu.com/equation?tex=\hat{\mathbf{x}} = \text{Projection} \left( \left[ \mathbf{P}_{\text{en}}^{(L)}, \mathbf{G}_{\text{en}}^{(L)} \right] \right)
" alt="\hat{\mathbf{x}} = \text{Projection} \left( \left[ \mathbf{P}_{\text{en}}^{(L)}, \mathbf{G}_{\text{en}}^{(L)} \right] \right)
" class="ee_img tr_noresize" eeimg="1">

采用平方损失（L2）衡量预测值与真实值之间的差异：


<img src="https://www.zhihu.com/equation?tex=\text{Loss} = \sum_{i=1}^{S} \left\| x_i - \hat{x}_i \right\|_2^2
" alt="\text{Loss} = \sum_{i=1}^{S} \left\| x_i - \hat{x}_i \right\|_2^2
" class="ee_img tr_noresize" eeimg="1">

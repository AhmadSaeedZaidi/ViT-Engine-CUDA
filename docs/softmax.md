## Safe Softmax Works (Numerical Stability)

Standard softmax calculates probabilities using the exponentials of inputs. However, calculating $e^x$ for large values of $x$ causes numerical overflow in computers (resulting in `NaN`). 

To fix this, libraries use **Safe Softmax**, which subtracts the maximum value $M = \max(\mathbf{x})$ from all inputs before calculating the exponentials. Mathematically, this does not change the final result.

**Standard Softmax:**
$$ \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}} $$

**Safe Softmax:**
$$ \text{SafeSoftmax}(x_i) = \frac{e^{x_i - M}}{\sum_{j} e^{x_j - M}} $$

**Proof of Equivalence:**

By applying the exponent rule $e^{a - b} = e^a \cdot e^{-b}$, we can expand the Safe Softmax equation:

$$ = \frac{e^{x_i} \cdot e^{-M}}{\sum_{j} (e^{x_j} \cdot e^{-M})} $$

Since $e^{-M}$ is a constant relative to the summation index $j$, we can factor it out of the denominator:

$$ = \frac{e^{x_i} \cdot e^{-M}}{e^{-M} \cdot \sum_{j} e^{x_j}} $$

The $e^{-M}$ terms in the numerator and denominator cancel each other out perfectly:

$$ = \frac{e^{x_i}}{\sum_{j} e^{x_j}} $$

This proves that $\text{SafeSoftmax}(x_i) = \text{softmax}(x_i)$, allowing for numerically stable computation without altering the mathematical output.

## Online Softmax (FlashAttention Algorithm)

To compute Safe Softmax without making multiple passes over the data to find the global maximum, FlashAttention uses **Online Softmax**. This algorithm processes data in chunks and maintains running tallies, "correcting" previous sums on the fly using the exponent addition rule.

### The Problem
Given two blocks of sequence data (Block 1 and Block 2), let:
* $m_1, m_2$ be the local maximums of each block.
* $d_1, d_2$ be the local sums of exponentials for each block (e.g., $d_1 = \sum_{x \in B_1} e^{x - m_1}$).
* $m_{new} = \max(m_1, m_2)$ be the true global maximum.

We need to compute the global sum $d_{new} = \sum_{\text{All } x} e^{x - m_{new}}$ using only the previously calculated variables $d_1, m_1, d_2, m_2$.

### The Proof

The global sum can be split into the two blocks:
$$ d_{new} = \sum_{x \in B_1} e^{x - m_{new}} + \sum_{x \in B_2} e^{x - m_{new}} $$

We can add and subtract the local maximums inside the exponents without changing the mathematical value:
$$ d_{new} = \sum_{x \in B_1} e^{(x - m_1) + (m_1 - m_{new})} + \sum_{x \in B_2} e^{(x - m_2) + (m_2 - m_{new})} $$

Using the exponent rule $e^{a+b} = e^a \cdot e^b$, we split the terms:
$$ d_{new} = \sum_{x \in B_1} \left( e^{x - m_1} \cdot e^{m_1 - m_{new}} \right) + \sum_{x \in B_2} \left( e^{x - m_2} \cdot e^{m_2 - m_{new}} \right) $$

Because $e^{m_1 - m_{new}}$ is a constant relative to $x$, we factor it outside the summation:
$$ d_{new} = e^{m_1 - m_{new}} \left( \sum_{x \in B_1} e^{x - m_1} \right) + e^{m_2 - m_{new}} \left( \sum_{x \in B_2} e^{x - m_2} \right) $$

Finally, we recognize that the remaining summations are simply our stored variables $d_1$ and $d_2$:
$$ d_{new} = \left( d_1 \cdot e^{m_1 - m_{new}} \right) + \left( d_2 \cdot e^{m_2 - m_{new}} \right) $$

**Conclusion:** 
By multiplying the previous running sum ($d_1$) by a correction factor $e^{m_1 - m_{new}}$, we perfectly update the denominator for Safe Softmax in a single pass.
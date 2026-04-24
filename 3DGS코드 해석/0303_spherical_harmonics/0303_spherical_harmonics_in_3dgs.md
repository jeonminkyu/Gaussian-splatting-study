# 구면 조화 함수(Spherical Harmonics)

3D GS에서는 각 가우시안 스플랫의 **시점(뷰) 의존적 색상**을 표현하기 위해 **구면 조화 함수(Spherical Harmonics, SH)** 를 사용합니다. 이를 통해 각 가우시안은 보는 방향에 따라 다른 색을 나타낼 수 있습니다. 

장점:
- (특히 GPU에서) 다항식 형태로 빠르게 평가 가능해서 계산 효율이 높습니다.
- 저차수(낮은 band)로도 자연스러운 뷰-의존 색 변화(저주파 성분)를 잘 표현합니다.
- 계수에 대해 선형 결합이라 미분/역전파가 단순하고 안정적(부드러운 변화)입니다.
- 적은 파라미터로 압축 효과가 있습니다(예: $L=3$이면 채널당 16개 계수).

단점:
- 저차수 SH는 고주파(날카로운 스펙큘러/거울 반사, 아주 뚜렷한 경계)를 잘 못 담습니다.
- 표현력을 올리려면 band를 키워야 하고, 계수 수가 $(L+1)^2$로 늘어 메모리/연산이 증가합니다.
- 선형 결합 특성상 중간 값이 음수가 될 수 있어(3DGS는 +0.5 shift 후 clamp) 클램핑에 따른 정보 손실/gradient 이슈가 생길 수 있습니다.
- 계수의 의미가 직관적이지 않아(특정 재질 파라미터처럼) 직접 제어/해석이 어렵습니다.

### 푸리에 급수와의 비교

| Aspect | 1D Fourier | 2D Fourier | Spherical Harmonics (Here) |
|--------|-------------------|-------------------|---------------------------|
| **Domain** | Line (time/space) | Plane (image) | Sphere surface (viewing directions) |
| **Basis Functions** | $\sin(nt), \cos(nt)$ | $e^{i(mx+ny)}$ | $Y_l^m(\theta,\phi)$ |
| **Indices** | $n$ (frequency) | $(m,n)$ (x/y freq) | $(l,m)$ (band/order) |
| **Application** | Audio, seasonal patterns | Image compression (JPEG) | **View-dependent color in 3DGS** |


---

### 수학적 기초

일반적인 구면 조화 함수는 복소수를 출력할 수 있습니다. 수학에서는 입력/출력 대신에 사상(map)이라는 용어를 사용합니다.

예시: "The function $f^{\text{real}}(\theta,\phi)$ **maps** from the 2-sphere to the reals"

$$
f(\theta,\phi) : \mathbb{S}^2 \to \mathbb{C}
$$

**Angles and measure**
- $\theta \in [0,\pi]$: polar angle (from the +z axis)
- $\phi \in [0,2\pi)$: azimuth angle (around the z axis)
- $d\Omega = \sin\theta\, d\theta\, d\phi$: surface element on the unit sphere

**Indices (what “degree/band” means)**
- **$l$ (degree / band)**: controls frequency/detail level ($l=0,1,2,\dots$)
- **$m$ (order)**: orientation within a fixed $l$ ($-l \le m \le l$)

For a fixed $l$, there are $2l+1$ basis functions (all $m$ values). Using degrees $0$ through $L$ gives $(L+1)^2$ total coefficients.

**Spherical harmonics expansion (top-down view)**

We represent the function $f$ as a linear combination of spherical-harmonic basis functions:

$$
f(\theta,\phi) = \sum_{l=0}^{\infty}\sum_{m=-l}^{l} c_{l,m}\, Y_l^m(\theta,\phi)
$$

Here $c_{l,m}$ are the coefficients (trainable in 3DGS, per color channel), and $Y_l^m$ are the basis functions defined next.

In practice we truncate to a finite degree $L$:

$$
f(\theta,\phi) \approx \sum_{l=0}^{L}\sum_{m=-l}^{l} c_{l,m}\, Y_l^m(\theta,\phi)
$$

**Definition (complex SH)**

$$
Y_l^m(\theta,\phi) = N_l^m\, P_l^m(\cos\theta)\, e^{im\phi}, \quad l\ge 0,\; -l\le m\le l
$$

where $P_l^m$ is the associated Legendre polynomial and $N_l^m$ is a normalization constant. One standard choice is:

**Associated Legendre polynomials (detail)**

Let $x = \cos\theta$. The $\theta$-dependence of spherical harmonics is carried by the associated Legendre polynomials $P_l^m(x)$.

Start from the Legendre polynomials $P_l(x)$ (Rodrigues' formula):

$$
P_l(x) = \frac{1}{2^l l!}\,\frac{d^l}{dx^l}(x^2 - 1)^l
$$

Then the associated Legendre polynomials (including the Condon–Shortley phase) are:

$$
P_l^m(x) = (-1)^m (1 - x^2)^{m/2}\,\frac{d^m}{dx^m} P_l(x), \quad 0 \le m \le l
$$

For negative orders, one commonly uses:

$$
P_l^{-m}(x) = (-1)^m\,\frac{(l-m)!}{(l+m)!}\,P_l^{m}(x)
$$

With the Condon–Shortley phase already included in $P_l^m$, a common normalization choice is:

$$
N_l^m = \sqrt{\frac{2l+1}{4\pi}\,\frac{(l-m)!}{(l+m)!}}
$$

With this normalization, SH basis functions are orthonormal on the unit sphere:

$$
\int_{\Omega} Y_l^m(\theta,\phi)\, \overline{Y_{l'}^{m'}(\theta,\phi)}\, d\Omega = \delta_{ll'}\,\delta_{mm'}
$$

Given orthonormality, the coefficients are obtained by projection:

$$
c_{l,m} = \int_{\Omega} f(\theta,\phi)\, \overline{Y_l^m(\theta,\phi)}\, d\Omega
$$

### 실수 구면 조화 함수(Real Spherical Harmonics)

3DGS에서는 실수 결과를 출력하는 **실수** 구면 조화 함수를 사용합니다. 예를 들어서 어떤 가우시안 포인트를 바라보는 방향을 입력하면 RGB 각각의 값들을 실수로 출력합니다.

$$
f^{\text{real}}(\theta,\phi) : \mathbb{S}^2 \to \mathbb{R}
$$

Using a real SH basis, we represent this real-valued signal as a linear combination of **real** basis functions with **real** coefficients.

$$
f^{\text{real}}(\theta,\phi)=\sum_{l=0}^{\infty}\sum_{m=-l}^{l} a_{l,m}\,Y_{l,m}^{\text{real}}(\theta,\phi),\qquad a_{l,m}\in\mathbb{R}.
$$

Complex spherical harmonics contain terms of the form $e^{im\phi}$, where $\phi$ is the azimuthal angle. Using **Euler's formula**:

$$e^{im\phi} = \cos(m\phi) + i\sin(m\phi)$$

We can decompose the complex basis into real and imaginary parts. A common **real SH** basis (used in graphics) is defined by linear combinations of the complex SH:

- $Y_{l,0}^{\text{real}} = Y_l^0$
- For $m>0$:
   - $Y_{l,m}^{\text{real}} = \sqrt{2}\,\mathrm{Re}(Y_l^{m})$  ("cosine" terms)
   - $Y_{l,-m}^{\text{real}} = \sqrt{2}\,\mathrm{Im}(Y_l^{m})$ ("sine" terms)

This matches the intuition from Euler's formula: $\mathrm{Re}(e^{im\phi})=\cos(m\phi)$ and $\mathrm{Im}(e^{im\phi})=\sin(m\phi)$.

**Important (common misunderstanding in graphics):** “real spherical harmonics” does **not** mean “take $\mathrm{Re}(Y_l^m)$ and throw away the rest”.
It means we choose a different (but equivalent) orthonormal basis for each fixed $l$ so that the basis functions are **real-valued**.
For $m>0$ you need *two* real basis functions (a cosine-like and a sine-like one) corresponding to the complex-conjugate pair $\pm m$.

In practice (as in 3DGS), we truncate to a finite maximum degree $L$:

$$
f^{\text{real}}(\theta,\phi)\approx\sum_{l=0}^{L}\sum_{m=-l}^{l} a_{l,m}\,Y_{l,m}^{\text{real}}(\theta,\phi).
$$

In the CUDA implementation, each coefficient is stored as an **RGB vector** (a `glm::vec3`). Mathematically, you can think of this as storing $\mathbf{a}_{l,m}\in\mathbb{R}^3$.
In derivations, we often write a scalar $a_{l,m}\in\mathbb{R}$ and apply the same formula independently per color channel.

### The Resulting Real SH Basis (Used in Graphics)

The formal definition of real spherical harmonics in terms of spherical coordinates $(\theta, \phi)$ is:

$$
Y_{l,m}^{\text{real}}(\theta,\phi) =
\begin{cases}
\sqrt{2}\, N_l^{|m|}\, P_l^{|m|}(\cos\theta)\, \cos(m\phi), & m > 0, \\[6pt]
N_l^{0}\, P_l^{0}(\cos\theta), & m = 0, \\[6pt]
\sqrt{2}\, N_l^{|m|}\, P_l^{|m|}(\cos\theta)\, \sin(|m|\phi), & m < 0.
\end{cases}
$$

where:
- $P_l^m(\cos\theta)$ are the **associated Legendre polynomials**
- $N_l^m$ are **normalization constants** ensuring orthonormality:
  $$N_l^m = \sqrt{\frac{(2l+1)}{4\pi} \cdot \frac{(l-m)!}{(l+m)!}}$$
- The $\sqrt{2}$ factor for $m \neq 0$ accounts for combining complex conjugate pairs

**Key observations**:
1. For $m > 0$: Basis functions contain $\cos(m\phi)$ → vary as cosines in azimuth
2. For $m = 0$: No azimuthal dependence → rotationally symmetric about z-axis
3. For $m < 0$: Basis functions contain $\sin(|m|\phi)$ → vary as sines in azimuth

This piecewise structure directly corresponds to the three cases in the conversion formulas above (Re for $m>0$, unchanged for $m=0$, Im for $m<0$).

**Is this consistent with the 3DGS CUDA implementation?** Yes: the 3DGS rasterizer evaluates a **real SH basis** up to $L=3$, but it uses a particular **graphics-friendly ordering and sign convention**.
In practice, any sign flips can be absorbed into the stored coefficients, so “matching the code” is about matching the exact basis/order used in `computeColorFromSH`.

Concretely, in `computeColorFromSH` the real SH basis is evaluated directly in **Cartesian polynomials** of the normalized direction $(x,y,z)$ with fixed constants `SH_C0`, `SH_C1`, `SH_C2[·]`, `SH_C3[·]`:
- Band 0: `SH_C0 * sh[0]` (constant)
- Band 1: `-SH_C1*y*sh[1] + SH_C1*z*sh[2] - SH_C1*x*sh[3]` (note the minus signs on $y$ and $x$)
- Band 2/3: the same idea with quadratic/cubic polynomials (some constants are negative, which is equivalent to flipping the corresponding basis function sign)

If we denote the CUDA-evaluated real basis functions by $Y_{l,m}^{\text{cuda}}(\mathbf{d})$ (with $\mathbf{d}=(x,y,z)$ normalized), then the code is exactly:

$$
\mathrm{RGB}(\mathbf{d}) = \sum_{l=0}^{3}\sum_{m=-l}^{l} \mathbf{sh}[\mathrm{idx}(l,m)]\; Y_{l,m}^{\text{cuda}}(\mathbf{d}) + 0.5,
$$

where the coefficient ordering is **by increasing $l$, then increasing $m$** (i.e., $m=-l,\ldots,l$), and the mapping is:

| CUDA index | $(l,m)$ | $Y_{l,m}^{\text{cuda}}(x,y,z)$ |
|---:|:---:|---|
| 0 | $(0,0)$ | $\mathrm{SH\_C0}$ |
| 1 | $(1,-1)$ | $-\mathrm{SH\_C1}\,y$ |
| 2 | $(1,0)$ | $\mathrm{SH\_C1}\,z$ |
| 3 | $(1,1)$ | $-\mathrm{SH\_C1}\,x$ |
| 4 | $(2,-2)$ | $\mathrm{SH\_C2}[0]\,xy$ |
| 5 | $(2,-1)$ | $\mathrm{SH\_C2}[1]\,yz$ *(note: `SH_C2[1]` is negative in code)* |
| 6 | $(2,0)$ | $\mathrm{SH\_C2}[2]\,(2z^2-x^2-y^2)$ |
| 7 | $(2,1)$ | $\mathrm{SH\_C2}[3]\,xz$ *(note: `SH_C2[3]` is negative in code)* |
| 8 | $(2,2)$ | $\mathrm{SH\_C2}[4]\,(x^2-y^2)$ |
| 9 | $(3,-3)$ | $\mathrm{SH\_C3}[0]\,y\,(3x^2-y^2)$ |
| 10 | $(3,-2)$ | $\mathrm{SH\_C3}[1]\,xyz$ |
| 11 | $(3,-1)$ | $\mathrm{SH\_C3}[2]\,y\,(4z^2-x^2-y^2)$ |
| 12 | $(3,0)$ | $\mathrm{SH\_C3}[3]\,z\,(2z^2-3x^2-3y^2)$ |
| 13 | $(3,1)$ | $\mathrm{SH\_C3}[4]\,x\,(4z^2-x^2-y^2)$ |
| 14 | $(3,2)$ | $\mathrm{SH\_C3}[5]\,z\,(x^2-y^2)$ |
| 15 | $(3,3)$ | $\mathrm{SH\_C3}[6]\,x\,(x^2-3y^2)$ |

So: your notation $a_{l,m}$ is consistent with CUDA **iff** you interpret $Y_{l,m}^{\text{real}}$ as *this exact* $Y_{l,m}^{\text{cuda}}$ basis (including the embedded sign flips via `SH_C2[1]`, `SH_C2[3]`, `SH_C3[0]`, `SH_C3[2]`, `SH_C3[4]`, `SH_C3[6]`, plus the explicit `-y` and `-x` at band 1).

### Cartesian Coordinate Transformation

A key insight is that spherical harmonics can be expressed directly in Cartesian coordinates $(x, y, z)$ of a normalized direction vector:

$$
\begin{aligned}
x &= \sin\theta \cos\phi \\
y &= \sin\theta \sin\phi \\
z &= \cos\theta
\end{aligned}
$$

#### Why This Matters for GPU Performance

**The problem with spherical coordinates:**

If we used the standard spherical form $Y_l^m(\theta, \phi)$ directly, we would need to:
1. Compute $\theta = \arccos(z/r)$ → expensive `acos()` function
2. Compute $\phi = \arctan2(y, x)$ → expensive `atan2()` function  
3. Evaluate $\sin(\theta), \cos(\theta), \sin(m\phi), \cos(m\phi)$ → more trigonometric functions

**GPU trigonometric functions are slow:**
- `acos()`, `atan2()`, `sin()`, `cos()` can take **10-100 cycles** each on GPU
- For millions of Gaussians × thousands of pixels, this adds up dramatically

**The Cartesian solution:**

By expressing SH directly as polynomials in $(x, y, z)$, we eliminate ALL trigonometric operations:

| Band | Spherical Form (Slow) | Cartesian Form (Fast) | Operations Saved |
|------|----------------------|----------------------|------------------|
| Band 1 | $Y_1^0 \propto \cos\theta$ | $z$ | Just use the z-coordinate! ✅ |
| Band 1 | $Y_1^{\pm1} \propto \sin\theta \cos\phi$ | $x$ | Just use the x-coordinate! ✅ |
| Band 2 | $Y_2^0 \propto (3\cos^2\theta - 1)$ | $2z^2 - x^2 - y^2$ | Only multiplications ✅ |
| Band 2 | $Y_2^{\pm2} \propto \sin^2\theta \cos(2\phi)$ | $x^2 - y^2$ | Only multiplications ✅ |

**Performance gain:**
- **Cartesian**: ~3-5 multiply-add operations per basis function
- **Spherical**: 2 transcendental functions + multiple trig evaluations per basis function
- **Speedup**: Typically **10×-20× faster** in practice!

**Why this works:**
Modern GPUs have specialized hardware for polynomial arithmetic (FMA units: Fused Multiply-Add). Operations like `2*z*z - x*x - y*y` are essentially free compared to `3*cos(theta)*cos(theta) - 1`.

---

## Forward Pass: Color Computation

### Implementation (from `forward.cu`)

The `computeColorFromSH` function evaluates the SH representation to compute RGB color:

```cpp
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, 
                                         const glm::vec3* means, glm::vec3 campos, 
                                         const float* shs, bool* clamped)
{
    // Step 1: Compute viewing direction
    glm::vec3 pos = means[idx];
    glm::vec3 dir = pos - campos;
    dir = dir / glm::length(dir);  // Normalize
    
    // Extract Cartesian components
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    
    // Step 2: Access SH coefficients for this Gaussian
    glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
    
    // Step 3: Evaluate SH bands
    glm::vec3 result = SH_C0 * sh[0];  // Band 0
    
    if (deg > 0)
    {
        // Band 1: Linear terms
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
        
        if (deg > 1)
        {
            // Band 2: Quadratic terms
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                SH_C2[0] * xy * sh[4] +
                SH_C2[1] * yz * sh[5] +
                SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                SH_C2[3] * xz * sh[7] +
                SH_C2[4] * (xx - yy) * sh[8];
            
            if (deg > 2)
            {
                // Band 3: Cubic terms
                result = result +
                    SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                    SH_C3[1] * xy * z * sh[10] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                    SH_C3[5] * z * (xx - yy) * sh[14] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    
    // Step 4: Shift from [-0.5, 0.5] to [0, 1] range
    result += 0.5f;
    
    // Step 5: Clamp negative values and record for backward pass
    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
    return glm::max(result, 0.0f);
}
```

### Mathematical Form

The evaluation computes:

$$\text{RGB}(\mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} \mathbf{c}_{l,m} \cdot Y_l^m(\mathbf{d}) + 0.5$$

#### Expanded Form with Explicit Trainable Coefficients

For **$L=3$** (default in 3DGS), the formula expands to show all **16 trainable coefficient vectors**:

$$
\begin{aligned}
\text{RGB}(\mathbf{d}) = \quad &\underbrace{C_0}_{\text{constant}} \cdot \underbrace{\mathbf{c}_{0,0}}_{\text{trainable}} \\
&+ C_1 \cdot (-y \cdot \mathbf{c}_{1,-1} + z \cdot \mathbf{c}_{1,0} - x \cdot \mathbf{c}_{1,+1}) \\
&+ C_2[0] \cdot xy \cdot \mathbf{c}_{2,-2} \\
&+ C_2[1] \cdot yz \cdot \mathbf{c}_{2,-1} \\
&+ C_2[2] \cdot (2z^2 - x^2 - y^2) \cdot \mathbf{c}_{2,0} \\
&+ C_2[3] \cdot xz \cdot \mathbf{c}_{2,+1} \\
&+ C_2[4] \cdot (x^2 - y^2) \cdot \mathbf{c}_{2,+2} \\
&+ C_3[0] \cdot y(3x^2 - y^2) \cdot \mathbf{c}_{3,-3} \\
&+ C_3[1] \cdot xyz \cdot \mathbf{c}_{3,-2} \\
&+ C_3[2] \cdot y(4z^2 - x^2 - y^2) \cdot \mathbf{c}_{3,-1} \\
&+ C_3[3] \cdot z(2z^2 - 3x^2 - 3y^2) \cdot \mathbf{c}_{3,0} \\
&+ C_3[4] \cdot x(4z^2 - x^2 - y^2) \cdot \mathbf{c}_{3,+1} \\
&+ C_3[5] \cdot z(x^2 - y^2) \cdot \mathbf{c}_{3,+2} \\
&+ C_3[6] \cdot x(x^2 - 3y^2) \cdot \mathbf{c}_{3,+3} \\
&+ 0.5
\end{aligned}
$$

**Trainable Parameters** (optimized during training):
- $\mathbf{c}_{l,m} \in \mathbb{R}^3$ for all $(l,m)$ pairs: **16 RGB vectors = 48 floats per Gaussian**
- Each $\mathbf{c}_{l,m}$ = `glm::vec3(R, G, B)` stored in `sh[0]` through `sh[15]`

**Fixed Constants** (pre-computed normalization factors):
- $C_0 = 0.28209...$, $C_1 = 0.48860...$, $C_2[\cdot]$, $C_3[\cdot]$ (from `auxiliary.h`)
- $(x, y, z)$ components of normalized viewing direction $\mathbf{d}$
- Polynomial basis functions: $xy$, $z^2$, $x^2 - y^2$, etc.

**Coefficient Breakdown by Band:**

| Band | Degree | Count | Indices | Array Positions |
|------|--------|-------|---------|-----------------|
| 0 | $l=0$ | 1 | $m=0$ | `sh[0]` = $\mathbf{c}_{0,0}$ |
| 1 | $l=1$ | 3 | $m \in \{-1,0,+1\}$ | `sh[1:3]` = $\mathbf{c}_{1,-1}, \mathbf{c}_{1,0}, \mathbf{c}_{1,+1}$ |
| 2 | $l=2$ | 5 | $m \in \{-2,...,+2\}$ | `sh[4:8]` = $\mathbf{c}_{2,-2}$ to $\mathbf{c}_{2,+2}$ |
| 3 | $l=3$ | 7 | $m \in \{-3,...,+3\}$ | `sh[9:15]` = $\mathbf{c}_{3,-3}$ to $\mathbf{c}_{3,+3}$ |

where:
- $\mathbf{d} = (x, y, z)$ is the normalized viewing direction
- $\mathbf{c}_{l,m} \in \mathbb{R}^3$ are the stored SH coefficients as **RGB vectors** (`glm::vec3` type: `sh[0]`, `sh[1]`, ...)
  - Each coefficient stores 3 floats (R, G, B channels)
  - Data type: `glm::vec3` = 3 floats = 12 bytes per coefficient
- $Y_l^m(\mathbf{d})$ are the basis functions evaluated in Cartesian form (scalar values)

### Handling Negative Color Values (What the Original 3DGS Actually Does)

Because SH evaluation is a **linear combination** of basis functions, the intermediate RGB produced by the SH sum can be negative (and can also exceed 1). The original 3DGS CUDA rasterizer handles this in a simple, GPU-friendly way:

1. **Add a bias of +0.5** after SH evaluation:
  $$\text{RGB}(\mathbf{d}) \leftarrow \text{RGB}(\mathbf{d}) + 0.5$$
  This shifts the typical range from roughly $[-0.5, 0.5]$ toward $[0, 1]$.

2. **Clamp only the lower bound (ReLU-style), per channel**:
  $$\text{RGB}(\mathbf{d}) \leftarrow \max(\text{RGB}(\mathbf{d}), 0)$$
  In code this is `return glm::max(result, 0.0f);`.

3. **Record which channels were clamped** for the backward pass:
  - `clamped[3*idx + c] = (result[c] < 0)`
  This mask is used during backprop so gradients behave correctly when the clamp is active (i.e., clamped channels should not receive gradients that would “push further negative”).

**Important note**: in this `computeColorFromSH` function, the original code clamps negatives but does **not** clamp the upper bound to 1.0 here.

---

## SH Constants and Normalization

### Constants Definition (from `auxiliary.h`)

The implementation pre-computes normalization constants for bands 0-3:

```cpp
// Band 0 (degree 0): DC component
__device__ const float SH_C0 = 0.28209479177387814f;  // = 1/(2*sqrt(π))

// Band 1 (degree 1): Linear directional variation
__device__ const float SH_C1 = 0.4886025119029199f;   // = sqrt(3/(4*π))

// Band 2 (degree 2): Quadratic variation (5 coefficients)
__device__ const float SH_C2[] = {
    1.0925484305920792f,   // sqrt(15/π) for xy term
    -1.0925484305920792f,  // -sqrt(15/π) for yz term
    0.31539156525252005f,  // (1/4)*sqrt(5/π) for (2z²-x²-y²) term
    -1.0925484305920792f,  // -sqrt(15/π) for xz term
    0.5462742152960396f    // (1/2)*sqrt(15/π) for (x²-y²) term
};

// Band 3 (degree 3): Cubic variation (7 coefficients)
__device__ const float SH_C3[] = {
    -0.5900435899266435f,   // Coefficient for y(3x²-y²)
    2.890611442640554f,     // Coefficient for xyz
    -0.4570457994644658f,   // Coefficient for y(4z²-x²-y²)
    0.3731763325901154f,    // Coefficient for z(2z²-3x²-3y²)
    -0.4570457994644658f,   // Coefficient for x(4z²-x²-y²)
    1.445305721320277f,     // Coefficient for z(x²-y²)
    -0.5900435899266435f    // Coefficient for x(x²-3y²)
};
```

### Derivation of Constants

These constants come from the normalization condition for spherical harmonics:

$$\int_\Omega Y_l^m(\theta,\phi) \cdot Y_{l'}^{m'}(\theta,\phi) \, d\Omega = \delta_{ll'}\delta_{mm'}$$

#### Band 0 Example:
$$Y_0^0 = \frac{1}{2\sqrt{\pi}} \quad \Rightarrow \quad \text{SH\_C0} = 0.28209479...$$

#### Band 1 Derivation:

**Step 1: Complex Spherical Harmonics**

The complex spherical harmonics for $l=1$ are:

$$
\begin{aligned}
Y_1^{-1}(\theta, \phi) &= \sqrt{\frac{3}{8\pi}} \sin\theta \, e^{-i\phi} \\
Y_1^{0}(\theta, \phi) &= \sqrt{\frac{3}{4\pi}} \cos\theta \\
Y_1^{+1}(\theta, \phi) &= -\sqrt{\frac{3}{8\pi}} \sin\theta \, e^{i\phi}
\end{aligned}
$$

**Step 2: Convert to Real Spherical Harmonics**

Using Euler's formula $e^{i\phi} = \cos\phi + i\sin\phi$ and recalling that $Y_1^1$ has a **negative sign** from the Condon-Shortley phase ($-\sqrt{\dots}$):

For $m < 0$ (based on imaginary part of $Y_1^1$):
$$Y_{1,-1}^{\text{real}} = \sqrt{2} \cdot \text{Im}(Y_1^{1}) = \sqrt{2} \cdot \left(-\sqrt{\frac{3}{8\pi}}\right) \sin\theta \sin\phi = -\sqrt{\frac{3}{4\pi}} \sin\theta \sin\phi$$

For $m = 0$ (already real):
$$Y_{1,0}^{\text{real}} = \sqrt{\frac{3}{4\pi}} \cos\theta$$

For $m > 0$ (based on real part of $Y_1^1$):
$$Y_{1,1}^{\text{real}} = \sqrt{2} \cdot \text{Re}(Y_1^{1}) = \sqrt{2} \cdot \left(-\sqrt{\frac{3}{8\pi}}\right) \sin\theta \cos\phi = -\sqrt{\frac{3}{4\pi}} \sin\theta \cos\phi$$

**Step 3: Convert to Cartesian Coordinates**

Using the transformations for a unit vector:
- $x = \sin\theta \cos\phi$
- $y = \sin\theta \sin\phi$
- $z = \cos\theta$

The three band-1 basis functions become:

$$
\begin{aligned}
Y_{1,-1}^{\text{real}} &= -\sqrt{\frac{3}{4\pi}} \cdot y \\
Y_{1,0}^{\text{real}} &= \sqrt{\frac{3}{4\pi}} \cdot z \\
Y_{1,1}^{\text{real}} &= -\sqrt{\frac{3}{4\pi}} \cdot x
\end{aligned}
$$

**Step 4: Extract Constant**

All three share the same constant magnitude:

$$\text{SH\_C1} = \left|\sqrt{\frac{3}{4\pi}}\right| = \sqrt{\frac{3}{4\pi}} \approx 0.4886$$

The **sign information** (minus on $x$ and $y$, plus on $z$) is handled separately in the code evaluation.

This derivation perfectly explains why the code uses `-SH_C1 * y` and `-SH_C1 * x` but `+SH_C1 * z`!

**Step 5: Relationship to Forward Pass**

These constants ($C_0$, $C_1$, $C_2[\cdot]$, $C_3[\cdot]$) are the **magnitudes of the normalization factors** used in the forward pass computation. They ensure the basis functions maintain their orthonormal properties. The signs (like the minus signs for $x$ and $y$ in Band 1) are applied separately during evaluation.

#### Band 2 Example:
One of the five band-2 basis functions:
$$Y_2^0 = \frac{1}{4}\sqrt{\frac{5}{\pi}}(3z^2 - 1) = \frac{1}{4}\sqrt{\frac{5}{\pi}}(2z^2 - x^2 - y^2)$$

Where we use $x^2 + y^2 + z^2 = 1$ for normalized vectors.

**Code Usage:**

````cpp
// Band 2, m=0: Y_2^0 basis function
// Mathematical form: Y_2^0 = (1/4)√(5/π) * (2z² - x² - y²)
// where SH_C2[2] = (1/4)√(5/π) ≈ 0.31539156525252005

// From computeColorFromSH in forward.cu:
float xx = x * x, yy = y * y, zz = z * z;

// Evaluate Y_2^0 and accumulate contribution
result = result + SH_C2[2] * (2.0f * zz - xx - yy) * sh[6];
//                  ^^^^^^   ^^^^^^^^^^^^^^^^^^^^   ^^^^^
//                  constant  polynomial basis      trainable
//                  (1/4)√(5/π)  (2z²-x²-y²)        coefficient c_{2,0}
````

**Breakdown:**
- `SH_C2[2]` = normalization constant $\frac{1}{4}\sqrt{\frac{5}{\pi}}$
- `(2.0f * zz - xx - yy)` = polynomial basis $(2z^2 - x^2 - y^2)$
- `sh[6]` = trainable RGB coefficient vector $\mathbf{c}_{2,0} \in \mathbb{R}^3$

This single line computes the Band 2, order 0 contribution to the final RGB color, adding it to the accumulated result from previous bands.

---

## Implementation Details

### Storage Format

Each Gaussian stores:
- **Position**: 3 floats (x, y, z)
- **SH Coefficients**: $(L+1)^2$ RGB vectors (each is `glm::vec3`)
  - For $L=3$: 16 coefficients × 3 floats each = **48 floats** per Gaussian
  - Total memory: 48 floats × 4 bytes = **192 bytes** of SH data per Gaussian

### Memory Layout

```cpp
// SH coefficients stored as glm::vec3 (RGB vector) per coefficient
// Array structure: [Gaussian 0: vec3×16, Gaussian 1: vec3×16, ...]
glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
// sh[0] = glm::vec3(R0, G0, B0)  // Band 0
// sh[1] = glm::vec3(R1, G1, B1)  // Band 1, m=-1
// ...
// sh[15] = glm::vec3(R15, G15, B15)  // Band 3, m=+3
```

**Key insight**: Each coefficient $\mathbf{c}_{l,m}$ is a `glm::vec3`, not a scalar. This allows each basis function to contribute differently to R, G, and B channels, enabling full color variation.

### Computational Complexity

**Forward pass** per Gaussian:
- Band 0: 1 multiplication per channel
- Band 1: 3 multiplications per channel
- Band 2: 5 multiplications per channel
- Band 3: 7 multiplications per channel
- Total for $L=3$: 16 multiplications × 3 channels = 48 ops

**Backward pass**: Roughly 2× forward pass cost (gradients w.r.t. both coefficients and direction)

### Optimization Tricks

1. **Cartesian evaluation**: Avoids `atan2`, `acos` calls
2. **Pre-computed constants**: Normalization factors computed at compile time
3. **Fused operations**: Direction computation and normalization combined
4. **Early termination**: Can stop at lower bands if desired

---

## References

### Primary 3DGS Papers

1. **Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G.** (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics (SIGGRAPH 2023)*, 42(4).
   - Original paper introducing 3DGS with SH-based view-dependent colors
   - Paper: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
   - Code: https://github.com/graphdeco-inria/gaussian-splatting

2. **Zhang, X., Bi, S., Sunkavalli, K., Su, H., & Xu, Z.** (2022). "Differentiable Point-Based Radiance Fields for Efficient View Synthesis." *SIGGRAPH 2022*.
   - Predecessor work that 3DGS built upon for SH color representation

### Spherical Harmonics Fundamentals

3. **Green, R.** (2003). "Spherical Harmonic Lighting: The Gritty Details." *Game Developers Conference (GDC)*.
   - Excellent practical introduction to SH in computer graphics
   - Available: https://www.cse.chalmers.se/~uffe/xjobb/Readings/GlobalIllumination/Spherical%20Harmonic%20Lighting%20-%20the%20gritty%20details.pdf 

4. **Sloan, P.** (2008). "Stupid Spherical Harmonics (SH) Tricks." *Game Developers Conference (GDC)*.
   - Practical implementation tricks and optimizations
   - Covers rotation, convolution, and efficient evaluation

5. **Ramamoorthi, R., & Hanrahan, P.** (2001). "An Efficient Representation for Irradiance Environment Maps." *SIGGRAPH 2001*.
   - Seminal paper on using SH for environment lighting
   - Established SH as standard tool in graphics

### Mathematical Background

6. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2013). *Mathematical Methods for Physicists* (7th ed.). Academic Press.
   - Chapter on spherical harmonics (comprehensive mathematical treatment)
   - Covers orthogonality, recursion relations, and properties

7. **Wikipedia**: "Spherical Harmonics" - https://en.wikipedia.org/wiki/Spherical_harmonics
   - Good overview with visualizations
   - Includes real vs. complex SH conversion formulas

8. **Wikipedia**: "Table of spherical harmonics" - https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
   - Handy table of explicit low-degree SH expressions (good for sanity-checking signs/order)

### Implementation Resources

9. **3DGS CUDA Implementation**:
   - Forward pass: `diff-gaussian-rasterization/cuda_rasterizer/forward.cu` (lines 20-76)
   - Backward pass: `diff-gaussian-rasterization/cuda_rasterizer/backward.cu` (lines 23-141)
   - Constants: `diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h` (lines 22-38)


### 1.3.3 Fusing adjacent segments

The segment detection used previously is not prefect. One of the imprefection lays in the fact of mutliple segments drawn close to each other to describe the same unique edge. In order to solve this problem, 'close segments' will be fused together.

The segments detected are given in the cartesian space with its two endpoints which makes it difficult to characterize the closeness of two edges. In contrast, the Hough space can easily describe the distance between two segments and fuse them if needed. Thus, the idea of the algorithm is to map the segments in a variant of the Hough space, fuse all close segments then map the segments back to the cartesian space.

#### Mapping from cartesian to a variant of the Hough space

Let's take the example of one segment described by the two points: $A = (a_v, a_h)$ and $B = (b_v, b_h)$ where $(i,j)$ is the point with $i$ in vertical value and $j$ in horizontal values and the origin is fixed at the top left corner of the image.

The Hough space will be described by 4 values : $\theta, \rho, p, d$

![part1_project_Hough_space.png](attachment:part1_project_Hough_space.png)

By convention, $A$ will be such that $a_h < b_h$.

**Computing $\theta$**

$$\theta = arctan \Big(\frac{|b_v - a_v|}{|b_h-a_h|} \Big)$$

Then we want to have $\theta \in ]\pi/2, \pi [$ if the slope of the line is decreasing, thus : 

$$\theta_{final} = \pi - \theta \quad \text{ if $a_v < b_v$}$$

**Computing $\rho$**

$$\rho = \frac{|| \overrightarrow{AB} \times \overrightarrow{OA} ||}{|| \overrightarrow{AB} ||}$$

$\rho$ must be lower than zero  when $c_h < 0$. However, $c_h$ has not been computed yet. Thus, the intersection between the line and $x = 0$ will be computed instead. If, this point has a horizontal value lower than 0, it means $c_h < 0$ and thus $\rho < 0$. 

$$\quad \left\{
    \begin{array}{ll}
      y = mx+n \\
      x  = 0
    \end{array}
  \right.$$ 
$$\Leftrightarrow \left\{
    \begin{array}{ll}
      y = \frac{a_h-b_h}{a_v-b_v}x+(a_h - a_v \frac{a_h-b_h}{a_v-b_v}) \\
      x  = 0
    \end{array}
  \right.$$ 
$$\Leftrightarrow \left\{
    \begin{array}{ll}
      y = a_h - a_v \frac{a_h-b_h}{a_v-b_v} \\
      x  = 0
    \end{array}
  \right.$$ 

Thus, $$\rho_{final} = - \rho \quad \text{ if $a_h - a_v \frac{a_h-b_h}{a_v-b_v} < 0$}$$


**Computing C**
$$c_v = \rho * cos(\theta)$$
$$c_h = \rho * sin(\theta)$$

**Computing $p$ and $d$**
$$p = \left\{
        \begin{array}{ll}
          -sign(c_h-a_h) * ||\overrightarrow{AC}|| & \text{ if } a_h < b_h \text{ or } (a_h = b_h \text{ & } a_v > b_v) \\
          -sign(c_h-b_h) * ||\overrightarrow{BC}|| & \text{ else }
        \end{array}
      \right.$$
$$d = ||\overrightarrow{AB}||$$


#### Mapping back from the variant of the Hough space to  the cartesian space
This action allows to go back in the cartesian space after fusing the close segments.

**Computing C**
$$c_v = \rho * cos(\theta)$$
$$c_h = \rho * sin(\theta)$$

**Computing A and B**
$$A : \left\{
        \begin{array}{ll}
          a_v =  c_v - sign(\frac{\pi}{2}-\theta) * p * sin(\theta)\\
          a_h =  c_h + sign(\frac{\pi}{2}-\theta) * p * cos(\theta)
        \end{array}
      \right.$$
$$B : \left\{
        \begin{array}{ll}
          b_v = c_v - sign(\frac{\pi}{2}-\theta) * (p+d) * sin(\theta) \\
          b_h = c_h + sign(\frac{\pi}{2}-\theta) * (p+d) * cos(\theta)
        \end{array}
      \right.$$

#### Fusing two segments
Two segments will be fused together if they fulfill the conditions below:

$$\left\{
        \begin{array}{ll}
          \Big| \theta_1-\theta_2\Big| \leq \Delta_{\theta}\\
          \Big| \rho_1-\rho_2\Big| \leq \Delta_{\rho}\\
          \Big( p_1 \leq p_2 \quad \text{ & } \quad p_1+d_1 > p_2 \Big) \text{ or }
				    \Big( p_2 \leq p_1  \quad \text{ & } \quad  p_2+d_2 > p_1 \Big)
        \end{array}
      \right.$$

The two first conditions are restricting the orientation and position of the lines extending the segments while the last one restrict the position of the segment on the line.


Then to fuse two segments, the two segments are removed and a new one is created with the following values:

$$\left\{
        \begin{array}{ll}
          \theta_n = \frac{\theta_1+\theta_2}{2}\\
          \rho_n = \frac{\rho_1+\rho_2}{2}\\
          p_n = min(p_1, p_2)\\
          d_n = max(p_1+d_1, p_2+d_2) - p_n
        \end{array}
      \right.$$


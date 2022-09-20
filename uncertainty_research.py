import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt

count = 50

# u_min = 7*1e5
# u_max = u_min + 1e8
# u_count = 50

# p_min = 2*1e-2
# p_max = p_min + 0.08
# p_count = 50

u_min = 7*1e5
u_max = u_min + 1e8
u_count = 50

p_min = 2*1e-2
p_max = p_min + 0.08
p_count = 50

u = np.arange(u_min, u_max, (u_max - u_min)/u_count)
p = np.arange(p_min, p_max, (p_max - p_min)/p_count)
U, P = np.meshgrid(u, p)

c = 5
H = 6

up = U*P
upc = U*np.power(P, c)

Z = 2*(upc + 1)*np.power(c, H)/(up + 1)

Z_zero = np.zeros(Z.shape)

alpha = 0.5

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(U, P, np.log10(Z), rstride=1,
                       cstride=1, cmap=cm.jet, linewidth=0, antialiased=False, alpha=alpha)
# ax.plot_surface(U, P, Z_zero, rstride=1,
#                 cstride=1, cmap=cm.gray, linewidth=0, antialiased=False, alpha=alpha)
ax.set_xlabel("u")
ax.set_ylabel("p")
ax.set_zlabel("log_10(Eg/Ed)")
plt.show()

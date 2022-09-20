from Simulator import UncertaintyCond
import matplotlib.pyplot as plt
import numpy as np

count1 = 100
count2 = 100
uc = UncertaintyCond(0.2, 100)

res_ar = np.array([])
res_p = np.array([])
for i in range(count1):
    ar, p = uc.sample(count2)
    res_ar = np.concatenate((res_ar, ar))
    res_p = np.concatenate((res_p, p))

print(sum(res_ar > 5)*100./count1/count2)
print(sum(res_ar < 5)*100./count1/count2)

plt.hist(res_ar)
plt.show()

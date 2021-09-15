import matplotlib.pyplot as plt
from data import doData, mixedData

t1 = doData.iloc[:, 0]
do_in = doData.iloc[:, 2]
do_second = doData.iloc[:, 3]
do_third = doData.iloc[:, 4]
do_forth = doData.iloc[:, 5]
do_effluent = doData.iloc[:, 6]
do_flow = doData.iloc[:, 1]

t2 = mixedData.iloc[:, 0]
mixed_in = mixedData.iloc[:, 2]
mixed_second = mixedData.iloc[:, 3]
mixed_third = mixedData.iloc[:, 4]
mixed_forth = mixedData.iloc[:, 5]
mixed_effluent = mixedData.iloc[:, 6]
mixed_flow = mixedData.iloc[:, 1]

plt.figure()
plt.plot(t1, do_in, label='进水')
plt.show()
plt.figure()
plt.plot(t1, do_second, label='2L')
plt.plot(t1, do_third, label='3L')
plt.plot(t1, do_forth, label='4L')
plt.show()
plt.figure()
plt.plot(t1, do_effluent, label='出水')
plt.show()


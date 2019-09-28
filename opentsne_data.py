import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('bmh')


d_vals_mnist = [7,22,27,25,38,57,86,129,194,291,437,656]
d_vals_norb = [7, 11, 17, 25, 38, 57, 86, 129, 194, 291, 437, 656, 985, 1477, 2216, 3325, 4987, 7481]

acc_mnist = [0.4094, 0.5277, 0.7213, 0.8152, 0.8855, 0.9207, 0.9345, 0.9445, 0.9498, 0.9529, 0.9549, 0.9568]
true_mnist_acc = 0.9588
time_mnist = [185.9748518, 171.0116487, 177.0415264, 178.67, 180.1015150, 180.2270748, 180.3537571, 181.02, 187.8079300, 189.2498338, 197.5514301, 209.3710042]
true_mnist_time = 209.0762799

acc_kmnist = [0.2927, 0.4125, 0.5448, 0.6888, 0.8232, 0.8768, 0.9147, 0.9337, 0.9441, 0.9495, 0.9523, 0.9519]
true_kmnist_acc = 0.9582
time_kmnist = [163.8790, 174.5829, 182.0811, 231.5397, 216.6945, 194.6193, 194.4430, 194.8401, 199.3443, 205.0506, 210.2584, 222.3113]
true_kmnist_time = 225.6847

acc_fashion_mnist =[0.5389, 0.6364, 0.6998, 0.7238, 0.7533, 0.7614, 0.7821, 0.7843, 0.7850, 0.7892, 0.791 , 0.7936]
true_fashion_mnist_acc = 0.7947
time_fashion_mnist = [ 167.8972751759993, 175.1750074340007, 181.58003444000042, 184.71215555300023, 186.27375400300025, 186.61024547000034, 190.6421691630003, 197.8325613980005, 195.8524769630003, 202.22533570799988, 208.0037303399995, 222.36887219399978]
true_fashion_mnist_time = 222.36013991799973


acc_norb = [0.5924375, 0.7372083, 0.8214166, 0.8273541, 0.7946666, 0.7934583, 0.7785833, 0.768875 , 0.7775833, 0.7689375, 0.7563958, 0.7696458, 0.7695833, 0.7681666, 0.7691458, 0.7587708, 0.7585625, 0.7666458]
true_norb_acc = 0.7634791
time_norb = [149.303615648, 137.080802422, 138.643598361, 139.983338415, 141.91233770400004, 142.65694187999998, 145.38589050200005, 144.428661078, 147.9448315720001, 149.220199782, 153.96759155999985, 160.41284792300007, 170.8012414299999, 182.282863035, 207.22143607500038, 246.4354574439999, 330.31511787, 512.9663577339998]
true_norb_time =  501.401006918



# plt.plot([np.log(d)/np.log(1.5) for d in d_vals], [k/true_mnist_acc for k in acc_mnist], ".--", label="Accuracy")
# plt.plot([np.log(d)/np.log(1.5) for d in d_vals], [k/true_mnist_time for k in time_mnist], ".--", label="Time")
# plt.legend()
# plt.title("Mnist Dataset")
# plt.xlabel("Log(Dimension)")
# plt.ylabel("Ratio")
# plt.show()

plt.plot([np.log(d)/np.log(1.5) for d in d_vals_norb], [k/true_norb_acc for k in acc_norb], ".--", label="Accuracy")
plt.plot([np.log(d)/np.log(1.5) for d in d_vals_norb], [k/true_norb_time for k in time_norb], ".--", label="Time")
plt.legend()
plt.title("Norb Dataset")
plt.xlabel("Log(Dimension)")
plt.ylabel("Ratio")
plt.show()
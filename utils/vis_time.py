import pandas as pd
import matplotlib.pyplot as plt

df_cuda = pd.read_csv('..//time_cuda.csv')
df_cpu = pd.read_csv('..//time_cpu.csv')

# print(df_cuda.to_string())
# print("\n")
# print(df_cpu.to_string())
# print("\n")

epoch_cuda = df_cuda.iloc[:, 0]
time_cuda = df_cuda.iloc[:, 1]
time_cuda_h = time_cuda.transform(func = lambda x : x / 3600)

epoch_cpu = df_cpu.iloc[:, 0]
time_cpu = df_cpu.iloc[:, 1]
time_cpu_h = time_cuda.transform(func = lambda x : x / 3600)

# fig=plt.figure()

plt.figure(figsize=(8, 6))
plt.title('CPU vs GPU')

plt.plot(epoch_cuda, time_cuda_h, label='gpu', color="C0")
# plt.scatter(epoch_cuda, time_cuda_h, color="C0")
plt.plot(epoch_cpu, time_cpu_h, label='cpu', color="C1")
# plt.scatter(epoch_cpu, time_cpu_h, color="C1")


'''ax1=fig.add_subplot(111, label="gpu")
ax2=fig.add_subplot(111, label="cpu", frame_on=False)

ax1.set_xlabel("time")
ax2.set_ylabel("epoches")

ax1.plot(time_cuda_h, epoch_cuda, label='gpu', color="C0")
ax1.scatter(time_cuda_h, epoch_cuda, color="C0")

ax2.plot(time_cpu_h, epoch_cpu, label='cpu', color="C1")
#ax2.scatter(time_cpu_h, epoch_cpu, color="C1")'''

plt.legend()
# fig.legend()
plt.show()







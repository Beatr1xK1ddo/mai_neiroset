import numpy as np
import matplotlib.pyplot as plt

def fun1(t):
    return np.sin(t**2 - 2*t + 3)

def res_fun(t):
    return 1/4*np.sin(t**2 - 2*t)

def draw(time, x1, predict_res, delay): 
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('X1')

    plt.plot(time[delay:], x1[delay:], 'r', label='Default')
    plt.plot(time[delay:], predict_res, 'b', label='Predict')
    plt.legend()
    plt.show()

time = np.linspace(0, 6, int(6 / 0.025))
x1 = fun1(time)
x2 = fun1(time)
y = res_fun(time)

def countFunc(x1, x1_predict, learn=0.001 ,periods=50, delay=5):
    # Генерируем начальный массив из 6 столбцов, 
    # где 1 элемент это единица, а 2-6 - это элемент в диапазоне от i - i+5
    data = np.array([np.hstack([1, x1[i:i+delay]]) for i in range(len(x1) - delay)]) 
    # Генерируем веса от -0.1 до 0.1 в размере делей + 1 т.к. у нас 6 столбцов и необходимо 6 весов
    w = np.random.uniform(-0.1, 0.1, delay + 1)
    # Обучаем сеть с периодом в 50 раз
    for _ in range(periods):
        exit = data @ w
        error = x1_predict - exit
        w += learn * data.T @ error
    predict_res = data @ w
    draw(time, x1, predict_res, delay)
    return w, predict_res, ((predict_res - x1_predict)**2).sum() / len(data)

weights_1, predict_1, mse_1  = countFunc(x1, x1[5:], 0.001, 50)
print('среднеквадратическая ошибка: 1')
print(f'RMSE = {np.sqrt(mse_1)}') 

# 2nd 
weights_2, predict_2, mse_2  = countFunc(x1, x1[3:], 0.002, 600, 3)
print(f'RMSE = {np.sqrt(mse_2)}')
print('среднеквадратическая ошибка: 2')
test = np.array([5 + 0.025 * i for i in range(1, 11)])
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('X1')
plt.plot(test, fun1(test), 'r', label='Default')
tmp_x = x1[-3:]
for i in range(10):
    tmp_x = np.append(tmp_x, np.hstack([1, tmp_x[-3:]]) @ weights_2)
plt.plot(test, tmp_x[3:], 'b', label='Predict')
plt.legend()
plt.show()

# 3rd
weights_3, predict_3, mse_3  = countFunc(x1, x1[3:], 0.002, 600, 3)
print('среднеквадратическая ошибка: 3')
print(f'RMSE = {np.sqrt(mse_3)}')




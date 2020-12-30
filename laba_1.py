import numpy as np
from matplotlib import pyplot as plt

def run() :
    etta = 0.3  # Коэффициент изначального значения весов 
    learn_rate = 0.2 # Свободный коэффициент обучающий сеть
    weights = [] # Массив весов 
    periods = 10 # Здесь вы можете установить скорость обучения сети 
                 # (кол-во операций за которые нейросеть будет обучаться)
    matrix = [#  bias  x1    x2    y
                [1.0, -1.1, -4.3, 0.0],
                [1.0, 1.8, -1.0, 1.0],
                [1.0, 4.8, -1.0, 1.0],
                [1.0, 1.2, -3.5, 1.0],
                [1.0, -1.2, -3.4, 0.0],
                [1.0, 2.5, 3.7, 1.0]
            ]
    # matrix = [
    #             [1, -0.8, -2.5, 0.0],
    #             [1, -2.1, -0.8, 0.0],
    #             [1, -3.9, -0.1, 0.0],
    #             [1, 2.0, -2.6, 0.0],
    #             [1, 2.8, -4.3, 0.0],
    #             [1, -1.1, -5.0, 1.0],
    #             [1, -2.8, -5.0, 1.0],
    #             [1, -3.2, -3.6, 1.0]
    #         ]

    # matrix = [
    #             [1, -0.8, -2.5, 1.0],
    #             [1, -2.1, -0.8, 0.0],
    #             [1, -3.9, -0.1, 0.0],
    #             [1, 2.0, -2.6, 1.0],
    #             [1, 2.8, -4.3, 1.0],
    #             [1, -1.1, -5.0, 1.0],
    #             [1, -2.8, -5.0, 0.0],
    #             [1, -3.2, -3.6, 0.0]
    #         ]

    # Создаем 3 веся стартовых для смещения (bias) и наших х1 и х2 
    for i in range(3):
        weight = np.dot(etta, matrix[i][:1])
        weights.append(weight)
    learn_pers(matrix, weights, learn_rate)

# Функция суммирующая результат работы весов и входных данных    
def summ_pers(inputs, weights):
    persCount = 0
    for i,w in zip(inputs, weights):
        persCount += i*w
    return 1.0 if persCount >= 0.0 else 0.0

# Функция учащая персептрон
def learn_pers(matrix, weights, learn_rate, periods=10):
    for period in range(periods):
        checker = check_pers(matrix, weights)
        if checker == 1: # Каждый новый круг мы проверяем персептрон на сходимость
            print('ready!!!')
            return
        for i in range(len(matrix)): # Меняем коэффициенты весов в случае, если фу-ия "check_pers" !== 1
            current_res = summ_pers(matrix[i][:-1], weights)
            if current_res == matrix[i][-1]:
                continue
            else:
                error = matrix[i][-1] - current_res
                for j in range(len(weights)):
                    weights[j] = weights[j] + (learn_rate * error * matrix[i][j])

        showSchedule(matrix, weights)

# Функция проверяющая сходимость персептрона
def check_pers(matrix, weights):
    count = 0 
    for i in range(len(matrix)):
        result = summ_pers(matrix[i][:-1], weights)
        if result == matrix[i][-1]: count += 1
    return count / float(len(matrix))

def showSchedule(matrix, weights):
    # Характеристики графика
    fig,ax = plt.subplots()
    ax.set_title('Chart')
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")

    # Отрисовка плоскости 
    map_min=-5.0
    map_max=5.1
    y_res=0.1
    x_res=0.1
    ys=np.arange(map_min,map_max,y_res)
    xs=np.arange(map_min,map_max,x_res)
    zs=[]
    for cur_y in np.arange(map_min,map_max,y_res):
        for cur_x in np.arange(map_min,map_max,x_res):
            zs.append(summ_pers([1.0,cur_x,cur_y],weights))
    xs,ys=np.meshgrid(xs,ys)
    zs=np.array(zs)
    zs = zs.reshape(xs.shape)
    cp=plt.contourf(xs,ys,zs,levels=[-1,0,1],colors=('g','r'),alpha=0.3)

    # Выходные точки графика
    dot_exit_0 = [[], []]
    dot_exit_1 = [[], []]
    for i in range(len(matrix)):
        cur_x1 = matrix[i][1]
        cur_x2 = matrix[i][2]
        if matrix[i][-1] == 0:
            dot_exit_0[0].append(cur_x1)
            dot_exit_0[1].append(cur_x2)
        else:
            dot_exit_1[0].append(cur_x1)
            dot_exit_1[1].append(cur_x2)
    ax.scatter(dot_exit_0[0], dot_exit_0[1], c='deeppink')
    ax.scatter(dot_exit_1[0], dot_exit_1[1], c='b')

    plt.show()

if __name__ == '__main__':
    run()
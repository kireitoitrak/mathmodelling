import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

class Mnk:
    global StatTab

    StatTab=[]
    ######  ДОРАБОТАТЬ ИМОРТ ДАННЫХ ДЛЯ НАРЕЗКИ!!!!!!!
    def __init__(self,file:str,name:str,N:int):
        self.N=N
        xl = pd.ExcelFile(file)
        df1 = xl.parse(name)
        StatTab.append(df1) # Окончание импорта данных из Excel
        self.df=df1
    ##### ДОРАБОТАТЬ ИМОРТ ДАННЫХ ДЛЯ НАРЕЗКИ!!!!!!!
    """
    def Умный анализ данных(self):
        Анализ данных на основе их типов и создание кучи отдельных 
        матриц-столбцов на их основе, а пока чисто по примеру. 
    """  
    def badan(self):
        x0 = self.df.iloc[:self.N:,0] #фиктивная переменная
        
        x1 = []
        for i in range(self.N):
            x1.append(i+1)
        i=0
        
        x1 = pd.Series(x1,name="Часы, общее кол-во") # количество часов
        x2 = self.df.iloc[:self.N:,1].dt.day
        x3 = self.df.iloc[:self.N:,1].dt.month # месяцы
        x4 = self.df.iloc[:self.N:,2] # час суток
        x5 = self.df.iloc[:self.N:,3] # рабочий\нерабочий день

        X1 = self.df.iloc[:self.N:,4]
        X2 = self.df.iloc[:self.N:,5]
        x6 = []
        for i in range(len(x0)):
            x6.append(int(abs((int(X1[i][0]+X1[i][1])*60)+(int(X1[i][3]+X1[i][4]))-
                              ((int(X2[i][0]+X2[i][1])*60)+(int(X2[i][3]+X2[i][4]))))/60))
        del X1, X2
        x6 = pd.Series(x6,name="Световые часы") # световые часы
        y = self.df.iloc[:self.N:,6] # зависимая переменная
        
        self.daTab = [x0,x1,x2,x3,x4,x5,x6,y]
        
    """ 
    def Проверка на разложение и прочие и прочие факторы, можно 
    ли вообще работать с этим набором данных.
    """
    def graf(self):
        plt.plot(self.daTab[1],self.daTab[7])
        plt.title("График зависимости переменных.")
        plt.ylabel('у - Потребление')
        plt.xlabel('х1 - сутки')
        plt.grid()
        plt.show()
        print('')
    
    def prompt(self,prompt:str):
        StatTab.append(["Модель: "+prompt])
        prompt = prompt.replace("x","")
        prompt = prompt.replace(","," ")
        prompt = prompt.split()
        f =[]
        for i in range(len(prompt)):
            if (len(prompt[i])==1):
                f.append(self.daTab[int(prompt[i])])

            if ((len(prompt[i])==3)and(prompt[i][1]=="+")):
                f.append(self.daTab[int(prompt[i][0])]+self.daTab[int(prompt[i][2])])
            if ((len(prompt[i])==3)and(prompt[i][1]=="-")):
                f.append(self.daTab[int(prompt[i][0])]-self.daTab[int(prompt[i][2])])
            if ((len(prompt[i])==3)and(prompt[i][1]=="*")):
                f.append(self.daTab[int(prompt[i][0])]*self.daTab[int(prompt[i][2])])
                
            if ((len(prompt[i])==4)and(prompt[i][1]=="*")and(prompt[i][2]=="*")):
                f.append(self.daTab[int(prompt[i][0])]**int(prompt[i][3]))
        f = tuple(f)
        
        k = len(self.daTab)
        x = np.vstack(f)#Неполный полином второй степени(Pnp).
        Y = np.vstack(self.daTab[7])
        x = np.transpose(x) #### ПРОВЕРКА ПРОВЕРКА ПРОВЕРКА
        xt = np.transpose(x) #### ПРОВЕРКА ПРОВЕРКА ПРОВЕРКА
        
        B = np.matmul(np.matmul(np.linalg.inv(np.matmul(xt,x)),xt),Y)
        StatTab[-1].append(("Коэффициенты мат модели:",B))
        YR = np.matmul(x,B)
        StatTab[-1].append(("Y расчётные значения:",YR))
        
        ## График зависимости YR от дней
        plt.plot(self.daTab[1],YR)
        plt.title("График зависимости переменных.(Pnp)")
        plt.ylabel('YR - Потребление')
        plt.xlabel('X1 - сутки')
        plt.grid()
        plt.show()
        print('')
        ##
        
        Dad = np.divide(np.sum((Y-YR)**2),(self.N-k))
        StatTab[-1].append(("Дисперсия адекватности:",Dad))
        YSR = np.sum(Y/self.N)
        StatTab[-1].append(("Средняя арифметическая зависимоть переменной:",YSR))
        DY = np.divide(np.sum((Y-YSR)**2),self.N-1)
        StatTab[-1].append(("Дисперсия зависимой пременной:",DY))
        FR = np.divide(DY,Dad)
        StatTab[-1].append(("Расчётное значение коэф. Фишера:",FR))
        F = scipy.stats.f.ppf(q = 1-0.05,dfn = self.N-1,dfd = self.N-k)
        StatTab[-1].append(("Табличное значение коэф. Фишера:",F))
        
        if FR > F:
            StatTab[-1].append(("В связи с тем, что",FR,'>',F,
                  "уравнение регрессии признано адекватным экспериментальным путём."))
        else:
            StatTab[-1].append(("В связи с тем, что",FR,'<',F,
                  "уравнение регрессии признано не адекватным экспериментальным путём."))
        
        corr = np.corrcoef(Y,YR,rowvar = False)
        StatTab[-1].append(("Коэффициент корреляции:",corr))
        tcorr = abs(corr[0][1]) * np.sqrt((self.N-2)/(1-(corr[0][1]**2)))
        StatTab[-1].append(("tcorr:",tcorr))
        t = scipy.stats.t.ppf(0.975,self.N-k)
        StatTab[-1].append(("t:",t))
        corrCr = np.sqrt((t**2)/((t**2) + self.N - 2)) 
        StatTab[-1].append(("corrCr:",corrCr))
        
        if corr[0][1] > corrCr:
            StatTab[-1].append(('\nКоэффициент корреляции признан статичстически значимым:',
                  '\nКоэффицент корреляции:',corr[0][1],'\t','Критическое значение корреляции:',
                  corrCr))
        else:
            StatTab[-1].append(('\nКоэффициент корреляции не может быть признан статичстически значимым:',
                  '\nКоэффицент корреляции:',corr[0][1],'\t','Критическое значение корреляции:',
                  corrCr))
        
        YSR = sum(Y)/self.N
        StatTab[-1].append(("Y среднее:",YSR))
        R_2 = 1 - (sum(Y-YR)**2/(sum(Y-YSR)**2))
        StatTab[-1].append(("R_2:",R_2))
        R_2adj = 1 - ((1-R_2)*(self.N-1/(self.N-k)))
        StatTab[-1].append(("R_2adj:",R_2adj))
        SD = (sum(Y-YR)**2)/(self.N-k) # Среднеквад ошибка
        StatTab[-1].append(("Среднеквадратичная ошибка:",SD))
        MAE = 1/self.N*sum(abs(Y-YR))
        MAPE = 1/self.N*sum(abs((Y-YR)/Y))*100
        StatTab[-1].append(("MAE:",MAE))
        StatTab[-1].append(("MAPE:",MAPE))
        
        SDsq = np.sqrt(SD) # Кв. корень из среднеквад ошибки 
        StatTab[-1].append(("Квадратный корень среднеквадратичной ошибки:",SDsq))
        
        do = np.linalg.inv(np.matmul(xt,x))
        d1 = np.matmul(x,do)
        D = np.matmul(d1,xt)
        S = list(range(0,self.N,1))
        for i in range(len(S)):
            S[i] = t * np.sqrt(Dad * (1+D[i][i]))
        StatTab[-1].append(("S =",np.vstack(S)))
        yrmax = YR + S[i]
        yrmin = YR - S[i]
        
        plt.plot(self.daTab[1], Y)
        plt.title("График зависимости Y от Х1.(Pnp)")
        plt.ylabel("Y - потребление электроэенергии.")
        plt.xlabel("Х1 - сутки.")
        plt.plot(YR, color = 'green', label = 'YR')
        plt.plot(yrmax, color = 'red', label = yrmax,linestyle = '--')
        plt.plot(yrmin, color = 'red', label = yrmin,linestyle = '--')
        plt.grid()
        plt.show()
        print(' ')
    
    def savestat(self):
        file = open("C:/Users/snake/Desktop/Модель6.txt","w")
        l=len(StatTab[-1])
        file.write("\n")
        for i in range(l):
            file.write(str(StatTab[-1][i]))
            file.write("\n")
        file.close()

data = Mnk("dataset.xlsx","Лист1",N=29000)
data.badan()
data.graf()
data.prompt("x0,x1,x2,x3,x5")
data.savestat()
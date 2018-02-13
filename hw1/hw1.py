import sys
import ssl
import urllib.request
import matplotlib.pyplot as plt

def get_data():
    url = 'https://ceiba.ntu.edu.tw/course/481ea4/hw1_data.csv'
    ssl._create_default_https_context = ssl._create_unverified_context
    data = urllib.request.urlopen(url)
    csvfile = str(data.read(), 'utf-8').split('\n')[1:-1]
    edu_list = []
    income_list = []
    working_list = []
    for line in csvfile:
        xx = []
        data = line.split(',')
        if data[1] == '':
            if data[0] == "Education level":
                key = edu_list
            elif data[0] == "Average monthly income":
                key = income_list
            elif data[0] == "Working environment":
                key = working_list
            key.append(data)
        else:
            xx.append(data[0])
            for i in range(1, len(data)):
                xx.append(float(data[i]))
            key.append(xx)
    print(edu_list)
    print(income_list)
    print(working_list)
    return edu_list, income_list, working_list

def line_chart(table, title):
    
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    
    plt.title(title)
    plt.xlabel(table[0][0])
    plt.ylabel("Smoking Percentage (%)")
    male = [ table[index][2] for index in range(1, len(table)) ]
    female = [ table[index][4] for index in range(1, len(table)) ]
    ax.set_ylim([0, max(male)+10])
    
    total = []
    for index in range(1, len(table)):
        tmp = table[index][1]*table[index][2] + table[index][3]*table[index][4]
        tmp = tmp/(table[index][1]+table[index][3])
        total.append(round(tmp,1))
    
    L = range(1, len(table))
        
    line_male, = plt.plot(L, male, marker='s', color='r', label='Male')
    line_female, = plt.plot(L, female, marker='o', color='g', label='Female')
    line_total, = plt.plot(L, total, marker='^', color='b', label='Total')
    
    for i, num in enumerate(male):
        ax.annotate(num, (L[i]+0.1, male[i]))
    for i, num in enumerate(female):
        ax.annotate(num, (L[i]+0.1, female[i]))
    for i, num in enumerate(total):
        ax.annotate(num, (L[i]+0.1, total[i]))
        
    plt.xticks(range(0, len(table)+1), [""]+[table[index][0] for index in range(1, len(table))], fontsize=12)
    plt.legend(handles=[line_male, line_female, line_total], numpoints=1, prop={'size': 15})
    plt.show()

def bar_chart(table, title):
    
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    
    plt.title(title)
    plt.xlabel(table[0][0])
    plt.ylabel("Smoking Percentage (%)")
    male = [ table[index][2] for index in range(1, len(table)) ]
    female = [ table[index][4] for index in range(1, len(table)) ]
    ax.set_ylim([0, max(male)+10])
    
    total = []
    for index in range(1, len(table)):
        tmp = table[index][1]*table[index][2] + table[index][3]*table[index][4]
        tmp = tmp/(table[index][1]+table[index][3])
        total.append(round(tmp,1))
    
    move = 0.2
    L = range(1, len(table))
        
    bar_male = ax.bar([num - 1*move for num in L], male, move, color='mediumslateblue', label='Male')
    bar_female = ax.bar([num - 0*move for num in L], female, move, color='salmon', label='Female')
    bar_total = ax.bar([num + 1*move for num in L], total, move, color='yellow', label='Total')
    
    for i, num in enumerate(male):
        ax.annotate(num, (L[i]-1.3*move, male[i]+0.5))
    for i, num in enumerate(female):
        ax.annotate(num, (L[i]-0.3*move, female[i]+0.5))
    for i, num in enumerate(total):
        ax.annotate(num, (L[i]+0.7*move, total[i]+0.5))
        
    plt.xticks(range(0, len(table)+1), [""]+[table[index][0] for index in range(1, len(table))], fontsize=12)
    plt.legend(handles=[bar_male, bar_female, bar_total], numpoints=1, prop={'size': 15})
    plt.show()

def pie_chart(table, title):
    fig, ax = plt.subplots(figsize=(12,8))
    plt.title(title)
    
    total = []
    p = 0
    for index in range(1, len(table)):
        p += table[index][1]*table[index][2] + table[index][3]*table[index][4]
        
    for index in range(1, len(table)):
        tmp = table[index][1]*table[index][2] + table[index][3]*table[index][4]
        total.append(round(100*tmp/p, 1))
        
    labels = [ table[index][0] for index in range(1, len(table))]
    colors = ['yellowgreen', 'yellow', 'lightskyblue', 'orchid', 'lightcoral']
    ax.pie(total, labels=labels, autopct='%1.1f%%', colors=colors)
    ax.axis('equal')
    plt.show()   

if __name__ == '__main__':
    edu_list, income_list, working_list = get_data()
    for index in range(1, len(sys.argv)):
        if sys.argv[index][1] == 'E':
            if sys.argv[index][2] == 'l':
                line_chart(edu_list, "Smoking percentage vs Educational level")
            elif sys.argv[index][2] == 'b':
                bar_chart(edu_list, "Smoking percentage vs Educational level")
            else:
                pie_chart(edu_list, "Proportion of different education level in smoking population")
        elif sys.argv[index][1] == 'A':
            if sys.argv[index][2] == 'l':
                line_chart(income_list, "Smoking percentage vs Avergae monthly income")
            elif sys.argv[index][2] == 'b':
                bar_chart(income_list, "Smoking percentage vs Avergae monthly income")
            else:
                pie_chart(income_list, "Proportion of different average monthly income in smoking population")
        else:
            if sys.argv[index][2] == 'l':
                line_chart(working_list, "Smoking percentage vs Working environment")
            elif sys.argv[index][2] == 'b':
                bar_chart(working_list, "Smoking percentage vs Working environment")
            else:
                pie_chart(working_list, "Proportion of different working environment in smoking population")

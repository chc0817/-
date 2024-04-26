import pandas as pd
import matplotlib.pyplot as plt  
 
pdata = pd.read_csv("no_load_rr.csv", header=0)
p_grade='5'

  
# 使用条形图可视化信誉评级  
plt.figure(figsize=(10, 8))  
plt.scatter(pdata.index, pdata[p_grade], marker='o', s=100)       
plt.xticks(rotation=90) 
tick_positions = range(0, len(pdata), 10) 
tick_labels = [str(i) if i % 10 == 0 else '' for i in pdata.index] 
plt.xticks(tick_positions, [tick_labels[pos] for pos in tick_positions])   
plt.ylabel('Enterprise rating')  
plt.yticks(range(4),['Bronze','Gold ','Iron ','Silver'] )
plt.ylim(-0.5, 3.5)
plt.title('Corporate rating distribution')  


# 统计各个等级的个数  
grade_counts = pdata[p_grade].value_counts().sort_index()  
  
# 打印各个等级的个数  
print(grade_counts)  
  
# 可视化各个等级的个数 
plt.figure(figsize=(10, 6))  
bars = grade_counts.plot(kind='bar') 
plt.title('Counts for each level')  
plt.xlabel('Enterprise rating')
plt.xticks(range(4),['Bronze','Gold ','Iron ','Silver'] )  
plt.ylabel('Count')
plt.ylim(0,200)  
plt.xticks(rotation=0) 
for p in bars.patches:  
    count = str(int(p.get_height()))  # 获取条形的高度（即企业数量）并转换为整数和字符串  
    bars.annotate(count, (p.get_x() + p.get_width() / 2, p.get_height() + 10), ha='center', va='bottom')  
   
plt.show()
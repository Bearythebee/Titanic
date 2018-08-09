import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns',50)

train = pd.read_csv(r"C:\Users\Admin\Desktop\Kaggle datasets\Titanic\train.csv")

print(train.columns)

### Plotting survival rate against sex#

sns.barplot(x="Sex", y="Survived", data=train)
#plt.show()

#fem_survived = train.query('Survived == 1 and Sex == "female" ').count()
#num_females = train[train['Sex']== "female"].count()
#female_survival_rate =  fem_survived/num_females

female_survival_rate = train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100

#male_survived = train.query('Survived == 1 and Sex == "male"').count()
#num_males = train[train['Sex']=="male"].count()
#male_survival_rate =  male_survived/num_males

male_survival_rate = train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100
print("----------")
print("Sex")
print("----------")
print("Female survival rate = " + str(female_survival_rate)+"%")
print("Male survival rate = " + str(male_survival_rate)+ "%")
print("----------")

# Male survival rate = 18.890814558058924%
# Female survival rate = 74.20382165605095%


### Plotting survival rate against ticket class(Pclass)

sns.barplot(x="Pclass", y="Survived", data=train)
#plt.show()

Pclass1 = train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100
Pclass2 = train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100
Pclass3 = train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100

print("Ticket class")
print("----------")
print("Pclass1 survival rate = " + str(Pclass1)+ "%")
print("Pclass2 survival rate = " + str(Pclass2)+ "%")
print("Pclass3 survival rate = " + str(Pclass3)+ "%")
print("----------")

# Ticket class 1 survival rate = 62.96296296296296%
# Ticket class 2 survival rate = 47.28260869565217%
# Ticket class 3 survival rate = 24.236252545824847%




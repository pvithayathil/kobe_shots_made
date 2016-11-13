import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier


# Thanks for https://www.kaggle.com/arthurlu/titanic/exploratory-tutorial-titanic/notebook
# For Helping with the Analysis

# Load Data
data = pd.read_csv('kobedata.csv')
# Setup Data
data['remaining_time'] = data['minutes_remaining'] * 60 + data['seconds_remaining']
data['Home'] = pd.Series(1, index=data.index)
_ = data.set_value(data.matchup.str.contains('@')==True, 'Home', 0)

train = data[(~data.shot_made_flag.isnull())]
test = data[(data.shot_made_flag.isnull())]


print "-------Basic Statistics-------"
print ("Dimension: {}".format(data.shape))

print "-------Train Statistics-------"
print ("Dimension: {}".format(train.shape))
print train.describe(include=['number'])
print train.describe(include=['object', 'category'])

print "-------Test Statistics-------"
print ("Dimension: {}".format(test.shape))



sns.set_style('white')

ncount = len(train)

# Graph on Made Shots for Career
plt.figure(figsize=(12,8))
plt.title('Kobe Shots Missed or Made')
plt.xlabel('Miss or Make')
ax = sns.countplot(x="shot_made_flag", data=train, order=[0,1],palette='Set1')
# Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Frequency [%]')
ax.set_xticklabels(['Missed','Made'])

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
# Use a LinearLocator to ensure the correct number of ticks
ax.yaxis.set_major_locator(ticker.LinearLocator(11))
# Fix the frequency range to 0-100
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)
# And use a MultipleLocator to ensure a tick spacing of 10
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
plt.savefig('kobe_career_totals.svg')

#########
# Graph on Made Shots for Career

#loc_x,loc_y
g = sns.JointGrid("loc_x", "loc_y", train)
colors = ['#d6604d','#4393c3']
colors = [(0.69607843644917011, 0.30000000260770321, 0.30392157100141015),(0.34198770274718604, 0.48141100064796549, 0.59391389223290425)]
for day, day_tips in train.groupby("shot_made_flag"):
    #print day
    sns.kdeplot(day_tips["loc_x"], ax=g.ax_marg_x, legend=False, color = colors[int(day)])
    sns.kdeplot(day_tips["loc_y"], ax=g.ax_marg_y, vertical=True, legend=False, color = colors[int(day)])
    g.ax_joint.plot(day_tips["loc_x"], day_tips["loc_y"], "o",ms=5, alpha = .1,color = colors[int(day)])
plt.legend(["Missed","Made"])
plt.savefig('kobe_career_loc_xy.svg')
#loc_x,shot dist
g = sns.JointGrid("loc_x", "shot_distance", train)

for day, day_tips in train.groupby("shot_made_flag"):
    #print day
    sns.kdeplot(day_tips["loc_x"], ax=g.ax_marg_x, legend=False, color = colors[int(day)])
    sns.kdeplot(day_tips["shot_distance"], ax=g.ax_marg_y, vertical=True, legend=False, color = colors[int(day)])
    g.ax_joint.plot(day_tips["loc_x"], day_tips["shot_distance"], "o",ms=5, alpha = .1,color = colors[int(day)])
plt.legend(["Missed","Made"])
plt.savefig('kobe_career_loc_x_shotdistance.svg')
#loc_x,loc_y
g = sns.JointGrid("shot_distance", "loc_y", train)
for day, day_tips in train.groupby("shot_made_flag"):
    #print day
    sns.kdeplot(day_tips["shot_distance"], ax=g.ax_marg_x, legend=False, color = colors[int(day)])
    sns.kdeplot(day_tips["loc_y"], ax=g.ax_marg_y, vertical=True, legend=False, color = colors[int(day)])
    g.ax_joint.plot(day_tips["shot_distance"], day_tips["loc_y"], "o",ms=5, alpha = .1,color = colors[int(day)])
plt.legend(["Missed","Made"])
plt.savefig('kobe_career_shotdistance_loc_y.svg')
with sns.axes_style("white"):
    g = sns.FacetGrid(train, row="Home", col="period", margin_titles=False, size=2.5,sharex=False,sharey=False)
g.map(sns.countplot, "shot_made_flag",palette="Set1", edgecolor="white", lw=.5,order=[0,1]);
for axlist in g.axes:
    for ax in axlist:
        ncount = 0
        for p in ax.patches:
            ncount += p.get_height()
        # Make twin axis
        ax2=ax.twinx()
        # Switch so count axis is on right, frequency on left
        ax2.yaxis.tick_left()
        ax.yaxis.tick_right()
        # Also switch the labels over
        ax.yaxis.set_label_position('right')
        ax2.yaxis.set_label_position('left')
        #ax2.set_ylabel('Frequency [%]')
        ax.set_xticklabels(['Missed','Made'])
        ax.set_yticks([])

        # Use a LinearLocator to ensure the correct number of ticks
        ax.yaxis.set_major_locator(ticker.LinearLocator(11))
        # Fix the frequency range to 0-100
        ax2.set_ylim(0,100)
        ax.set_ylim(0,ncount)
        # And use a MultipleLocator to ensure a tick spacing of 10
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

#g.set_axis_labels("Total bill (US Dollars)", "Tip");
#g.set(xticks=[10, 30, 50], yticks=[2, 6, 10]);
g.fig.subplots_adjust(wspace=.25, hspace=.5)
plt.savefig('kobe_career_homeperiod.svg')
###

f, axarr = plt.subplots(3, 2, figsize=(15, 15))

sns.boxplot(x='shot_made_flag', y='lat', data=data, ax=axarr[0,0],palette="Set1",vert=False)
sns.boxplot(x='shot_made_flag', y='lon', data=data, ax=axarr[0, 1],palette="Set1",vert = False)
sns.boxplot(x='shot_made_flag', y='loc_y', data=data, ax=axarr[1, 0],palette="Set1",vert=False)
sns.boxplot(x='shot_made_flag', y='loc_x', data=data, ax=axarr[1, 1],palette="Set1",vert=False)
sns.boxplot(x='shot_made_flag', y='remaining_time',  data=data, ax=axarr[2, 0],palette="Set1",vert=False)
sns.boxplot(x='shot_made_flag', y='shot_distance', data=data, ax=axarr[2, 1],palette="Set1",vert=False)


axarr[0, 0].set_title('Latitude')
axarr[0, 1].set_title('Longitude')
axarr[1, 0].set_title('Loc y')
axarr[1, 1].set_title('Loc x')
axarr[2, 0].set_title('Time Remaining (seconds)')
axarr[2, 1].set_title('Shot Distance')

plt.tight_layout()
plt.savefig('kobe_career_boxplots.svg')
###
season_order = ['1996-97','1997-98','1998-99','1999-00','2000-01','2001-02','2002-03','2003-04','2004-05','2005-06','2006-07','2007-08','2008-09','2009-10','2010-11','2011-12','2012-13','2013-14','2014-15','2015-16']
season_label = ['96','97','98','99','00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15']
f, axarr = plt.subplots(9, figsize=(15, 25))

sns.countplot(x="season", hue="shot_made_flag", data=data, ax=axarr[0],order=season_order,palette="Set1")
sns.countplot(x="period",hue="shot_made_flag", data=data, ax=axarr[1],palette="Set1")
sns.countplot(x="playoffs", hue="shot_made_flag", data=data, ax=axarr[2],palette="Set1")
sns.countplot(x="Home", hue="shot_made_flag", data=data, ax=axarr[3],palette="Set1")
sns.countplot(x="combined_shot_type", hue="shot_made_flag", data=data, ax=axarr[4],palette="Set1")
sns.countplot(x="shot_type", hue="shot_made_flag", data=data, ax=axarr[5],palette="Set1")
sns.countplot(x="shot_zone_area", hue="shot_made_flag", data=data, ax=axarr[6],palette="Set1")
sns.countplot(x="shot_zone_basic", hue="shot_made_flag", data=data, ax=axarr[7],palette="Set1")
sns.countplot(x="shot_zone_range", hue="shot_made_flag", data=data, ax=axarr[8],palette="Set1")


#axarr[0].set_title('Season')
#axarr[1].set_title('Period')
#axarr[2].set_title('Playoffs')
#axarr[3].set_title('Home')
#axarr[4].set_title('Combined shot type')
#axarr[5].set_title('Shot Type')
#axarr[6].set_title('Shot Zone Area')
#axarr[7].set_title('Shot Zone Basic')
#axarr[8].set_title('Shot Zone Range')

plt.tight_layout()
plt.savefig('kobe_career_barplots.svg')
plt.show()


#plt.show()

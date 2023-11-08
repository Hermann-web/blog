# imports


```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from statsmodels.api import Logit

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


```


```python
import os
def check_path(path):
  path = os.path.normpath(os.path.abspath(path)).replace('\\','/')
  _split = path.split("/")
  for i in range(1,len(_split)):
    _filename,_path_to_file,_path = _split[i],"/".join(_split[:i+1]),"/".join(_split[:i])
    assert os.path.exists(_path_to_file),f"{_filename} doest not exist in {_path}"

#aller dans le dossier de travail
from google.colab import drive
import os
drive.mount('/gdrive')

drive_path = '/gdrive/My Drive/'
shared_path = drive_path + 'Colab Notebooks/3A/Projet_final/'
check_path(shared_path) #check if dir exists
print("test passed")
os.chdir(shared_path) #change dir
```

    Drive already mounted at /gdrive; to attempt to forcibly remount, call drive.mount("/gdrive", force_remount=True).
    test passed



```python
df = pd.read_csv("Projet_ML.csv")
```


```python
df.bidder_id.nunique()
```




    87




```python
TARGET_COL = "outcome"
REMOVE_EVIDENT_MERCHANDISE = False
FILE_VERSION = "v7"
PROD_INSTEAD_OF_SUM =  True
ADD_LEN_TO_GROUPBY = True
#prod+nolen < prod+len
```

# Utils

## visualization


```python
def get_weight(arr: pd.Series):
  arr = pd.Series(arr)
  dd = {elt:1/len(arr[arr==elt]) for elt in arr.unique()}
  total = sum(dd.values())
  return arr.apply(lambda x: dd[x]/total)

get_weight(pd.Series([0,0,1,2,3]))
```




    0    0.142857
    1    0.142857
    2    0.285714
    3    0.285714
    4    0.285714
    dtype: float64




```python
def add_labels_to_histplot(ax,title):
  ax.set(title=title)# label each bar in histogram
  for p in ax.patches:
    height = p.get_height() # get the height of each bar
    # adding text to each bar
    ax.text(x = p.get_x()+(p.get_width()/2), # x-coordinate position of data label, padded to be in the middle of the bar
    y = height+0.2, # y-coordinate position of data label, padded 0.2 above bar
    s = "{:.0f}".format(height), # data label, formatted to ignore decimals
    ha = "center") # sets horizontal alignment (ha) to center
    plt.xticks(rotation=45)
```


```python
%matplotlib inline
#plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
font = {'family':'Helvetica, Ariel',
        'weight':'normal',
        'size':12}
plt.rc('font', **font)
sns.set(rc={
    #"figure.dpi": 300,
    'savefig.dpi': 300})
sns.set_context('notebook')
sns.set_style("ticks")
FIG_FONT = dict(family="Helvetica, Ariel", weight="bold")#, color="#7f7f7f")
#sns.set_palette('Spectral')

#plot function
def univariate_double_plot(df, x=None, xlabel=None, explode=None,ylabel=None,palette=None,order=True,hue=None):
    sns.set_palette(palette)
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    #count bar
    if order == True:
        feature_data = df[x].value_counts(ascending=True)
        sns.countplot(data=df, x=x, ax=ax[0], order=feature_data.index,hue = ylabel)

    else:
        feature_data = df[x].value_counts(sort=False).sort_index()
        sns.countplot(data=df, x=x, ax=ax[0],order=feature_data.index,hue = ylabel)

    #pie chart
    patches, texts, autotexts = ax[1].pie(feature_data.values,labels=feature_data.index,
                                      autopct='%.0f%%',textprops={'size': 20})
    for i in range(len(autotexts)):
        autotexts[i].set_color('white')

    #reduce non-data ink
    sns.despine(bottom=True, left=True)
    ax[0].set_xlabel(xlabel=xlabel, size=12, fontdict=FIG_FONT)
    #ax[0].set_xticklabels(feature_data.index,rotation=20,fontsize = 'small')
    ax[0].set_ylabel(ylabel=ylabel)
    ax[1].set_ylabel(ylabel=ylabel)
    fig.text(0.5, 1, f'{xlabel} Distribution', size=16, fontdict=FIG_FONT, ha="center", va="center")
    plt.show()

def univariate_single_plot(df, x=None, xlabel=None, rotation=None,ylabel=None,palette=None):
    sns.set_palette(palette)
    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    feature_data = df[x].value_counts(ascending=True)
    sns.countplot(data=df, x=x, order=df[x].value_counts(ascending=True).index)
    sns.despine(bottom=True, left=True)
    plt.xlabel(xlabel=xlabel, size=14, fontdict=FIG_FONT)
    plt.xticks(rotation=rotation)
    plt.ylabel(ylabel=ylabel)
#     if bar_label:
#         ax.bar_label(ax.containers[0], label_type='edge', size=15, padding=1, fontname="Helvetica, Ariel",
#                         color="k")
    for i in range(len(feature_data.index)):
        ax.text(i,feature_data.iloc[i]*0.9,feature_data.iloc[i],ha='center',
                   fontsize=20,color='white')
    plt.title(label=f'{xlabel} Distribution', size=18, fontdict=FIG_FONT)
    plt.show()

def plot_dist_fill(x, y, ax, gamma=True):
  fit = stats.gamma if gamma else None
  kde = False if gamma else True
  sns.distplot(x, fit_kws={"color":"red"}, kde=False,
        fit=fit, hist=None, label="label 1", ax=ax);
  sns.distplot(y, fit_kws={"color":"blue"}, kde=False,
          fit=fit, hist=None, label="label 2", ax=ax);

  # Get the two lines from the axes to generate shading
  l1 = ax.lines[0]
  l2 = ax.lines[1]

  # Get the xy data from the lines so that we can shade
  x1 = l1.get_xydata()[:,0]
  y1 = l1.get_xydata()[:,1]
  x2 = l2.get_xydata()[:,0]
  y2 = l2.get_xydata()[:,1]
  ax.fill_between(x1,y1, color="red", alpha=0.3)
  ax.fill_between(x2,y2, color="blue", alpha=0.3)

def univariate_numerical_plot(df, x=None, xlabel=None,ylabel=None,palette=None,bins=20,target_as_hue=None, gamma=True, use_weights=False):
    assert isinstance(target_as_hue, str)
    palette = ["b","r"]
    sns.set_palette(palette)
    fig, ax = plt.subplots(1, 5, figsize=(20, 8))
    #hist
    if use_weights: sns.histplot(bins=bins,data=df, x=x, kde=True, ax=ax[0], weights=get_weight(target_as_hue), hue=target_as_hue)
    else: sns.histplot(bins=bins,data=df, x=x, kde=True, ax=ax[0], hue=target_as_hue)
    #box
    sns.boxplot(data=df, y=x, ax=ax[1], hue=target_as_hue)
    #prob
    plt.sca(ax[2])
    stats.probplot(df[x], dist = "norm", plot = plt)
    #plt.ylabel('Variable quantiles')

    #sns.displot(data=df, x=x, hue=target_as_hue, kind="kde", ax=ax[3])
    #sns.displot(data=df, x=x, hue=target_as_hue, kind="kde", color="g", ax=ax[3])
    try:
      plot_dist_fill(df[df[target_as_hue]==True][x], df[df[target_as_hue]==False][x], ax=ax[3], gamma=gamma)
    except: pass
    if use_weights: sns.displot(data=df, x=x, hue=target_as_hue, kind="kde", color="g", ax=ax[4], weights = get_weight(target_as_hue) )
    else: sns.displot(data=df, x=x, hue=target_as_hue, kind="kde", color="g", ax=ax[4])

    sns.despine(bottom=True, left=True)
    ax[0].set_xlabel(xlabel=xlabel, size=12, fontdict=FIG_FONT)
    ax[0].set_title(f'The histogram of {x}')
    ax[1].set_xlabel(xlabel=ylabel, size=12, fontdict=FIG_FONT)
    ax[0].set_ylabel(ylabel=ylabel,size=12, fontdict=FIG_FONT)
    ax[1].set_ylabel(ylabel=xlabel, size=12, fontdict=FIG_FONT)
    ax[1].set_title(f'The boxplot of {x}')
    fig.text(0.5, 1, f'{xlabel} Distribution', size=16, fontdict=FIG_FONT, ha="center", va="center")
    plt.show()

```

## testing proportions


```python
from scipy.stats import norm

# p_value => fct de rpartition
def get_p_value_from_tail(prob, tail, debug=False):
    '''
    get p value based on cdf and tail
    If tail=Tails.middle, the distribution is assumed symmetric because we double F(Z)
    if tail
        - right: return P(N > Z) = 1- F(Z) =  1 - prob
        - left: return P(N < Z) = F(Z) = prob
        - middle: return P(N < -|Z|) + P(N > |Z|) => return  2*P(N > |Z|)
    '''
    if tail == "right":  # Z est à droite. On compte P(N>Z)
        return 1 - prob
    elif tail == "left":  # Z est à gauche. On compte P(N<Z)
        return prob
    elif tail == "middle":  # We take Z>0 so at the right => then double to take into account the left part #double because "normal" and "student" are both symetric!!!!
        '''En supposant la distribution symetrique'''
        if debug:
            print(f"p_val = 2*(1 - prob) = 2*(1 - {prob}) = {2*(1 - prob)}")
        return 2 * (1 - prob)
    else:
        raise Exception("tail not correct. get", tail)

def get_p_value_z_test(Z: float, tail: str, debug=False):
    '''
    get p value based on normal distribution N(0, 1)
    if tail
        - right: return P(N > Z)
        - left: return P(N < Z)
        - middle: return P(N < -|Z|) + P(N > |Z|) => return  2*P(N > |Z|)
    '''
    if tail == "middle":
        Z = abs(Z)
    prob = norm.cdf(Z)
    return get_p_value_from_tail(prob, tail, debug)


def proportion_comparison_test(p1, p2, n1, n2, debug=False):
    """
    Hypotheses
    - H0: p0 = p1
    - H1: p0 !=p1
    """
    alpha = 0.05
    evcpp = True
    tail = "middle"
    # cdt
    p1 = float(p1)
    p2 = float(p2)
    n2 = int(n2)
    n1 = int(n1)

    # parameter
    p_hat = abs(p1 - p2)  # estimator
    p0 = 0

    # calculate the Z-statistic:
    # standard error of the estimate
    # the null standard error #la proportion suit une loi de bernouli => on passe à plusieurs samples
    if evcpp:
        # estimation de sqrt(p_hat*(1-p_hat)*(1/n1 + 1/n2))
        # Estimate of the combined population proportion
        phat2 = (p1 * n1 + p2 * n2) / (n1 + n2)
        # Estimate of the variance of the combined population proportion
        va = phat2 * (1 - phat2)
        # Estimate of the standard error of the combined population proportion
        std_stat_eval = np.sqrt(va * (1 / n1 + 1 / n2))
        if debug: print(f"phat2={phat2} = va={va}")
    else:
        std_stat_eval = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))

        # compute Z corresponding to normal distribtion
    # "assume" that th eproportion folow a normal for many samples
    Z = (p_hat - p0) / std_stat_eval
    if debug: print(f"Z = ({p_hat - p0})/{std_stat_eval} = ", Z)
    # Z = abs(Z) #to use tail=right it is now a H1:p>p0 problem

    # compute p_value
    p_value = get_p_value_z_test(Z, tail=tail, debug=debug)

    # rejection (we want to be far away from the mean to reject the null)
    # use alpha or alpha/2 ??
    reject_null = True if p_value < alpha else False
    return {"p_hat":p_hat, "std_stat_eval":std_stat_eval, "Z":Z, "p_value":p_value, "reject_null":reject_null,
            "message": "significant difference" if reject_null else "no enough evidence to support that p0!=p1"}


proportion_comparison_test(0.5, 0.6, 1000, 500)
```




    {'p_hat': 0.09999999999999998,
     'std_stat_eval': 0.02732520204255893,
     'Z': 3.6596252735569985,
     'p_value': 0.0002525843403027306,
     'reject_null': True,
     'message': 'significant difference'}




```python
proportion_comparison_test(0.710, 0.100, 146, 16)
```




    {'p_hat': 0.61,
     'std_stat_eval': 0.12562683605190134,
     'Z': 4.855650426060123,
     'p_value': 1.1999222317982117e-06,
     'reject_null': True,
     'message': 'significant difference'}



## transformations


```python
from sklearn.preprocessing import LabelBinarizer

def one_hot_encoder(data, col_name):
  data = data.reset_index()
  jobs_encoder = LabelBinarizer()
  jobs_encoder.fit(data[col_name])
  transformed = jobs_encoder.transform(data[col_name])
  new_cols = [col_name+str(i) for i in range(len(data[col_name].unique()))]
  ohe_df = pd.DataFrame(transformed, columns=new_cols)
  data = pd.concat([data, ohe_df], axis=1).drop([col_name], axis=1)
  if "level_0" in data.columns: data.drop("level_0", axis=1, inplace=True)
  if "index" in data.columns: data.drop("index", axis=1, inplace=True)
  return data, new_cols

def one_hot_encoder_v2(data, col_name):
  data = data.copy()
  new_cols = []
  for i, elt in enumerate(data[col_name].unique()):
    new_cols.append(f"{col_name}_{i+1}")
    data[new_cols[-1]] = data[col_name].apply(lambda x: int(x==elt))
  del data[col_name]
  return data, new_cols

df_, cols = one_hot_encoder_v2(pd.DataFrame({"merchandise":["a","b","a","a"],"pp":[1, 2, 3, 4]}), "merchandise")

df_
```





  <div id="df-4848f33a-fd10-46dd-b4fe-7b0067d3f8ca">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pp</th>
      <th>merchandise_1</th>
      <th>merchandise_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4848f33a-fd10-46dd-b4fe-7b0067d3f8ca')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-4848f33a-fd10-46dd-b4fe-7b0067d3f8ca button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4848f33a-fd10-46dd-b4fe-7b0067d3f8ca');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## models


```python
PRINT_BEST_TRESH = False #see if i could have done better with

## feature selection
USE_RFE = True #feature selection
NB_OF_BEST_FEATURES = 4 if ADD_LEN_TO_GROUPBY else 5

USE_BEST_THRESH = True #utile pour les algo qui predisent des probas


CLASS_NAMES = ["Humain", "Robot"]
```


```python
from sklearn import tree
import pydotplus
import matplotlib.image as pltimg

def show_decision_tree(model, features, class_names=None):
  data = tree.export_graphviz(model, out_file=None, feature_names=features, class_names=CLASS_NAMES)
  graph = pydotplus.graph_from_dot_data(data)
  graph.write_png('mydecisiontree.png')

  img=pltimg.imread('mydecisiontree.png')
  imgplot = plt.imshow(img)
  plt.show()

def show_random_forest(model, features, class_names=None):
  fn=features
  cn=list(map(str, model.classes_)) #CLASS_NAMES
  fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
  for index in range(0, 5):
      tree.plot_tree(model.estimators_[index],
                    feature_names = fn,
                    class_names=CLASS_NAMES,
                    filled = True,
                    ax = axes[index]);

      axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
  fig.savefig('rf_5trees.png')
```


```python
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix,r2_score,accuracy_score,roc_auc_score,f1_score,precision_score,recall_score
from sklearn.feature_selection import RFE





def return_dict_scores(y, y_pred, pred_proba):
  return {"acc":round(accuracy_score(y,y_pred),3), "f1":round(f1_score(y,y_pred),3),
          "pres":round(precision_score(y,y_pred),3), "rec":round(recall_score(y, y_pred),3),
          "roc":round(roc_auc_score(y, pred_proba),3)}

def find_best_thresh(y_true,y_pred_proba):
    best_thresh = None
    best_score = 0
    for thresh in np.arange(0.1, 0.501, 0.01):
        y_pred = np.array(y_pred_proba)>thresh
        score = f1_score(y_true, y_pred)
        if score > best_score:
            best_thresh = thresh
            best_score = score
    return best_score, best_thresh


def bestThressholdForF1(y_true,y_pred_proba):
    best_score, best_thresh = find_best_thresh(y_true,y_pred_proba)
    y_pred = np.array(y_pred_proba)>best_thresh
    return best_score , best_thresh, return_dict_scores(y_true, y_pred, y_pred_proba)


def print_metrics(y, y_pred,pred_proba):
  print("- confusion_matrix\n",confusion_matrix(y, y_pred))
  print(f"- accuracy = {100*accuracy_score(y, y_pred):.2f}%") #better ->1 ##accuracy = nb_sucess/nb_sample
  print(f"- f1 = {100*f1_score(y, y_pred):.2f}%") #better ->1 ##f1 = 2 * (precision * recall) / (precision + recall)
  print(f"- roc(area under the curve) = {100*roc_auc_score(y_pred, pred_proba):.2f}%") #better ->1 ##area under ROC and AUC
  print(f"- precision = {100*precision_score(y, y_pred):.2f}%") #better->1 ##precision = tp / (tp + fp) where (tp=true_positive; fp:false_positive)
  print(f"- recall = {100*recall_score(y, y_pred):.2f}%") #better->1 ##precision = tp / (tp + fn) where (tp=true_positive; fn:false_negative)
  if PRINT_BEST_TRESH: print(f"- bestThressholdForF1 = (sc,tr,res) = {bestThressholdForF1(y,pred_proba)}")
  return return_dict_scores(y, y_pred, pred_proba)

def show_results(y_train, y_train_pred, y_train_proba,y_test, y_test_pred, y_test_proba ):
  print("\n>>>> metriques sur la base de données d'entrainement")
  train_res = print_metrics(y_train,y_train_pred,y_train_proba)
  print("\n>>>> metriques sur la base de données de test")
  test_res = print_metrics(y_test, y_test_pred,y_test_proba)
  return train_res, test_res


def knn_classifier(X_train, y_train, X_test, y_test):
  parameters = {"n_neighbors":[1,2,5,10,15,20]}
  clf = GridSearchCV(KNeighborsClassifier(), parameters).fit(X_train, y_train)
  print(clf.best_params_)
  #clf = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
  y_train_proba = clf.predict_proba(X_train)[:, 1] #[proba({label=1}/row_data) for row_data in X_train]
  print("y_train_proba:",len(set(y_train_proba)))
  y_test_proba = clf.predict_proba(X_test)[:, 1]
  if USE_BEST_THRESH:
    best_sc, best_thresh = find_best_thresh(y_train,y_train_proba)
    y_train_pred = y_train_proba > best_thresh
    y_test_pred = y_test_proba > best_thresh
  else:
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
  return y_test_proba, y_train_proba, y_train_pred, y_test_pred, len(clf.feature_names_in_)

def lda_classifier(X_train, y_train, X_test, y_test):
  '''discriinant: estime les params de p(x/yk) qq k'''
  parameters = {"solver":["svd", "eigen", "lsqr"]}
  clf = GridSearchCV(LinearDiscriminantAnalysis(), parameters).fit(X_train, y_train)
  #clf = LinearDiscriminantAnalysis().fit(X_train, y_train)
  print(clf.best_params_)
  y_train_pred = clf.predict(X_train)
  y_test_pred = clf.predict(X_test)
  y_train_proba = clf.predict_proba(X_train)[:, 1] #[proba({label=1}/row_data) for row_data in X_train]
  print("y_train_proba:",len(set(y_train_proba)))
  y_test_proba = clf.predict_proba(X_test)[:, 1]
  return y_test_proba, y_train_proba, y_train_pred, y_test_pred, len(clf.feature_names_in_)


def logistic_regression_sklearn(X_train, y_train, X_test, y_test):
  parameters = {"max_iter":[5,100,1000,2000], "penalty":["l1", "l2", "elasticnet", "none"], "C":[0.01, 0.1, 0.5, 1], "solver":["lbfgs", "liblinear"]}
  clf = GridSearchCV(LogisticRegression(random_state=0, class_weight="balanced"), parameters).fit(X_train, y_train)
  print(clf.best_params_)
  if USE_RFE:
    clf = RFE(LogisticRegression(random_state=0, class_weight="balanced", **clf.best_params_), n_features_to_select=NB_OF_BEST_FEATURES).fit(X_train, y_train)
    nb_features = clf.n_features_
    print(f"ranking = {sorted(list(zip(clf.feature_names_in_,clf.ranking_)), key=lambda x:x[1])}")
  else: nb_features = len(clf.feature_names_in_)
  print(f"coeffs = {clf.estimator_.coef_}")
  #clf = LogisticRegression(random_state=0, max_iter=1000, class_weight="balanced").fit(X_train, y_train)
  #nb_features = len(clf.feature_names_in_)

  y_train_proba = clf.predict_proba(X_train)[:, 1] #[proba({label=1}/row_data) for row_data in X_train]
  print("y_train_proba:",len(set(y_train_proba)))
  y_test_proba = clf.predict_proba(X_test)[:, 1]
  if USE_BEST_THRESH:
    best_sc, best_thresh = find_best_thresh(y_train,y_train_proba)
    y_train_pred = y_train_proba > best_thresh
    y_test_pred = y_test_proba > best_thresh
  else:
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)


  return y_test_proba, y_train_proba, y_train_pred, y_test_pred, nb_features

def svc_classifier(X_train, y_train, X_test, y_test):
  parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.1, 1]} #make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
  #clf = GridSearchCV(SVC(probability=True, class_weight="balanced"), parameters).fit(X_train, y_train)
  #print(clf.best_params_)
  if USE_RFE:
    clf_svc = make_pipeline(StandardScaler(), RFE(SVC(gamma='scale',probability=True, C=1, kernel="linear", class_weight="balanced"),n_features_to_select=NB_OF_BEST_FEATURES)).fit(X_train, y_train)
    clf = clf_svc[1]
    nb_features = clf.n_features_
    print(f"ranking = {sorted(list(zip(clf_svc.feature_names_in_,clf.ranking_)), key=lambda x:x[1])}")
  else:
    clf_svc = make_pipeline(StandardScaler(), SVC(gamma='scale',probability=True, C=1, kernel="linear", class_weight="balanced")).fit(X_train, y_train)
    nb_features = len(clf_svc.feature_names_in_)

  y_train_proba = clf.predict_proba(X_train)[:, 1] #[proba({label=1}/row_data) for row_data in X_train]
  print("y_train_proba:",len(set(y_train_proba)))
  y_test_proba = clf.predict_proba(X_test)[:, 1]
  if USE_BEST_THRESH:
    best_sc, best_thresh = find_best_thresh(y_train,y_train_proba)
    y_train_pred = y_train_proba > best_thresh
    y_test_pred = y_test_proba > best_thresh
  else:
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
  return y_test_proba, y_train_proba, y_train_pred, y_test_pred, nb_features


def sdg_classifier(X_train, y_train, X_test, y_test):
  parameters = {"max_iter":[10,15,20,30,50], "average":[1,2,10,20], "loss":["hinge", "modified_huber"]}
  clf = GridSearchCV(SGDClassifier(class_weight="balanced"), parameters).fit(X_train, y_train)
  print(clf.best_params_)
  #loss = "hinge"
  #clf = SGDClassifier(loss=loss, max_iter=5, average=10).fit(X_train, y_train)
  y_train_pred = clf.predict(X_train)
  y_test_pred = clf.predict(X_test)
  try:
    y_train_proba = clf.predict_proba(X_train)[:, 1] #[proba({label=1}/row_data) for row_data in X_train]
    y_test_proba = clf.predict_proba(X_test)[:, 1]
  except:
    y_train_proba = y_train_pred
    y_test_proba = y_test_pred
  print("y_train_proba:",len(set(y_train_proba)))
  return y_test_proba, y_train_proba, y_train_pred, y_test_pred, len(clf.feature_names_in_)



def decision_tree_classifier(X_train, y_train, X_test, y_test):
  '''
    - il overfitte vite car en profonteur, il est plus precis(err_train=0)
    - random forest is a go-to
  '''
  parameters = {"max_depth":[5,15,25], "max_leaf_nodes":[5,15,25]}
  clf = GridSearchCV(DecisionTreeClassifier(class_weight="balanced"), parameters).fit(X_train, y_train)
  #clf = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=5).fit(X_train, y_train)
  print(clf.best_params_)
  if USE_RFE:
    clf = RFE(DecisionTreeClassifier(class_weight="balanced", **clf.best_params_), n_features_to_select=NB_OF_BEST_FEATURES).fit(X_train, y_train)
    nb_features = clf.n_features_
    _ = list(zip(clf.feature_names_in_,clf.ranking_))
    features = list(map(lambda x:x[0], list(filter(lambda x:x[1]==1, _))))
    print(f"ranking = {sorted(_, key=lambda x:x[1])}")
  else:
    nb_features = len(clf.feature_names_in_)
    print("used: ",clf.feature_names_in_)
  y_train_proba = clf.predict_proba(X_train)[:, 1] #[proba({label=1}/row_data) for row_data in X_train]
  print("y_train_proba:",len(set(y_train_proba)))
  y_test_proba = clf.predict_proba(X_test)[:, 1]
  if USE_BEST_THRESH:
    best_sc, best_thresh = find_best_thresh(y_train,y_train_proba)
    y_train_pred = y_train_proba > best_thresh
    y_test_pred = y_test_proba > best_thresh
  else:
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

  if nb_features<6:
    show_decision_tree(clf.estimator_, features)
  return y_test_proba, y_train_proba, y_train_pred, y_test_pred, nb_features



def random_forest_classifier(X_train, y_train, X_test, y_test):
  parameters = {"max_depth":[1,3,5], "max_leaf_nodes":[1,3,5], "n_estimators":[10,20], "max_features":["sqrt", "log2", None], "criterion":["gini", "entropy", "log_loss"]}
  clf = GridSearchCV(RandomForestClassifier(class_weight="balanced"), parameters).fit(X_train, y_train)
  print(clf.best_params_)
  if USE_RFE:
    clf = RFE(RandomForestClassifier(class_weight="balanced", **clf.best_params_), n_features_to_select=NB_OF_BEST_FEATURES).fit(X_train, y_train)
    nb_features = clf.n_features_
    _ = sorted(list(zip(clf.feature_names_in_,clf.ranking_)), key=lambda x:x[1])
    features = list(map(lambda x:x[0], _[:nb_features]))
    print(f"features:{features} ranking = {_}")
  else:
    nb_features = len(clf.feature_names_in_)
    print("used: ",clf.feature_names_in_)
  #clf = RandomForestClassifier(max_depth=15, max_leaf_nodes=5).fit(X_train, y_train)
  y_train_proba = clf.predict_proba(X_train)[:, 1] #[proba({label=1}/row_data) for row_data in X_train]
  print("y_train_proba:",len(set(y_train_proba)))
  y_test_proba = clf.predict_proba(X_test)[:, 1]
  if USE_BEST_THRESH:
    best_sc, best_thresh = find_best_thresh(y_train,y_train_proba)
    y_train_pred = y_train_proba > best_thresh
    y_test_pred = y_test_proba > best_thresh
  else:
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

  if nb_features<6:
    show_random_forest(clf.estimator_, features)
  return y_test_proba, y_train_proba, y_train_pred, y_test_pred, nb_features


def ada_boost_classifier(X_train, y_train, X_test, y_test):
  parameters = {"n_estimators":[5,25,100,200]}
  clf = GridSearchCV(AdaBoostClassifier(), parameters).fit(X_train, y_train)
  print(clf.best_params_)
  #clf = RandomForestClassifier(max_depth=15, max_leaf_nodes=5).fit(X_train, y_train)
  y_train_pred = clf.predict(X_train)
  y_test_pred = clf.predict(X_test)
  y_train_proba = clf.predict_proba(X_train)[:, 1] #[proba({label=1}/row_data) for row_data in X_train]
  print("y_train_proba:",len(set(y_train_proba)))
  y_test_proba = clf.predict_proba(X_test)[:, 1]
  return y_test_proba, y_train_proba, y_train_pred, y_test_pred, len(clf.feature_names_in_)

def gradient_boosting_classifier(X_train, y_train, X_test, y_test):
  parameters = {"max_depth":[1,5,10], "n_estimators":[10,50,100]}
  clf = GridSearchCV(GradientBoostingClassifier(learning_rate=1.0, random_state=0), parameters).fit(X_train, y_train)
  print(clf.best_params_)
  y_train_pred = clf.predict(X_train)
  y_test_pred = clf.predict(X_test)
  y_train_proba = clf.predict_proba(X_train)[:, 1] #[proba({label=1}/row_data) for row_data in X_train]
  print("y_train_proba:",len(set(y_train_proba)))
  y_test_proba = clf.predict_proba(X_test)[:, 1]
  return y_test_proba, y_train_proba, y_train_pred, y_test_pred, len(clf.feature_names_in_)


```


```python
class M_Classifier_res:
    def __init__(self, main_df, target_name:str, description:str, important_cols:list=None, test_size:float=0.2, oversample=False, feature_eng=True):
        self.description = description

        self.MAIN_DF = main_df
        assert isinstance(self.MAIN_DF, pd.DataFrame)

        self.TARGET_NAME:str = target_name
        assert self.TARGET_NAME in self.MAIN_DF.columns

        if important_cols is None: important_cols = self.MAIN_DF.columns.to_list()
        self.important_cols:list[str] = important_cols
        assert isinstance(self.important_cols, list)
        assert len(self.important_cols) > 0
        self.important_cols = list(set(self.important_cols)-{self.TARGET_NAME})
        assert len(set(important_cols)- set(self.MAIN_DF.columns) ) == 0, f"{set(important_cols)- set(self.MAIN_DF.columns)} not found in df"


        self.DICT_RES:dict = {}
        self.TEST_SIZE:float = float(test_size) or 0.2
        assert 0<self.TEST_SIZE<1

        self.choosen_columns = []

        self.oversample = bool(oversample)

        self.feature_eng = bool(feature_eng)

        print("MAIN_DF.shape:",self.MAIN_DF.shape)
        print("TARGET_NAME:",self.TARGET_NAME)
        print("important_cols:",self.important_cols)
        print("TEST_SIZE:",self.TEST_SIZE)
        print("oversample:",self.oversample)
        print("feature_eng:",self.feature_eng)

        self.dict_tresh = optimize_tresh(self.MAIN_DF, target=self.TARGET_NAME)

    def norm_test_size(self, test_size:float=None):
      if test_size is None: test_size=self.TEST_SIZE
      test_size = float(test_size);assert 0<test_size<1;print("test_size = ",test_size)
      return test_size

    def norm_select_cols(self, important_cols=None, remove_cols:list=None):
      # select cols
      if important_cols is None: important_cols = self.important_cols
      else: important_cols = list(important_cols)
      if remove_cols is not None:
        remove_cols = [remove_cols] if isinstance(remove_cols,str) else remove_cols
        important_cols = list(set(important_cols)-set(remove_cols))
      print("important_cols = ",important_cols)
      return important_cols

    def norm_oversampling(self, oversample):
      if oversample is None: oversample = self.oversample
      if oversample:
        dd = {"sm":"smote","du":"duplic"}
        default = list(dd.values())[0]
        if isinstance(oversample, str):
          oversample = oversample.lower()
          print('oversample used = ',oversample)
          method:str = oversample if oversample in dd.values() else  dd[oversample] if oversample in dd.keys() else default
        else: method = default
      else: method = None
      print("oversample = ",oversample, "method=",method)
      return method

    def norm_feature_eng(self, feature_eng:bool=None):
      if feature_eng is None: feature_eng = self.feature_eng
      print("feature_eng = ",feature_eng)
      return feature_eng


    def transform_data(self, model_name, feature_eng=False, cols=None, return_last=False):
      cols = self.norm_select_cols(important_cols=cols)
      cols = set(cols).union({self.TARGET_NAME})
      return self.MAIN_DF.copy() if not feature_eng else transform_data_from_feature_eng(self.MAIN_DF, model_name, dict_tresh=self.dict_tresh, cols=cols, return_last=False)


    def do_split(self, df, important_cols=None, remove_cols:list=None, test_size=None):
      important_cols = self.norm_select_cols(important_cols=important_cols, remove_cols=remove_cols)
      test_size = self.norm_test_size(test_size)

      X = df[important_cols]
      if self.TARGET_NAME in X.columns: X=X.drop(self.TARGET_NAME, axis=1)
      y = df[self.TARGET_NAME]
      return train_test_split(X, y, test_size=test_size, random_state=10, stratify=y, shuffle=True)



    def get_train_test(self, model_name:str=None, important_cols:list=None, oversample:bool=None, test_size:float=None, remove_cols:list=None, feature_eng:bool=None):
      feature_eng = self.norm_feature_eng(feature_eng)
      method = self.norm_oversampling(oversample)

      DF = self.transform_data(model_name, feature_eng=feature_eng)
      X_train, X_test, y_train, y_test = self.do_split(DF, important_cols=important_cols, remove_cols=remove_cols, test_size=test_size)

      # drop duplicates from x_test #what id duplicates have diff target hh
      '''y_test.index = X_test.index
      X_test = X_test.drop_duplicates()
      y_test = y_test.loc[X_test.index]'''
      if method: X_train, y_train = oversampling_(X_train, y_train, method=method)

      return X_train, X_test, y_train, y_test

    def compute_all(self, model_name, important_cols:list=None, oversample:bool=None, test_size:float=None, remove_cols:list=None, feature_eng:bool=None):
      print(f"model loader: {self.description}")
      list_fct = {
            "logit": self.logit_regression,
            "logistic": logistic_regression_sklearn,
            "svc": svc_classifier,
            "knn":knn_classifier,
            "sdg":sdg_classifier,
            "tree":decision_tree_classifier,
            "lda": lda_classifier,
            "forest":random_forest_classifier,
            "ada": ada_boost_classifier,
            "xgboost": gradient_boosting_classifier
            }

      # transform, split and oversample
      X_train, X_test, y_train, y_test = self.get_train_test(model_name=model_name, important_cols=important_cols, oversample=oversample, test_size=test_size, remove_cols=remove_cols, feature_eng=feature_eng)

      # get model
      my_fct = list_fct[model_name]
      print(f"{'':~^50}\n{model_name:~^50}\n{'':~^50}")

      # train
      y_test_proba, y_train_proba, y_train_pred, y_test_pred, nb_params = my_fct(X_train, y_train, X_test, y_test)

      # evaluate
      train_res, test_res = show_results(y_train, y_train_pred, y_train_proba,y_test, y_test_pred, y_test_proba)
      g = f"{model_name}: nb_params = {nb_params}"
      print(f"{g:~^50}")

      # check if it is an amelioration
      if not self.DICT_RES.get(model_name):self.DICT_RES[model_name] = {"train_res":train_res, "test_res":test_res};return
      a, b = test_res['f1'] - self.DICT_RES[model_name]["test_res"]["f1"], train_res['f1'] - self.DICT_RES[model_name]["train_res"]["f1"]
      get_pre = lambda x: f'gain {100*x:.2f}%' if x>0 else f'lost {100*x:.2f}%' if x<0 else '~~~~'
      add_ = "NOPE"
      if (a>0 and b>0): self.DICT_RES[model_name] = {"train_res":train_res, "test_res":test_res}; add_ = "GREAT"
      print(f"{add_} !! {get_pre(a)} on test and {get_pre(b)} on train")



    def logit_regression(self, X_train, y_train, X_test, y_test):
        '''permet de trouver les variables significatives pour lesquells aller aux featureengenieering as theyimpact the models'''
        list_redundant_cols = []
        list_used_cols = []
        for col_name in X_train.columns:
          list_used_cols.append(col_name)
          print("list_used_cols=",list_used_cols)
          try:
            log_reg = Logit(y_train, X_train[list_used_cols]).fit()
            print(log_reg.summary())
            self.choosen_columns = [x for x in log_reg.pvalues.index if log_reg.pvalues[x]<0.05];is_choosen_columns_set = True
            print(f"highly_significant_columns: {sum(log_reg.pvalues<0.05)}/{len(log_reg.pvalues)}: {self.choosen_columns}")
          except np.linalg.LinAlgError:
            list_redundant_cols.append(list_used_cols.pop())
            print("removing ",col_name)

        if not list_used_cols: print("wtf"); return


        y_test_proba = log_reg.predict(X_test[list_used_cols])
        y_train_proba = log_reg.predict(X_train[list_used_cols])
        y_train_pred = np.where(y_train_proba>0.5, 1, 0)
        y_test_pred = np.where(y_test_proba>0.5, 1, 0)
        return y_test_proba, y_train_proba, y_train_pred, y_test_pred, len(log_reg.params)

    def show_res(self):
        red_dd = []
        for model,res in self.DICT_RES.items():
          red_dd.append(append_dict({'type':'train',"model":model}, res['train_res']))
          red_dd.append(append_dict({'type':'test',"model":model}, res['test_res']))
        if red_dd: pd.DataFrame(red_dd).groupby("type").apply(print)

```


```python
from scipy.stats import shapiro

def blancheur(arr, name=""):
  _ = shapiro(arr)
  print(f"{name if name else 'data'}: {'not ' if _.pvalue>0.05 else ''}significantly normal")

def only_logit_runner(X_, y_):
  reg = Logit(y_, X_).fit()
  print(reg.summary())
  y_p = reg.predict(X_)
  print("scores:",return_dict_scores(y_, y_p>=0.35, y_p))
  blancheur(y_-y_p, name="residuals")
  sns.scatterplot(range(len(y_)), y_-y_p, hue=y_)
  plt.show()
  print("sorted pvalues\n",reg.pvalues.sort_values())
```


```python
'''_ = np.array([1,2,3,4,5])
print(_)
__ = list(map(lambda x: int(x<3), _ + np.random.randn(len(_))))
print(__)
only_logit_runner(pd.DataFrame({"a":_}), __)'''
```




    '_ = np.array([1,2,3,4,5])\nprint(_)\n__ = list(map(lambda x: int(x<3), _ + np.random.randn(len(_))))\nprint(__)\nonly_logit_runner(pd.DataFrame({"a":_}), __)'



## sampling


```python

def append_dict(d1, d2):
  d1.update(d2)
  return d1

NB_NEIGHBORS = 3

def oversampling_(X, y, method:str):
  if method=="duplic":
    from imblearn.over_sampling import RandomOverSampler
    X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(X, y)
  elif method=="smote":
    from imblearn.over_sampling import SMOTE #SMOTENC
    X_resampled, y_resampled = SMOTE(k_neighbors=NB_NEIGHBORS).fit_resample(X, y)
  return X_resampled, y_resampled

oversampling_(np.array([1.5,2,4,4,5,6,7,8,9,10]).reshape(-1,1),[1,1,1,1,0,0,0,0,0,0], method="smote")
```




    (array([[ 1.5       ],
            [ 2.        ],
            [ 4.        ],
            [ 4.        ],
            [ 5.        ],
            [ 6.        ],
            [ 7.        ],
            [ 8.        ],
            [ 9.        ],
            [10.        ],
            [ 3.51050293],
            [ 1.87149417]]), [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])



# Analyse du problème

# Analyse descriptive

## simple preview


```python
df
```





  <div id="df-1981ac98-f609-44ee-a56d-6009a7eadc9a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bidder_id</th>
      <th>bid_id</th>
      <th>auction</th>
      <th>merchandise</th>
      <th>device</th>
      <th>time</th>
      <th>country</th>
      <th>ip</th>
      <th>url</th>
      <th>payment_account</th>
      <th>address</th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>001068c415025a009fee375a12cff4fcnht8y</td>
      <td>7179832</td>
      <td>4ifac</td>
      <td>jewelry</td>
      <td>phone561</td>
      <td>5.140996e-308</td>
      <td>bn</td>
      <td>139.226.147.115</td>
      <td>vasstdc27m7nks3</td>
      <td>a3d2de7675556553a5f08e4c88d2c228iiasc</td>
      <td>a3d2de7675556553a5f08e4c88d2c2282aj35</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0030a2dd87ad2733e0873062e4f83954mkj86</td>
      <td>6805028</td>
      <td>obbny</td>
      <td>mobile</td>
      <td>phone313</td>
      <td>5.139226e-308</td>
      <td>ir</td>
      <td>21.67.17.162</td>
      <td>vnw40k8zzokijsv</td>
      <td>a3d2de7675556553a5f08e4c88d2c228jem8t</td>
      <td>f3bc67b04b43c3cebd1db5ed4941874c9br67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>2501797</td>
      <td>l3o6q</td>
      <td>books and music</td>
      <td>phone451</td>
      <td>5.067829e-308</td>
      <td>bh</td>
      <td>103.165.41.136</td>
      <td>kk7rxe25ehseyci</td>
      <td>52743ba515e9c1279ac76e19f00c0b001p3pm</td>
      <td>7578f951008bd0b64528bf81b8578d5djy0uy</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>2724778</td>
      <td>du967</td>
      <td>books and music</td>
      <td>phone117</td>
      <td>5.068704e-308</td>
      <td>tr</td>
      <td>239.250.228.152</td>
      <td>iu2iu3k137vakme</td>
      <td>52743ba515e9c1279ac76e19f00c0b001p3pm</td>
      <td>7578f951008bd0b64528bf81b8578d5djy0uy</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>2742648</td>
      <td>wx3kf</td>
      <td>books and music</td>
      <td>phone16</td>
      <td>5.068805e-308</td>
      <td>in</td>
      <td>255.108.248.101</td>
      <td>u85yj2e7owkz6xp</td>
      <td>52743ba515e9c1279ac76e19f00c0b001p3pm</td>
      <td>7578f951008bd0b64528bf81b8578d5djy0uy</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>1411172</td>
      <td>toxfq</td>
      <td>mobile</td>
      <td>phone1036</td>
      <td>5.201503e-308</td>
      <td>in</td>
      <td>186.94.48.203</td>
      <td>vasstdc27m7nks3</td>
      <td>22cdb26663f071c00de61cc2dcde7b556rido</td>
      <td>db147bf6056d00428b1bbf250c6e97594ewjy</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>1411587</td>
      <td>ucb4u</td>
      <td>mobile</td>
      <td>phone127</td>
      <td>5.201506e-308</td>
      <td>in</td>
      <td>119.27.26.126</td>
      <td>vasstdc27m7nks3</td>
      <td>22cdb26663f071c00de61cc2dcde7b556rido</td>
      <td>db147bf6056d00428b1bbf250c6e97594ewjy</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>1411727</td>
      <td>sg8yd</td>
      <td>mobile</td>
      <td>phone383</td>
      <td>5.201507e-308</td>
      <td>in</td>
      <td>243.25.54.63</td>
      <td>yweo7wfejrgbi2d</td>
      <td>22cdb26663f071c00de61cc2dcde7b556rido</td>
      <td>db147bf6056d00428b1bbf250c6e97594ewjy</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>1411877</td>
      <td>toaj7</td>
      <td>mobile</td>
      <td>phone26</td>
      <td>5.201508e-308</td>
      <td>in</td>
      <td>17.66.120.232</td>
      <td>4dd8ei0o5oqsua3</td>
      <td>22cdb26663f071c00de61cc2dcde7b556rido</td>
      <td>db147bf6056d00428b1bbf250c6e97594ewjy</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>1412085</td>
      <td>07axb</td>
      <td>mobile</td>
      <td>phone25</td>
      <td>5.201509e-308</td>
      <td>in</td>
      <td>64.30.57.156</td>
      <td>8zdkeqk4yby6lz2</td>
      <td>22cdb26663f071c00de61cc2dcde7b556rido</td>
      <td>db147bf6056d00428b1bbf250c6e97594ewjy</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 12 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1981ac98-f609-44ee-a56d-6009a7eadc9a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1981ac98-f609-44ee-a56d-6009a7eadc9a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1981ac98-f609-44ee-a56d-6009a7eadc9a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




c'est une classification binaire


```python
df[TARGET_COL].value_counts()
```




    0    90877
    1     9123
    Name: outcome, dtype: int64



lignes vides par colonne


```python
df.isnull().sum()
```




    bidder_id            0
    bid_id               0
    auction              0
    merchandise          0
    device               0
    time                 0
    country            184
    ip                   0
    url                  0
    payment_account      0
    address              0
    outcome              0
    dtype: int64



des cellules vides que dans country


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 12 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   bidder_id        100000 non-null  object 
     1   bid_id           100000 non-null  int64  
     2   auction          100000 non-null  object 
     3   merchandise      100000 non-null  object 
     4   device           100000 non-null  object 
     5   time             100000 non-null  float64
     6   country          99816 non-null   object 
     7   ip               100000 non-null  object 
     8   url              100000 non-null  object 
     9   payment_account  100000 non-null  object 
     10  address          100000 non-null  object 
     11  outcome          100000 non-null  int64  
    dtypes: float64(1), int64(2), object(9)
    memory usage: 9.2+ MB


près de 200 cellules vides dans country. on verra ça après


```python
df.head()
```





  <div id="df-601513fd-ebf0-4896-b6b9-266045820772">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bidder_id</th>
      <th>bid_id</th>
      <th>auction</th>
      <th>merchandise</th>
      <th>device</th>
      <th>time</th>
      <th>country</th>
      <th>ip</th>
      <th>url</th>
      <th>payment_account</th>
      <th>address</th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>001068c415025a009fee375a12cff4fcnht8y</td>
      <td>7179832</td>
      <td>4ifac</td>
      <td>jewelry</td>
      <td>phone561</td>
      <td>5.140996e-308</td>
      <td>bn</td>
      <td>139.226.147.115</td>
      <td>vasstdc27m7nks3</td>
      <td>a3d2de7675556553a5f08e4c88d2c228iiasc</td>
      <td>a3d2de7675556553a5f08e4c88d2c2282aj35</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0030a2dd87ad2733e0873062e4f83954mkj86</td>
      <td>6805028</td>
      <td>obbny</td>
      <td>mobile</td>
      <td>phone313</td>
      <td>5.139226e-308</td>
      <td>ir</td>
      <td>21.67.17.162</td>
      <td>vnw40k8zzokijsv</td>
      <td>a3d2de7675556553a5f08e4c88d2c228jem8t</td>
      <td>f3bc67b04b43c3cebd1db5ed4941874c9br67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>2501797</td>
      <td>l3o6q</td>
      <td>books and music</td>
      <td>phone451</td>
      <td>5.067829e-308</td>
      <td>bh</td>
      <td>103.165.41.136</td>
      <td>kk7rxe25ehseyci</td>
      <td>52743ba515e9c1279ac76e19f00c0b001p3pm</td>
      <td>7578f951008bd0b64528bf81b8578d5djy0uy</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>2724778</td>
      <td>du967</td>
      <td>books and music</td>
      <td>phone117</td>
      <td>5.068704e-308</td>
      <td>tr</td>
      <td>239.250.228.152</td>
      <td>iu2iu3k137vakme</td>
      <td>52743ba515e9c1279ac76e19f00c0b001p3pm</td>
      <td>7578f951008bd0b64528bf81b8578d5djy0uy</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>2742648</td>
      <td>wx3kf</td>
      <td>books and music</td>
      <td>phone16</td>
      <td>5.068805e-308</td>
      <td>in</td>
      <td>255.108.248.101</td>
      <td>u85yj2e7owkz6xp</td>
      <td>52743ba515e9c1279ac76e19f00c0b001p3pm</td>
      <td>7578f951008bd0b64528bf81b8578d5djy0uy</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-601513fd-ebf0-4896-b6b9-266045820772')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-601513fd-ebf0-4896-b6b9-266045820772 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-601513fd-ebf0-4896-b6b9-266045820772');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## transformations


```python
# proportion des valeurs nulles
df.country.isnull().sum()/len(df)
```




    0.00184




```python
df[df.country.isnull()]
```





  <div id="df-bffeb60d-285a-41de-96c0-3adadcb1f935">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bidder_id</th>
      <th>bid_id</th>
      <th>auction</th>
      <th>merchandise</th>
      <th>device</th>
      <th>time</th>
      <th>country</th>
      <th>ip</th>
      <th>url</th>
      <th>payment_account</th>
      <th>address</th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>179</th>
      <td>01067975436d123f717ee5aba0dd4bbfa0937</td>
      <td>205998</td>
      <td>jefix</td>
      <td>jewelry</td>
      <td>phone106</td>
      <td>5.193927e-308</td>
      <td>NaN</td>
      <td>184.120.87.225</td>
      <td>vasstdc27m7nks3</td>
      <td>a3d2de7675556553a5f08e4c88d2c228d0upt</td>
      <td>ca7ebbf817ced6e7194200eb690eda41u4mda</td>
      <td>0</td>
    </tr>
    <tr>
      <th>208</th>
      <td>01067975436d123f717ee5aba0dd4bbfa0937</td>
      <td>332482</td>
      <td>jefix</td>
      <td>jewelry</td>
      <td>phone20</td>
      <td>5.194699e-308</td>
      <td>NaN</td>
      <td>80.26.116.170</td>
      <td>vasstdc27m7nks3</td>
      <td>a3d2de7675556553a5f08e4c88d2c228d0upt</td>
      <td>ca7ebbf817ced6e7194200eb690eda41u4mda</td>
      <td>0</td>
    </tr>
    <tr>
      <th>253</th>
      <td>01067975436d123f717ee5aba0dd4bbfa0937</td>
      <td>490346</td>
      <td>jefix</td>
      <td>jewelry</td>
      <td>phone63</td>
      <td>5.196028e-308</td>
      <td>NaN</td>
      <td>5.65.174.254</td>
      <td>vasstdc27m7nks3</td>
      <td>a3d2de7675556553a5f08e4c88d2c228d0upt</td>
      <td>ca7ebbf817ced6e7194200eb690eda41u4mda</td>
      <td>0</td>
    </tr>
    <tr>
      <th>314</th>
      <td>01067975436d123f717ee5aba0dd4bbfa0937</td>
      <td>699781</td>
      <td>jefix</td>
      <td>jewelry</td>
      <td>phone26</td>
      <td>5.197355e-308</td>
      <td>NaN</td>
      <td>252.90.100.143</td>
      <td>vasstdc27m7nks3</td>
      <td>a3d2de7675556553a5f08e4c88d2c228d0upt</td>
      <td>ca7ebbf817ced6e7194200eb690eda41u4mda</td>
      <td>0</td>
    </tr>
    <tr>
      <th>410</th>
      <td>01067975436d123f717ee5aba0dd4bbfa0937</td>
      <td>1054589</td>
      <td>jefix</td>
      <td>jewelry</td>
      <td>phone1502</td>
      <td>5.198777e-308</td>
      <td>NaN</td>
      <td>208.208.32.103</td>
      <td>vasstdc27m7nks3</td>
      <td>a3d2de7675556553a5f08e4c88d2c228d0upt</td>
      <td>ca7ebbf817ced6e7194200eb690eda41u4mda</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>88231</th>
      <td>06a58d4af0fe7ee15324b0e921e8de1260vef</td>
      <td>7577841</td>
      <td>jqx39</td>
      <td>mobile</td>
      <td>phone150</td>
      <td>5.143484e-308</td>
      <td>NaN</td>
      <td>138.25.39.139</td>
      <td>6wm9l6ecu5ynnom</td>
      <td>19df84e356014e0098f8b4d5bcd18988g9dcq</td>
      <td>c0c4aa332b969cc70ab705e699f40aa3fxg29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>88357</th>
      <td>06a58d4af0fe7ee15324b0e921e8de1260vef</td>
      <td>7650495</td>
      <td>jqx39</td>
      <td>mobile</td>
      <td>phone125</td>
      <td>5.143813e-308</td>
      <td>NaN</td>
      <td>102.141.77.207</td>
      <td>xj4r8a6nnv6o9ns</td>
      <td>19df84e356014e0098f8b4d5bcd18988g9dcq</td>
      <td>c0c4aa332b969cc70ab705e699f40aa3fxg29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>88364</th>
      <td>06a58d4af0fe7ee15324b0e921e8de1260vef</td>
      <td>7655741</td>
      <td>jqx39</td>
      <td>mobile</td>
      <td>phone106</td>
      <td>5.143836e-308</td>
      <td>NaN</td>
      <td>200.78.201.247</td>
      <td>myn66laevndsnkc</td>
      <td>19df84e356014e0098f8b4d5bcd18988g9dcq</td>
      <td>c0c4aa332b969cc70ab705e699f40aa3fxg29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>88949</th>
      <td>0825d381aa09d1d0c47c77c372f07acbe02ky</td>
      <td>1030050</td>
      <td>jefix</td>
      <td>jewelry</td>
      <td>phone150</td>
      <td>5.198648e-308</td>
      <td>NaN</td>
      <td>153.240.160.26</td>
      <td>mzluxud8iffwg4h</td>
      <td>a3d2de7675556553a5f08e4c88d2c2284gyp4</td>
      <td>3ea07e75644ad41133cc05c3957395d7l50q9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>89019</th>
      <td>0825d381aa09d1d0c47c77c372f07acbe02ky</td>
      <td>1800536</td>
      <td>jefix</td>
      <td>jewelry</td>
      <td>phone169</td>
      <td>5.203042e-308</td>
      <td>NaN</td>
      <td>180.88.71.35</td>
      <td>57padg8kutycdcx</td>
      <td>a3d2de7675556553a5f08e4c88d2c2284gyp4</td>
      <td>3ea07e75644ad41133cc05c3957395d7l50q9</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>184 rows × 12 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bffeb60d-285a-41de-96c0-3adadcb1f935')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-bffeb60d-285a-41de-96c0-3adadcb1f935 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bffeb60d-285a-41de-96c0-3adadcb1f935');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# champs vides
df.loc[df.country.isnull(), "country"] = "NO_COUNTRY"
```


```python
df[df.country.isnull()]
```





  <div id="df-a251b3e4-1ac0-44e6-87cc-2d211db0aa3b">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bidder_id</th>
      <th>bid_id</th>
      <th>auction</th>
      <th>merchandise</th>
      <th>device</th>
      <th>time</th>
      <th>country</th>
      <th>ip</th>
      <th>url</th>
      <th>payment_account</th>
      <th>address</th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a251b3e4-1ac0-44e6-87cc-2d211db0aa3b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-a251b3e4-1ac0-44e6-87cc-2d211db0aa3b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a251b3e4-1ac0-44e6-87cc-2d211db0aa3b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df.ip
```




    0        139.226.147.115
    1           21.67.17.162
    2         103.165.41.136
    3        239.250.228.152
    4        255.108.248.101
                  ...       
    99995      186.94.48.203
    99996      119.27.26.126
    99997       243.25.54.63
    99998      17.66.120.232
    99999       64.30.57.156
    Name: ip, Length: 100000, dtype: object




```python
# identify network instead of device
df["ip"] = df.ip.apply(lambda x: '.'.join(x.split('.')[:2]))
```


```python
df.ip
```




    0        139.226
    1          21.67
    2        103.165
    3        239.250
    4        255.108
              ...   
    99995     186.94
    99996     119.27
    99997     243.25
    99998      17.66
    99999      64.30
    Name: ip, Length: 100000, dtype: object




```python
df.groupby("ip").agg({"country": lambda x: x.nunique()}).country.value_counts()
```




    1     25426
    2      7918
    3      1469
    4       205
    5        44
    6         9
    8         4
    7         3
    10        2
    9         2
    13        1
    Name: country, dtype: int64




```python
df.time.describe()
```




    count     1.000000e+05
    mean     5.143168e-308
    std       0.000000e+00
    min      5.067452e-308
    25%      5.079343e-308
    50%      5.139090e-308
    75%      5.197759e-308
    max      5.206746e-308
    Name: time, dtype: float64




```python
# normalize time
df["time"] = (df.time - df.time.min())/(df.time.max() - df.time.min())
```


```python
df["time"].describe()
```




    count    100000.000000
    mean          0.543571
    std           0.362483
    min           0.000000
    25%           0.085364
    50%           0.514295
    75%           0.935483
    max           1.000000
    Name: time, dtype: float64




```python
df.bidder_id.nunique()
```




    87



## infomations on columns
- dtype : type of the columns as read by pandas
- nunique  : nb of unique elements in the columns
- nunique(%)  : percentage = nunique/len(df)
- nunique_per_bid>1(%)  : percentage = nb_of_bidder_id_with_more_than_one_value_for_the_column / nb_bidder_id
- is_cat  : if the columns has less than 10 unique values


```python

def get_cols_info(df, index_col=None):
    from math import ceil
    print(">>> df.shape= ", df.shape)
    print("\n>>> df.info= ")
    df.info()
    dd = {"col":[],"dtype":[],"nunique":[],"nunique(%)":[],"nunique_per_bid>1(%)":[],"is_cat":[]}
    for elt in df.columns:
      dd["col"].append(elt)
      dd["nunique"].append(df[elt].nunique())
      dd["nunique(%)"].append(0.1*ceil(10*100*df[elt].nunique()/len(df)))
      dd["dtype"].append(df[elt].dtype)
      dd["is_cat"].append(int(df[elt].nunique()<10))
      if index_col: dd["nunique_per_bid>1(%)"].append(0.1*ceil(10*100*(df.groupby(index_col)[elt].nunique()>1).sum()/df[index_col].nunique()))
      else: dd["nunique_per_bid>1(%)"].append('')
    list_indx = dd["col"]
    del dd["col"]
    print("\n>>> df.more_info= ")
    print(pd.DataFrame(dd, index=list_indx).sort_values(by=['nunique']))
    print("\n>>> df.describe= ")
    print(df.describe())

get_cols_info(df, "bidder_id")
```

    >>> df.shape=  (100000, 12)
    
    >>> df.info= 
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 12 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   bidder_id        100000 non-null  object 
     1   bid_id           100000 non-null  int64  
     2   auction          100000 non-null  object 
     3   merchandise      100000 non-null  object 
     4   device           100000 non-null  object 
     5   time             100000 non-null  float64
     6   country          100000 non-null  object 
     7   ip               100000 non-null  object 
     8   url              100000 non-null  object 
     9   payment_account  100000 non-null  object 
     10  address          100000 non-null  object 
     11  outcome          100000 non-null  int64  
    dtypes: float64(1), int64(2), object(9)
    memory usage: 9.2+ MB
    
    >>> df.more_info= 
                       dtype  nunique  nunique(%)  nunique_per_bid>1(%)  is_cat
    outcome            int64        2         0.1                   0.0       1
    merchandise       object        6         0.1                   0.0       1
    bidder_id         object       87         0.1                   0.0       0
    payment_account   object       87         0.1                   0.0       0
    address           object       87         0.1                   0.0       0
    country           object      175         0.2                  70.2       0
    device            object     1871         1.9                  80.5       0
    auction           object     3438         3.5                  79.4       0
    url               object    21951        22.0                  72.5       0
    ip                object    35083        35.1                  84.0       0
    time             float64    92385        92.4                  85.1       0
    bid_id             int64   100000       100.0                  85.1       0
    
    >>> df.describe= 
                 bid_id           time        outcome
    count  1.000000e+05  100000.000000  100000.000000
    mean   3.697622e+06       0.543571       0.091230
    std    2.380217e+06       0.362483       0.287937
    min    8.900000e+01       0.000000       0.000000
    25%    1.463762e+06       0.085364       0.000000
    50%    3.660968e+06       0.514295       0.000000
    75%    5.881387e+06       0.935483       0.000000
    max    7.656326e+06       1.000000       1.000000



```python
if "merchandise" in df.columns: df.groupby("auction").agg({"merchandise":lambda x: x.nunique()}).merchandise.value_counts()
```


```python
df.isnull().sum()
```




    bidder_id          0
    bid_id             0
    auction            0
    merchandise        0
    device             0
    time               0
    country            0
    ip                 0
    url                0
    payment_account    0
    address            0
    outcome            0
    dtype: int64



actions
- outcome: okay
  - the target
- merchandise: okay
  - no info supp à bidder_id (nunique_per_bid>1(%))
  - categorical: okay with encoding (one hot encoding)
- bidder_id: okay
  - The first input of the model for a groupby
- payment_account: remove
  - no info supp à bidder_id (nunique_per_bid>1(%))
  - a str field with no interesting extractions
- address: remove
  - no info supp à bidder_id (nunique_per_bid>1(%))
  - a str field with no interesting extractions no even for spatial work identification unless i know how the tokens are created
- country: stay
  - 0.17 % info supp à bidder_id (nunique_per_bid>1(%))
  - a str field with no interesting extractions unless we're doing spatial work
  - i can pull a mean encoding
- device: quantic
  - 1.8 % who knows
  - less unique device than ip. Because many bidder_id can have the same device. But does the device matter ?
  - i can still pull a mean encoding
- auction: stay (on hypothesis testing)
  - 3.5% diff!!
  - maybe there are some events where only bots have an advantage ?! Or bots sttrugle ?!
  - no interesting extractions
- url: stay (on hypothesis testing)
  - 22% diff
  - may seem interesting for extraction but nope. it is like a token
  - let's pull a mean encoding if it adds some value
- ip: stay (on hypothesis testing)
  - 35% diff
  - may seem interesting for extraction but nope. it is like a token
  - let's pull a mean encoding if it adds some value
- time: stay
  - 92% diff
  - a bot is quick on manipulations. That may be helful
- bid_id: remove
  - 100% diff
  - no interesting extractions

## relevant feature searching


```python
df.groupby("merchandise").url.nunique()
```




    merchandise
    books and music      113
    home goods          6649
    jewelry             3992
    mobile              3534
    office equipment      14
    sporting goods      7700
    Name: url, dtype: int64




```python
df.groupby("url").merchandise.nunique().value_counts()
#for one url, we can have many product up to 47
```




    1    21904
    2       46
    6        1
    Name: merchandise, dtype: int64




```python
df.groupby("auction").merchandise.nunique().value_counts()
#for one auction, we can have many products up to 812
```




    1    2626
    2     640
    3     145
    4      23
    5       4
    Name: merchandise, dtype: int64



### agregation on bidder_id and time


```python
df.groupby(["bidder_id", "time"]).url.nunique().value_counts()
#1700
```




    1    95680
    2     1649
    3       49
    4        2
    Name: url, dtype: int64




```python
df.columns
```




    Index(['bidder_id', 'bid_id', 'auction', 'merchandise', 'device', 'time',
           'country', 'ip', 'url', 'payment_account', 'address', 'outcome'],
          dtype='object')




```python
def compute_groupby(filename):
  dd = df.groupby(["bidder_id", "time"]).agg({
              "auction":lambda x: x.nunique() - 1,
              "device":lambda x: x.nunique() - 1,
              "country":lambda x: x.nunique() - 1,
              "ip":lambda x: x.nunique() - 1,
              "url":lambda x: x.nunique() - 1,
              "outcome":lambda x: x.unique()[0]
              })
  cls_ =list(set(dd.columns)-{'outcome','bidder_id','time'})
  dd["my_agg"] = dd[cls_].product(axis=1)
  dd2 = dd.reset_index()
  dd_min_per_bidder_id = {bidder_id: dd2[dd2.bidder_id==bidder_id].time.min() for bidder_id in dd2.bidder_id.unique()}
  def modif_time(x):
    x["time"] = x["time"] - dd_min_per_bidder_id[x["bidder_id"]]
    return x
  dd2 = dd2.apply(modif_time, axis=1)
  dd2.to_csv(filename)
  return dd2
```


```python
filename_dd = f"df_groupby_{FILE_VERSION}_v2.csv" if PROD_INSTEAD_OF_SUM else f"df_groupby_{FILE_VERSION}.csv"
if not os.path.exists(filename_dd):
  print('running')
  dd2 = compute_groupby(filename_dd)
else:
  print('found')
  dd2 = pd.read_csv(filename_dd)
  if "Unnamed: 0" in dd2.columns: del dd2["Unnamed: 0"]

dd2
```

    found






  <div id="df-af4d503d-0cc9-421c-b6c0-664c23fab1dc">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bidder_id</th>
      <th>time</th>
      <th>auction</th>
      <th>device</th>
      <th>country</th>
      <th>ip</th>
      <th>url</th>
      <th>outcome</th>
      <th>my_agg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>001068c415025a009fee375a12cff4fcnht8y</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0030a2dd87ad2733e0873062e4f83954mkj86</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>0.006280</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>0.007004</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>97375</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>0.059113</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97376</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>0.059130</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97377</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>0.059136</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97378</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>0.059143</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97379</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>0.059152</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>97380 rows × 9 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-af4d503d-0cc9-421c-b6c0-664c23fab1dc')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-af4d503d-0cc9-421c-b6c0-664c23fab1dc button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-af4d503d-0cc9-421c-b6c0-664c23fab1dc');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
#dd3 = dd2[dd2.my_agg>0]#[["bidder_id","outcome","my_agg"]] #rm min
dd4 = dd2.groupby("bidder_id").agg({"time":["count", "sum", "mean", "max","std"],"my_agg":["sum","mean","max", "std"], "outcome":lambda x: x.unique()[0]}).reset_index()
```


```python
for a,b in dd4.columns:
  if b!='std': continue
  dd4.loc[dd4[(a,b)].isnull(), (a,b)] = 0
dd4
```





  <div id="df-ee7f4d73-4d8a-4348-842c-5b9a153aa2bc">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>bidder_id</th>
      <th colspan="5" halign="left">time</th>
      <th colspan="4" halign="left">my_agg</th>
      <th>outcome</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>sum</th>
      <th>mean</th>
      <th>max</th>
      <th>std</th>
      <th>sum</th>
      <th>mean</th>
      <th>max</th>
      <th>std</th>
      <th>&lt;lambda&gt;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>001068c415025a009fee375a12cff4fcnht8y</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0030a2dd87ad2733e0873062e4f83954mkj86</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>141</td>
      <td>63.451005</td>
      <td>0.450007</td>
      <td>0.544919</td>
      <td>0.154841</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00cc97158e6f4cb8eac3c0075918b7ffi5k8o</td>
      <td>3</td>
      <td>0.390319</td>
      <td>0.130106</td>
      <td>0.389935</td>
      <td>0.225019</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01067975436d123f717ee5aba0dd4bbfa0937</td>
      <td>543</td>
      <td>23.723997</td>
      <td>0.043691</td>
      <td>0.096402</td>
      <td>0.025655</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0a7446b63f183a4928ae762ed6cd1c4b894qz</td>
      <td>836</td>
      <td>227.149872</td>
      <td>0.271710</td>
      <td>0.547788</td>
      <td>0.229140</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0a92c4fa2bcdc8e952546d236b322053o5wbk</td>
      <td>4</td>
      <td>0.110689</td>
      <td>0.027672</td>
      <td>0.038823</td>
      <td>0.018562</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0a9c5e1f261c937c09817ea2d5e949f34v8oe</td>
      <td>21</td>
      <td>0.914778</td>
      <td>0.043561</td>
      <td>0.074500</td>
      <td>0.025280</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0a9d614009139236f198ddb4759a498cen5ix</td>
      <td>4</td>
      <td>0.894338</td>
      <td>0.223585</td>
      <td>0.450313</td>
      <td>0.258143</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>5355</td>
      <td>144.450372</td>
      <td>0.026975</td>
      <td>0.059152</td>
      <td>0.017412</td>
      <td>21</td>
      <td>0.003922</td>
      <td>1</td>
      <td>0.062505</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>87 rows × 11 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ee7f4d73-4d8a-4348-842c-5b9a153aa2bc')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ee7f4d73-4d8a-4348-842c-5b9a153aa2bc button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ee7f4d73-4d8a-4348-842c-5b9a153aa2bc');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### visualisation


```python
r= 1 - (dd4.outcome==0)['<lambda>'].sum()/len(dd4) #poids à donner à 0
weights = r + (1-2*r)*dd4.outcome['<lambda>']
```

nombre de connexion par bidder_id


```python
sns.scatterplot(x=dd4.index, y=dd4.my_agg["sum"], hue=dd4.outcome['<lambda>'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f20b9d80160>




    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_69_1.png)
    



```python
sns.histplot(x=dd4.time["std"], hue=dd4.outcome['<lambda>'], kde=True, weights=get_weight(dd4.outcome['<lambda>']))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f20b9d678e0>




    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_70_1.png)
    



```python
sns.histplot(x=dd4.my_agg["std"], hue=dd4.outcome['<lambda>'], kde=True, weights=get_weight(dd4.outcome['<lambda>']))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f20b9c4d280>




    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_71_1.png)
    



```python
interesting_cols = [("my_agg","max"), ("my_agg","mean"),("my_agg","std"),("my_agg","sum")]
```


```python
_ = dd4[list(set(dd4.columns)-{("outcome","<lambda>"),("bidder_id","")})]
only_logit_runner(_, dd4.outcome['<lambda>'])
```

    Warning: Maximum number of iterations has been exceeded.
             Current function value: 0.473319
             Iterations: 35
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:               <lambda>   No. Observations:                   87
    Model:                          Logit   Df Residuals:                       78
    Method:                           MLE   Df Model:                            8
    Date:                Sun, 18 Dec 2022   Pseudo R-squ.:                 -0.8861
    Time:                        06:10:30   Log-Likelihood:                -41.179
    converged:                      False   LL-Null:                       -21.833
    Covariance Type:            nonrobust   LLR p-value:                     1.000
    ===============================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------
    my_agg_std   -217.3434    280.850     -0.774      0.439    -767.798     333.112
    time_max       19.0841     17.659      1.081      0.280     -15.526      53.695
    time_mean     -27.0729     18.520     -1.462      0.144     -63.372       9.226
    time_std      -46.1073     41.392     -1.114      0.265    -127.233      35.019
    my_agg_sum     -0.3850      0.458     -0.840      0.401      -1.283       0.513
    time_count      0.0004      0.001      0.313      0.755      -0.002       0.003
    time_sum        0.0020      0.003      0.734      0.463      -0.003       0.007
    my_agg_max      1.1692      2.618      0.447      0.655      -3.963       6.301
    my_agg_mean  5262.6638   6069.349      0.867      0.386   -6633.041    1.72e+04
    ===============================================================================
    scores: {'acc': 0.598, 'f1': 0.146, 'pres': 0.086, 'rec': 0.5, 'roc': 0.576}
    residuals: significantly normal



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_73_1.png)
    


    sorted pvalues
     time_mean      0.143794
    time_std       0.265309
    time_max       0.279821
    my_agg_mean    0.385893
    my_agg_sum     0.400651
    my_agg_std     0.439003
    time_sum       0.463124
    my_agg_max     0.655208
    time_count     0.754590
    dtype: float64



```python
(dd4.my_agg["max"]>=8).sum()/len(dd4)
```




    0.04597701149425287



pour chaque bidder_id, le (nb_device - 1) + (nb_ip - 1) + ....


```python
X_ = dd4.my_agg[["max","mean"]].fillna(0)
y_ = dd4.outcome['<lambda>']
only_logit_runner(X_, y_)
```

    Optimization terminated successfully.
             Current function value: 0.657416
             Iterations 12
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:               <lambda>   No. Observations:                   87
    Model:                          Logit   Df Residuals:                       85
    Method:                           MLE   Df Model:                            1
    Date:                Sun, 18 Dec 2022   Pseudo R-squ.:                  -1.620
    Time:                        06:10:31   Log-Likelihood:                -57.195
    converged:                       True   LL-Null:                       -21.833
    Covariance Type:            nonrobust   LLR p-value:                     1.000
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    max           -0.6720      1.019     -0.660      0.509      -2.669       1.325
    mean         -30.0709    392.498     -0.077      0.939    -799.353     739.212
    ==============================================================================
    scores: {'acc': 0.138, 'f1': 0.096, 'pres': 0.052, 'rec': 0.667, 'roc': 0.393}
    residuals: significantly normal



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_76_1.png)
    


    sorted pvalues
     max     0.509497
    mean    0.938931
    dtype: float64



```python
sns.displot(x=dd4.my_agg["mean"], hue=dd4.outcome['<lambda>'], weights=weights, kde=True)
```




    <seaborn.axisgrid.FacetGrid at 0x7f20b9dce0a0>




    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_77_1.png)
    



```python
dd4.outcome.columns
```




    Index(['<lambda>'], dtype='object')




```python
len(dd4[dd4.outcome['<lambda>']==0]), len(dd4[dd4.outcome['<lambda>']==1])
```




    (81, 6)




```python
cls_ =list(set(dd2.columns)-{'outcome','bidder_id'})
dd_ = dd2[(dd2[cls_]>1).sum(axis=1)!=0].outcome
len(dd_[dd_==1]) / len(dd_) #sns.histplot(dd_)
dd_
```




    1406     1
    1492     1
    1500     1
    1502     1
    1580     1
            ..
    79525    0
    84839    0
    85922    1
    92652    1
    92900    1
    Name: outcome, Length: 128, dtype: int64




```python
dd2.columns
```




    Index(['bidder_id', 'time', 'auction', 'device', 'country', 'ip', 'url',
           'outcome', 'my_agg'],
          dtype='object')




```python
cls_ =list(set(dd2.columns)-{'outcome','bidder_id','time'})
dd_ = (dd2[cls_]-1).sum(axis=1)
print(len(dd_[dd_!=0])/len(dd_))
dd_[dd_!=0]
```

    0.9926473608543849





    0       -6
    1       -6
    2       -6
    3       -6
    4       -6
            ..
    97375   -6
    97376   -6
    97377   -6
    97378   -6
    97379   -6
    Length: 96664, dtype: int64




```python
set(dd2.columns)-{'outcome','bidder_id','time'}
```




    {'auction', 'country', 'device', 'ip', 'my_agg', 'url'}




```python
dd2
```





  <div id="df-b93a98d9-5d22-45df-86a3-115396cc0a2f">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bidder_id</th>
      <th>time</th>
      <th>auction</th>
      <th>device</th>
      <th>country</th>
      <th>ip</th>
      <th>url</th>
      <th>outcome</th>
      <th>my_agg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>001068c415025a009fee375a12cff4fcnht8y</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0030a2dd87ad2733e0873062e4f83954mkj86</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>0.006280</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>0.007004</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>97375</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>0.059113</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97376</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>0.059130</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97377</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>0.059136</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97378</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>0.059143</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97379</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>0.059152</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>97380 rows × 9 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b93a98d9-5d22-45df-86a3-115396cc0a2f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-b93a98d9-5d22-45df-86a3-115396cc0a2f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b93a98d9-5d22-45df-86a3-115396cc0a2f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
nb_human = df[(df.outcome==0)].bidder_id.nunique()
nb_bot = df[(df.outcome==1)].bidder_id.nunique()

nb_human = len(dd2[(dd2.outcome==0)])
nb_bot = len(dd2[(dd2.outcome==1)])
nb_total = len(dd2)
```


```python
df.columns
```




    Index(['bidder_id', 'bid_id', 'auction', 'merchandise', 'device', 'time',
           'country', 'ip', 'url', 'payment_account', 'address', 'outcome'],
          dtype='object')




```python
'''for col_name in ["country", "url", "ip", "adresse"]
a, b = len(dd[(dd.outcome==0)&(dd.url>1)])/nb_human, len(dd[(dd.outcome==1)&(dd.url>1)])/nb_bot
c,d = len(dd[(dd.outcome==0)&(dd.url==1)])/nb_human, len(dd[(dd.outcome==1)&(dd.url==1)])/nb_bot
print(a,b)
print(c,d)'''
```




    'for col_name in ["country", "url", "ip", "adresse"]\na, b = len(dd[(dd.outcome==0)&(dd.url>1)])/nb_human, len(dd[(dd.outcome==1)&(dd.url>1)])/nb_bot\nc,d = len(dd[(dd.outcome==0)&(dd.url==1)])/nb_human, len(dd[(dd.outcome==1)&(dd.url==1)])/nb_bot\nprint(a,b)\nprint(c,d)'




```python
nb_human, nb_bot
```




    (88710, 8670)




```python

```


```python

```


```python
len(dd2[dd2.url==1]), len(dd2[dd2.url>1])
```




    (1649, 51)




```python

```

## remove some columns


```python
df.bidder_id.nunique()
```




    87




```python
if "bid_id" in df.columns: del df["bid_id"]
if "address" in df.columns: del df["address"]
if "payment_account" in df.columns: del df["payment_account"]
df.columns
```




    Index(['bidder_id', 'auction', 'merchandise', 'device', 'time', 'country',
           'ip', 'url', 'outcome'],
          dtype='object')



## merchandise

### plotting


```python
for elt in df.merchandise.unique():
  print(f"{elt} {df[df.merchandise==elt].outcome.unique()}")

sns.scatterplot(x="merchandise", y="outcome", data=df, hue="outcome")
```

    jewelry [0]
    mobile [0 1]
    books and music [0]
    office equipment [0]
    sporting goods [0 1]
    home goods [0 1]





    <matplotlib.axes._subplots.AxesSubplot at 0x7f20bfbe5790>




    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_98_2.png)
    



```python
sns.barplot(x="merchandise", y="outcome", data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f20c479ea90>




    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_99_1.png)
    



```python
df.groupby("merchandise").merchandise.count().to_list()
```




    [227, 15095, 25004, 33081, 21, 26572]




```python
'''_ = df.groupby("merchandise").merchandise.count()
sns.catplot(
    data=df, kind="bar",
    x="merchandise", y=_, hue="outcome",
    errorbar="sd", palette="dark", alpha=.6, height=6
)'''
```




    '_ = df.groupby("merchandise").merchandise.count()\nsns.catplot(\n    data=df, kind="bar",\n    x="merchandise", y=_, hue="outcome",\n    errorbar="sd", palette="dark", alpha=.6, height=6\n)'



- seem like whenever merchandise is jewelry or "books and music" or "office equipment", the outcome is always 0
- but that may be a trap

### selection and one-hot-encoding


```python
_ = ''
```


```python
# histogrammes des produits
ax = sns.histplot(x="merchandise", data=df, color='b')
add_labels_to_histplot(ax, title="Distribution of merchandises")
```


    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_105_0.png)
    



```python
# histogramme des produits par outcome
ax = sns.histplot(x="merchandise", data=df, hue="outcome", palette=["b","orange"])
add_labels_to_histplot(ax, title="Distribution of merchandises by outcome")
```


    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_106_0.png)
    



```python
# liste des outcome par produit
def human(x): return (x==0).sum()
def bot(x): return (x==1).sum()
df.groupby("merchandise").agg({"outcome": [human, bot] })
```





  <div id="df-0fd06eb4-fda3-459f-a82e-b9f16ebaf9de">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">outcome</th>
    </tr>
    <tr>
      <th></th>
      <th>human</th>
      <th>bot</th>
    </tr>
    <tr>
      <th>merchandise</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>books and music</th>
      <td>227</td>
      <td>0</td>
    </tr>
    <tr>
      <th>home goods</th>
      <td>13002</td>
      <td>2093</td>
    </tr>
    <tr>
      <th>jewelry</th>
      <td>25004</td>
      <td>0</td>
    </tr>
    <tr>
      <th>mobile</th>
      <td>26228</td>
      <td>6853</td>
    </tr>
    <tr>
      <th>office equipment</th>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sporting goods</th>
      <td>26395</td>
      <td>177</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0fd06eb4-fda3-459f-a82e-b9f16ebaf9de')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0fd06eb4-fda3-459f-a82e-b9f16ebaf9de button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0fd06eb4-fda3-459f-a82e-b9f16ebaf9de');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df.merchandise.unique()
```




    array(['jewelry', 'mobile', 'books and music', 'office equipment',
           'sporting goods', 'home goods'], dtype=object)




```python
REMOVE_EVIDENT_MERCHANDISE = False
```


```python
if REMOVE_EVIDENT_MERCHANDISE : df = df[df.merchandise.apply(lambda x: x in ["mobile", "sporting goods", "home goods"])]
else: df["non_robot_merchandise"] = df.merchandise.apply(lambda x: int(x in ['jewelry', 'books and music', 'office equipment']))
```


```python
#
```


```python
df2, new_cols = one_hot_encoder_v2(df, "merchandise")
df2
```





  <div id="df-0fabecdc-ee02-4a97-b8fd-49ae7752d838">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bidder_id</th>
      <th>auction</th>
      <th>device</th>
      <th>time</th>
      <th>country</th>
      <th>ip</th>
      <th>url</th>
      <th>outcome</th>
      <th>non_robot_merchandise</th>
      <th>merchandise_1</th>
      <th>merchandise_2</th>
      <th>merchandise_3</th>
      <th>merchandise_4</th>
      <th>merchandise_5</th>
      <th>merchandise_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>001068c415025a009fee375a12cff4fcnht8y</td>
      <td>4ifac</td>
      <td>phone561</td>
      <td>0.527973</td>
      <td>bn</td>
      <td>139.226</td>
      <td>vasstdc27m7nks3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0030a2dd87ad2733e0873062e4f83954mkj86</td>
      <td>obbny</td>
      <td>phone313</td>
      <td>0.515267</td>
      <td>ir</td>
      <td>21.67</td>
      <td>vnw40k8zzokijsv</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>l3o6q</td>
      <td>phone451</td>
      <td>0.002705</td>
      <td>bh</td>
      <td>103.165</td>
      <td>kk7rxe25ehseyci</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>du967</td>
      <td>phone117</td>
      <td>0.008986</td>
      <td>tr</td>
      <td>239.250</td>
      <td>iu2iu3k137vakme</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>wx3kf</td>
      <td>phone16</td>
      <td>0.009710</td>
      <td>in</td>
      <td>255.108</td>
      <td>u85yj2e7owkz6xp</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>toxfq</td>
      <td>phone1036</td>
      <td>0.962363</td>
      <td>in</td>
      <td>186.94</td>
      <td>vasstdc27m7nks3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>ucb4u</td>
      <td>phone127</td>
      <td>0.962380</td>
      <td>in</td>
      <td>119.27</td>
      <td>vasstdc27m7nks3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>sg8yd</td>
      <td>phone383</td>
      <td>0.962386</td>
      <td>in</td>
      <td>243.25</td>
      <td>yweo7wfejrgbi2d</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>toaj7</td>
      <td>phone26</td>
      <td>0.962393</td>
      <td>in</td>
      <td>17.66</td>
      <td>4dd8ei0o5oqsua3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>07axb</td>
      <td>phone25</td>
      <td>0.962401</td>
      <td>in</td>
      <td>64.30</td>
      <td>8zdkeqk4yby6lz2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 15 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0fabecdc-ee02-4a97-b8fd-49ae7752d838')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0fabecdc-ee02-4a97-b8fd-49ae7752d838 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0fabecdc-ee02-4a97-b8fd-49ae7752d838');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df1_backup = df.copy()
df = df2.copy()
del df2
```

### country


```python
#sns.barplot(x="country", y="outcome", data=df)
```

# Méthode 1: Agregations on bidder_id
- no unique on merchandise because of unicite/bidder_id

## agregation


```python
df.columns
```




    Index(['bidder_id', 'auction', 'device', 'time', 'country', 'ip', 'url',
           'outcome', 'non_robot_merchandise', 'merchandise_1', 'merchandise_2',
           'merchandise_3', 'merchandise_4', 'merchandise_5', 'merchandise_6'],
          dtype='object')




```python

```


```python
def form_df_agg(filename):
    print("ADD_LEN_TO_GROUPBY = ",ADD_LEN_TO_GROUPBY)
    dd_rename = {col_name: f"nb_{col_name}" for col_name in new_cols}
    dd_rename.update({"bidder_id":"nb_bid", "ip": "nb_ip","auction": "nb_auction","country": "nb_country","device": "nb_device","url": "nb_url"})

    dd = { "ip": lambda x:x.nunique(),
                "auction": lambda x:x.nunique()/len(x) if ADD_LEN_TO_GROUPBY else x.nunique(),
                "device": lambda x:x.nunique()/len(x) if ADD_LEN_TO_GROUPBY else x.nunique(),
                "url": lambda x:x.nunique()/len(x) if ADD_LEN_TO_GROUPBY else x.nunique(),
                "country":lambda x: x.nunique()/len(x) if ADD_LEN_TO_GROUPBY else x.nunique(),
                "outcome":lambda x: x.unique()[0],
                "non_robot_merchandise":lambda x: x.unique()[0],
                "time":lambda x: np.mean(np.diff(np.sort(x))),
                "bidder_id": lambda x: len(x),
                }
    dd.update({col_name: lambda x: x.sum()/len(x) if ADD_LEN_TO_GROUPBY else x.sum() for col_name in new_cols})
    sss = df.groupby("bidder_id").agg(dd).rename(dd_rename, axis=1)
    sss.loc[sss.time.isnull(), "time"] = sss.time.mean()
    for col_name in sss.columns:
      sss[col_name] = (sss[col_name] - sss[col_name].min()) / (sss[col_name].max() - sss[col_name].min())

    sss.to_csv(filename)

    return sss
```


```python

'''filename = f"df_agg_{FILE_VERSION}_{'2' if ADD_LEN_TO_GROUPBY else '1'}.csv"
if not os.path.exists(filename):
  print("running")
  sss = form_df_agg(filename)
else:
  print("found")
  sss = pd.read_csv(filename)'''
```




    'filename = f"df_agg_{FILE_VERSION}_{\'2\' if ADD_LEN_TO_GROUPBY else \'1\'}.csv" \nif not os.path.exists(filename):\n  print("running")\n  sss = form_df_agg(filename)\nelse:\n  print("found")\n  sss = pd.read_csv(filename)'




```python
l_sss = []
for i in range(2):
  ADD_LEN_TO_GROUPBY = bool(i)
  filename = f"df_agg_{FILE_VERSION}_{'2' if ADD_LEN_TO_GROUPBY else '1'}.csv"
  if not os.path.exists(filename):
    print(f"running for {filename}")
    sss = form_df_agg(filename)
  else:
    print(f"{filename} found")
    sss = pd.read_csv(filename)
  l_sss.append(sss)

sss = pd.merge(left=l_sss[0], right=l_sss[1].drop(['outcome','non_robot_merchandise','time'], axis=1), on="bidder_id")
sss
```

    df_agg_v7_1.csv found
    running for df_agg_v7_2.csv
    ADD_LEN_TO_GROUPBY =  True






  <div id="df-711359d7-950a-41a8-925d-23249fc035ad">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bidder_id</th>
      <th>nb_ip_x</th>
      <th>nb_auction_x</th>
      <th>nb_device_x</th>
      <th>nb_url_x</th>
      <th>nb_country_x</th>
      <th>outcome</th>
      <th>non_robot_merchandise</th>
      <th>time</th>
      <th>nb_bid_x</th>
      <th>...</th>
      <th>nb_device_y</th>
      <th>nb_url_y</th>
      <th>nb_country_y</th>
      <th>nb_bid_y</th>
      <th>nb_merchandise_1_y</th>
      <th>nb_merchandise_2_y</th>
      <th>nb_merchandise_3_y</th>
      <th>nb_merchandise_4_y</th>
      <th>nb_merchandise_5_y</th>
      <th>nb_merchandise_6_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>001068c415025a009fee375a12cff4fcnht8y</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.089407</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0030a2dd87ad2733e0873062e4f83954mkj86</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.089407</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00a0517965f18610417ee784a05f494d4dw6e</td>
      <td>0.008946</td>
      <td>0.123519</td>
      <td>0.071817</td>
      <td>0.012561</td>
      <td>0.117188</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.019960</td>
      <td>0.006302</td>
      <td>...</td>
      <td>0.472195</td>
      <td>0.595318</td>
      <td>0.110372</td>
      <td>0.006302</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00cc97158e6f4cb8eac3c0075918b7ffi5k8o</td>
      <td>0.000175</td>
      <td>0.003384</td>
      <td>0.002176</td>
      <td>0.000000</td>
      <td>0.007812</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.000090</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.332630</td>
      <td>0.665500</td>
      <td>0.000090</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01067975436d123f717ee5aba0dd4bbfa0937</td>
      <td>0.034818</td>
      <td>0.027073</td>
      <td>0.178455</td>
      <td>0.000454</td>
      <td>0.554688</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000909</td>
      <td>0.024397</td>
      <td>...</td>
      <td>0.299912</td>
      <td>0.006320</td>
      <td>0.129561</td>
      <td>0.024397</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0a7446b63f183a4928ae762ed6cd1c4b894qz</td>
      <td>0.053236</td>
      <td>0.439932</td>
      <td>0.223069</td>
      <td>0.038741</td>
      <td>0.359375</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.003361</td>
      <td>0.037586</td>
      <td>...</td>
      <td>0.242130</td>
      <td>0.306686</td>
      <td>0.052917</td>
      <td>0.037586</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0a92c4fa2bcdc8e952546d236b322053o5wbk</td>
      <td>0.000263</td>
      <td>0.005076</td>
      <td>0.003264</td>
      <td>0.000303</td>
      <td>0.007812</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.066372</td>
      <td>0.000135</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.749736</td>
      <td>0.498250</td>
      <td>0.000135</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0a9c5e1f261c937c09817ea2d5e949f34v8oe</td>
      <td>0.001579</td>
      <td>0.010152</td>
      <td>0.004353</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.019102</td>
      <td>0.000900</td>
      <td>...</td>
      <td>0.233766</td>
      <td>0.046615</td>
      <td>0.235428</td>
      <td>0.000900</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0a9d614009139236f198ddb4759a498cen5ix</td>
      <td>0.000263</td>
      <td>0.005076</td>
      <td>0.002176</td>
      <td>0.000303</td>
      <td>0.007812</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.769893</td>
      <td>0.000135</td>
      <td>...</td>
      <td>0.748580</td>
      <td>0.749736</td>
      <td>0.498250</td>
      <td>0.000135</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0ad17aa9111f657d71cd3005599afc24fd44y</td>
      <td>0.352482</td>
      <td>0.778342</td>
      <td>0.510337</td>
      <td>0.196429</td>
      <td>0.625000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000052</td>
      <td>0.244283</td>
      <td>...</td>
      <td>0.081398</td>
      <td>0.238513</td>
      <td>0.011475</td>
      <td>0.244283</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>87 rows × 28 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-711359d7-950a-41a8-925d-23249fc035ad')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-711359d7-950a-41a8-925d-23249fc035ad button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-711359d7-950a-41a8-925d-23249fc035ad');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## simple preview


```python
sss_backup = sss.copy()
```


```python
sss.outcome.unique()
```




    array([0., 1.])




```python
sss.columns
```




    Index(['bidder_id', 'nb_ip_x', 'nb_auction_x', 'nb_device_x', 'nb_url_x',
           'nb_country_x', 'outcome', 'non_robot_merchandise', 'time', 'nb_bid_x',
           'nb_merchandise_1_x', 'nb_merchandise_2_x', 'nb_merchandise_3_x',
           'nb_merchandise_4_x', 'nb_merchandise_5_x', 'nb_merchandise_6_x',
           'nb_ip_y', 'nb_auction_y', 'nb_device_y', 'nb_url_y', 'nb_country_y',
           'nb_bid_y', 'nb_merchandise_1_y', 'nb_merchandise_2_y',
           'nb_merchandise_3_y', 'nb_merchandise_4_y', 'nb_merchandise_5_y',
           'nb_merchandise_6_y'],
          dtype='object')




```python
print(sss.shape)
dd4_ = pd.DataFrame({f"{a}_{b}" if b else a: dd4[(a,b)] for (a,b) in  dd4[interesting_cols+[("bidder_id","")]]})
sss = pd.merge(left = sss, right=dd4_, on="bidder_id")
del sss["bidder_id"]
print(sss.shape)
```

    (87, 28)
    (87, 31)



```python
get_cols_info(sss)
```

    >>> df.shape=  (87, 31)
    
    >>> df.info= 
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 87 entries, 0 to 86
    Data columns (total 31 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   nb_ip_x                87 non-null     float64
     1   nb_auction_x           87 non-null     float64
     2   nb_device_x            87 non-null     float64
     3   nb_url_x               87 non-null     float64
     4   nb_country_x           87 non-null     float64
     5   outcome                87 non-null     float64
     6   non_robot_merchandise  87 non-null     float64
     7   time                   87 non-null     float64
     8   nb_bid_x               87 non-null     float64
     9   nb_merchandise_1_x     87 non-null     float64
     10  nb_merchandise_2_x     87 non-null     float64
     11  nb_merchandise_3_x     87 non-null     float64
     12  nb_merchandise_4_x     87 non-null     float64
     13  nb_merchandise_5_x     87 non-null     float64
     14  nb_merchandise_6_x     87 non-null     float64
     15  nb_ip_y                87 non-null     float64
     16  nb_auction_y           87 non-null     float64
     17  nb_device_y            87 non-null     float64
     18  nb_url_y               87 non-null     float64
     19  nb_country_y           87 non-null     float64
     20  nb_bid_y               87 non-null     float64
     21  nb_merchandise_1_y     87 non-null     float64
     22  nb_merchandise_2_y     87 non-null     float64
     23  nb_merchandise_3_y     87 non-null     float64
     24  nb_merchandise_4_y     87 non-null     float64
     25  nb_merchandise_5_y     87 non-null     float64
     26  nb_merchandise_6_y     87 non-null     float64
     27  my_agg_max             87 non-null     int64  
     28  my_agg_mean            87 non-null     float64
     29  my_agg_std             87 non-null     float64
     30  my_agg_sum             87 non-null     int64  
    dtypes: float64(29), int64(2)
    memory usage: 21.8 KB
    
    >>> df.more_info= 
                             dtype  nunique  nunique(%) nunique_per_bid>1(%)  \
    nb_merchandise_1_y     float64        2         2.3                        
    nb_merchandise_3_y     float64        2         2.3                        
    nb_merchandise_4_y     float64        2         2.3                        
    nb_merchandise_5_y     float64        2         2.3                        
    nb_merchandise_6_y     float64        2         2.3                        
    outcome                float64        2         2.3                        
    non_robot_merchandise  float64        2         2.3                        
    nb_merchandise_2_y     float64        2         2.3                        
    nb_merchandise_3_x     float64        4         4.6                        
    my_agg_max               int64        5         5.8                        
    nb_merchandise_4_x     float64        5         5.8                        
    nb_merchandise_6_x     float64        8         9.2                        
    my_agg_sum               int64        9        10.4                        
    my_agg_mean            float64       11        12.7                        
    my_agg_std             float64       11        12.7                        
    nb_merchandise_1_x     float64       13        15.0                        
    nb_merchandise_5_x     float64       17        19.6                        
    nb_merchandise_2_x     float64       28        32.2                        
    nb_country_x           float64       29        33.4                        
    nb_url_x               float64       35        40.3                        
    nb_device_x            float64       43        49.5                        
    nb_auction_x           float64       45        51.8                        
    nb_ip_x                float64       49        56.4                        
    nb_ip_y                float64       49        56.4                        
    nb_country_y           float64       53        61.0                        
    nb_auction_y           float64       53        61.0                        
    nb_device_y            float64       54        62.1                        
    nb_url_y               float64       56        64.4                        
    nb_bid_x               float64       56        64.4                        
    nb_bid_y               float64       56        64.4                        
    time                   float64       75        86.3                        
    
                           is_cat  
    nb_merchandise_1_y          1  
    nb_merchandise_3_y          1  
    nb_merchandise_4_y          1  
    nb_merchandise_5_y          1  
    nb_merchandise_6_y          1  
    outcome                     1  
    non_robot_merchandise       1  
    nb_merchandise_2_y          1  
    nb_merchandise_3_x          1  
    my_agg_max                  1  
    nb_merchandise_4_x          1  
    nb_merchandise_6_x          1  
    my_agg_sum                  1  
    my_agg_mean                 0  
    my_agg_std                  0  
    nb_merchandise_1_x          0  
    nb_merchandise_5_x          0  
    nb_merchandise_2_x          0  
    nb_country_x                0  
    nb_url_x                    0  
    nb_device_x                 0  
    nb_auction_x                0  
    nb_ip_x                     0  
    nb_ip_y                     0  
    nb_country_y                0  
    nb_auction_y                0  
    nb_device_y                 0  
    nb_url_y                    0  
    nb_bid_x                    0  
    nb_bid_y                    0  
    time                        0  
    
    >>> df.describe= 
             nb_ip_x  nb_auction_x  nb_device_x   nb_url_x  nb_country_x  \
    count  87.000000     87.000000    87.000000  87.000000     87.000000   
    mean    0.051221      0.104226     0.088377   0.038231      0.109914   
    std     0.160340      0.206158     0.203158   0.152145      0.219148   
    min     0.000000      0.000000     0.000000   0.000000      0.000000   
    25%     0.000175      0.001692     0.001088   0.000000      0.000000   
    50%     0.001052      0.013536     0.006529   0.000454      0.015625   
    75%     0.010612      0.093909     0.049510   0.004086      0.078125   
    max     1.000000      1.000000     1.000000   1.000000      1.000000   
    
             outcome  non_robot_merchandise       time   nb_bid_x  \
    count  87.000000              87.000000  87.000000  87.000000   
    mean    0.068966               0.241379   0.089407   0.051694   
    std     0.254864               0.430400   0.193772   0.160625   
    min     0.000000               0.000000   0.000000   0.000000   
    25%     0.000000               0.000000   0.003469   0.000090   
    50%     0.000000               0.000000   0.025219   0.000765   
    75%     0.000000               0.000000   0.089407   0.009363   
    max     1.000000               1.000000   1.000000   1.000000   
    
           nb_merchandise_1_x  ...  nb_merchandise_1_y  nb_merchandise_2_y  \
    count           87.000000  ...           87.000000           87.000000   
    mean             0.012936  ...            0.149425            0.459770   
    std              0.107333  ...            0.358574            0.501268   
    min              0.000000  ...            0.000000            0.000000   
    25%              0.000000  ...            0.000000            0.000000   
    50%              0.000000  ...            0.000000            0.000000   
    75%              0.000000  ...            0.000000            1.000000   
    max              1.000000  ...            1.000000            1.000000   
    
           nb_merchandise_3_y  nb_merchandise_4_y  nb_merchandise_5_y  \
    count           87.000000           87.000000           87.000000   
    mean             0.034483            0.057471            0.218391   
    std              0.183523            0.234090            0.415549   
    min              0.000000            0.000000            0.000000   
    25%              0.000000            0.000000            0.000000   
    50%              0.000000            0.000000            0.000000   
    75%              0.000000            0.000000            0.000000   
    max              1.000000            1.000000            1.000000   
    
           nb_merchandise_6_y  my_agg_max  my_agg_mean  my_agg_std  my_agg_sum  
    count           87.000000   87.000000    87.000000   87.000000   87.000000  
    mean             0.080460    1.241379     0.001332    0.022730   15.850575  
    std              0.273581    6.768541     0.007910    0.125534   98.310676  
    min              0.000000    0.000000     0.000000    0.000000    0.000000  
    25%              0.000000    0.000000     0.000000    0.000000    0.000000  
    50%              0.000000    0.000000     0.000000    0.000000    0.000000  
    75%              0.000000    0.000000     0.000000    0.000000    0.000000  
    max              1.000000   54.000000     0.069002    1.092466  857.000000  
    
    [8 rows x 31 columns]



```python
sss[sss.outcome==1].describe()
```





  <div id="df-97b7ccc6-a5d1-4834-bc8d-0c50f1ed8f72">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nb_ip_x</th>
      <th>nb_auction_x</th>
      <th>nb_device_x</th>
      <th>nb_url_x</th>
      <th>nb_country_x</th>
      <th>outcome</th>
      <th>non_robot_merchandise</th>
      <th>time</th>
      <th>nb_bid_x</th>
      <th>nb_merchandise_1_x</th>
      <th>...</th>
      <th>nb_merchandise_1_y</th>
      <th>nb_merchandise_2_y</th>
      <th>nb_merchandise_3_y</th>
      <th>nb_merchandise_4_y</th>
      <th>nb_merchandise_5_y</th>
      <th>nb_merchandise_6_y</th>
      <th>my_agg_max</th>
      <th>my_agg_mean</th>
      <th>my_agg_std</th>
      <th>my_agg_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.0</td>
      <td>...</td>
      <td>6.0</td>
      <td>6.000000</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.068351</td>
      <td>0.200226</td>
      <td>0.093580</td>
      <td>0.033016</td>
      <td>0.135417</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.001988</td>
      <td>0.068397</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000759</td>
      <td>0.014618</td>
      <td>3.666667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.139486</td>
      <td>0.290020</td>
      <td>0.204311</td>
      <td>0.080056</td>
      <td>0.241944</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.002534</td>
      <td>0.091009</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.547723</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.408248</td>
      <td>0.516398</td>
      <td>0.516398</td>
      <td>0.001570</td>
      <td>0.025534</td>
      <td>8.500980</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000702</td>
      <td>0.008460</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.007922</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.005394</td>
      <td>0.044416</td>
      <td>0.005985</td>
      <td>0.000189</td>
      <td>0.023438</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000095</td>
      <td>0.011366</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.014427</td>
      <td>0.101523</td>
      <td>0.013058</td>
      <td>0.000454</td>
      <td>0.035156</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.001267</td>
      <td>0.032026</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.022869</td>
      <td>0.164975</td>
      <td>0.019314</td>
      <td>0.000605</td>
      <td>0.082031</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.002700</td>
      <td>0.075239</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.750000</td>
      <td>0.750000</td>
      <td>0.000476</td>
      <td>0.018904</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.352482</td>
      <td>0.778342</td>
      <td>0.510337</td>
      <td>0.196429</td>
      <td>0.625000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.006513</td>
      <td>0.244283</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.003922</td>
      <td>0.062505</td>
      <td>21.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-97b7ccc6-a5d1-4834-bc8d-0c50f1ed8f72')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-97b7ccc6-a5d1-4834-bc8d-0c50f1ed8f72 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-97b7ccc6-a5d1-4834-bc8d-0c50f1ed8f72');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
sss[sss.outcome==0].describe()
```





  <div id="df-f00cddca-adcf-451a-8056-68a62e1d2043">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nb_ip_x</th>
      <th>nb_auction_x</th>
      <th>nb_device_x</th>
      <th>nb_url_x</th>
      <th>nb_country_x</th>
      <th>outcome</th>
      <th>non_robot_merchandise</th>
      <th>time</th>
      <th>nb_bid_x</th>
      <th>nb_merchandise_1_x</th>
      <th>...</th>
      <th>nb_merchandise_1_y</th>
      <th>nb_merchandise_2_y</th>
      <th>nb_merchandise_3_y</th>
      <th>nb_merchandise_4_y</th>
      <th>nb_merchandise_5_y</th>
      <th>nb_merchandise_6_y</th>
      <th>my_agg_max</th>
      <th>my_agg_mean</th>
      <th>my_agg_std</th>
      <th>my_agg_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.0</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>...</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
      <td>81.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.049952</td>
      <td>0.097115</td>
      <td>0.087992</td>
      <td>0.038618</td>
      <td>0.108025</td>
      <td>0.0</td>
      <td>0.259259</td>
      <td>0.095882</td>
      <td>0.050456</td>
      <td>0.013894</td>
      <td>...</td>
      <td>0.160494</td>
      <td>0.456790</td>
      <td>0.037037</td>
      <td>0.061728</td>
      <td>0.222222</td>
      <td>0.061728</td>
      <td>1.308642</td>
      <td>0.001374</td>
      <td>0.023330</td>
      <td>16.753086</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.162472</td>
      <td>0.199222</td>
      <td>0.204346</td>
      <td>0.156465</td>
      <td>0.218899</td>
      <td>0.0</td>
      <td>0.440959</td>
      <td>0.199368</td>
      <td>0.164910</td>
      <td>0.111224</td>
      <td>...</td>
      <td>0.369350</td>
      <td>0.501233</td>
      <td>0.190029</td>
      <td>0.242161</td>
      <td>0.418330</td>
      <td>0.242161</td>
      <td>7.011851</td>
      <td>0.008191</td>
      <td>0.129979</td>
      <td>101.849832</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000034</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000175</td>
      <td>0.001692</td>
      <td>0.001088</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.007316</td>
      <td>0.000090</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000877</td>
      <td>0.013536</td>
      <td>0.004353</td>
      <td>0.000454</td>
      <td>0.015625</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.029591</td>
      <td>0.000585</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.007981</td>
      <td>0.071066</td>
      <td>0.051143</td>
      <td>0.004086</td>
      <td>0.078125</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.089407</td>
      <td>0.006302</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>54.000000</td>
      <td>0.069002</td>
      <td>1.092466</td>
      <td>857.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f00cddca-adcf-451a-8056-68a62e1d2043')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f00cddca-adcf-451a-8056-68a62e1d2043 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f00cddca-adcf-451a-8056-68a62e1d2043');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
tg = "outcome" #("outcome", "<lambda>")
sss[tg] = sss[tg].values.astype(int)
ax = sns.histplot(x=tg, data=sss)
add_labels_to_histplot(ax, title="distribution of outcome")
```


    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_131_0.png)
    



```python
def aff_hist(sss):
  for col_name in sss.columns:
    if col_name=="outcome": continue
    sns.histplot(x=col_name, data=sss, hue="outcome")
    plt.show()

#aff_hist(sss)
```


```python
print(df.groupby("outcome").time.mean())
print(df.groupby("outcome").time.std())
```

    outcome
    0    0.525292
    1    0.725660
    Name: time, dtype: float64
    outcome
    0    0.364287
    1    0.286109
    Name: time, dtype: float64



```python
for col_name in sss.columns:
  sss[col_name] = (sss[col_name] - sss[col_name].min()) / (sss[col_name].max() - sss[col_name].min())
```


```python
for elt in sss.columns:
  print(elt, sss[elt].isnull().any())
```

    nb_ip_x False
    nb_auction_x False
    nb_device_x False
    nb_url_x False
    nb_country_x False
    outcome False
    non_robot_merchandise False
    time False
    nb_bid_x False
    nb_merchandise_1_x False
    nb_merchandise_2_x False
    nb_merchandise_3_x False
    nb_merchandise_4_x False
    nb_merchandise_5_x False
    nb_merchandise_6_x False
    nb_ip_y False
    nb_auction_y False
    nb_device_y False
    nb_url_y False
    nb_country_y False
    nb_bid_y False
    nb_merchandise_1_y False
    nb_merchandise_2_y False
    nb_merchandise_3_y False
    nb_merchandise_4_y False
    nb_merchandise_5_y False
    nb_merchandise_6_y False
    my_agg_max False
    my_agg_mean False
    my_agg_std False
    my_agg_sum False



```python
sns.histplot(x="outcome", data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f20c0f86d90>




    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_136_1.png)
    



```python
sss_resampled,y = oversampling_(sss, sss.outcome, method="duplic")
sss_resampled.outcome = y
```


```python
sss_resampled_sm,y = oversampling_(sss, sss.outcome, method="smote")
sss_resampled_sm.outcome = y
```


```python
sns.histplot(x="nb_bid", hue="outcome", data=sss_resampled, kde=True)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-1023-b15ef753e424> in <module>
    ----> 1 sns.histplot(x="nb_bid", hue="outcome", data=sss_resampled, kde=True)
    

    /usr/local/lib/python3.8/dist-packages/seaborn/distributions.py in histplot(data, x, y, hue, weights, stat, bins, binwidth, binrange, discrete, cumulative, common_bins, common_norm, multiple, element, fill, shrink, kde, kde_kws, line_kws, thresh, pthresh, pmax, cbar, cbar_ax, cbar_kws, palette, hue_order, hue_norm, color, log_scale, legend, ax, **kwargs)
       1428 ):
       1429 
    -> 1430     p = _DistributionPlotter(
       1431         data=data,
       1432         variables=_DistributionPlotter.get_semantics(locals())


    /usr/local/lib/python3.8/dist-packages/seaborn/distributions.py in __init__(self, data, variables)
        109     ):
        110 
    --> 111         super().__init__(data=data, variables=variables)
        112 
        113     @property


    /usr/local/lib/python3.8/dist-packages/seaborn/_core.py in __init__(self, data, variables)
        603     def __init__(self, data=None, variables={}):
        604 
    --> 605         self.assign_variables(data, variables)
        606 
        607         for var, cls in self._semantic_mappings.items():


    /usr/local/lib/python3.8/dist-packages/seaborn/_core.py in assign_variables(self, data, variables)
        666         else:
        667             self.input_format = "long"
    --> 668             plot_data, variables = self._assign_variables_longform(
        669                 data, **variables,
        670             )


    /usr/local/lib/python3.8/dist-packages/seaborn/_core.py in _assign_variables_longform(self, data, **kwargs)
        901 
        902                 err = f"Could not interpret value `{val}` for parameter `{key}`"
    --> 903                 raise ValueError(err)
        904 
        905             else:


    ValueError: Could not interpret value `nb_bid` for parameter `x`



```python
sns.histplot(x="nb_bid", hue="outcome", data=sss, weights=get_weight(sss.outcome), kde=True)
```

replace nb_bid with nb_bide_at_the_same_time


```python
sss[sss.time.isnull()]
```

## test statistics


```python
sss
```


```python
from scipy.stats import ttest_ind
def run_ttest_ind(sss):
  for col_name in sss.columns:
      if col_name in ["outcome"]: continue
      print(col_name, end=": ")
      #t-test
      #l1 = l1[l1.isnull().apply(lambda x: not x)]
      #l2 = l2[l2.isnull().apply(lambda x: not x)]
      #if sss[col_name].nunique()>2:
      a,b = ttest_ind(sss[sss.outcome==1][col_name], sss[sss.outcome==0][col_name], equal_var=False)
      print(f"p-val={b:.2f} -->", "means seem different with a good p-value" if b<0.05 else "no enough evidence")

run_ttest_ind(sss)
```

    nb_ip_x: p-val=0.77 --> no enough evidence
    nb_auction_x: p-val=0.43 --> no enough evidence
    nb_device_x: p-val=0.95 --> no enough evidence
    nb_url_x: p-val=0.88 --> no enough evidence
    nb_country_x: p-val=0.80 --> no enough evidence
    non_robot_merchandise: p-val=0.00 --> means seem different with a good p-value
    time: p-val=0.00 --> means seem different with a good p-value
    nb_bid_x: p-val=0.68 --> no enough evidence
    nb_merchandise_1_x: p-val=0.26 --> no enough evidence
    nb_merchandise_2_x: p-val=0.40 --> no enough evidence
    nb_merchandise_3_x: p-val=0.17 --> no enough evidence
    nb_merchandise_4_x: p-val=0.06 --> no enough evidence
    nb_merchandise_5_x: p-val=0.17 --> no enough evidence
    nb_merchandise_6_x: p-val=0.61 --> no enough evidence
    nb_ip_y: p-val=0.77 --> no enough evidence
    nb_auction_y: p-val=0.00 --> means seem different with a good p-value
    nb_device_y: p-val=0.00 --> means seem different with a good p-value
    nb_url_y: p-val=0.00 --> means seem different with a good p-value
    nb_country_y: p-val=0.00 --> means seem different with a good p-value
    nb_bid_y: p-val=0.68 --> no enough evidence
    nb_merchandise_1_y: p-val=0.00 --> means seem different with a good p-value
    nb_merchandise_2_y: p-val=0.86 --> no enough evidence
    nb_merchandise_3_y: p-val=0.08 --> no enough evidence
    nb_merchandise_4_y: p-val=0.02 --> means seem different with a good p-value
    nb_merchandise_5_y: p-val=0.76 --> no enough evidence
    nb_merchandise_6_y: p-val=0.26 --> no enough evidence
    my_agg_max: p-val=0.23 --> no enough evidence
    my_agg_mean: p-val=0.58 --> no enough evidence
    my_agg_std: p-val=0.63 --> no enough evidence
    my_agg_sum: p-val=0.27 --> no enough evidence



```python
sss.columns
```


```python
sss.isnull().sum()
```


```python
cols = set(sss.columns) - {"outcome"}
cols = ['time', 'nb_merchandise_1', 'nb_merchandise_2', 'nb_merchandise_3', 'nb_ip',
       'nb_auction', 'nb_device'] #'nb_url', 'nb_country']
cols = ['time', 'nb_merchandise_1', 'nb_merchandise_2', 'nb_merchandise_3', 'nb_bid']
cols = ['non_robot_merchandise', 'time', 'nb_device', 'nb_ip', 'nb_country']
cols = ['non_robot_merchandise', 'time', 'nb_device', 'nb_ip', 'nb_country', 'my_agg_max', 'my_agg_mean',  'my_agg_std']
only_logit_runner(sss[cols], sss["outcome"])

'''y_test_proba = log_reg.predict(sss[cols])
y_train_proba = log_reg.predict(sss[cols])
y_train_pred = np.where(y_train_proba>0.4, 1, 0)
y_test_pred = np.where(y_test_proba>0.1, 1, 0)
_,_ = show_results(sss["outcome"], y_train_pred, y_train_proba,sss["outcome"], y_test_pred, y_test_proba)'''
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-1025-a9d2569cce2e> in <module>
          5 cols = ['non_robot_merchandise', 'time', 'nb_device', 'nb_ip', 'nb_country']
          6 cols = ['non_robot_merchandise', 'time', 'nb_device', 'nb_ip', 'nb_country', 'my_agg_max', 'my_agg_mean',  'my_agg_std']
    ----> 7 only_logit_runner(sss[cols], sss["outcome"])
          8 
          9 '''y_test_proba = log_reg.predict(sss[cols])


    /usr/local/lib/python3.8/dist-packages/pandas/core/frame.py in __getitem__(self, key)
       3462             if is_iterator(key):
       3463                 key = list(key)
    -> 3464             indexer = self.loc._get_listlike_indexer(key, axis=1)[1]
       3465 
       3466         # take() does not accept boolean indexers


    /usr/local/lib/python3.8/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
       1312             keyarr, indexer, new_indexer = ax._reindex_non_unique(keyarr)
       1313 
    -> 1314         self._validate_read_indexer(keyarr, indexer, axis)
       1315 
       1316         if needs_i8_conversion(ax.dtype) or isinstance(


    /usr/local/lib/python3.8/dist-packages/pandas/core/indexing.py in _validate_read_indexer(self, key, indexer, axis)
       1375 
       1376             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
    -> 1377             raise KeyError(f"{not_found} not in index")
       1378 
       1379 


    KeyError: "['nb_device', 'nb_ip', 'nb_country'] not in index"



```python
only_logit_runner(sss[["non_robot_merchandise"]], sss["outcome"])
```


```python
only_logit_runner(df[["merchandise_1", "merchandise_2", "merchandise_3"]], df["outcome"])
```

- time (speed) seems to be a good predictor than the nb of ipv4 or the nb of countries
- oddly enough, non-bots connect from many countries

### merchandise


```python
print(pd.read_csv("Projet_ML.csv")["merchandise"].unique())
for col_name in new_cols:
  print(col_name,":",np.corrcoef(df[col_name], df.outcome)[0,1])
```



## visualizations

### plotting


```python
TARGET_COL_NAME = TARGET_COL
```


```python
#remove outliers
from collections import Counter
def detect_outliers(df, n, target_name, features_list):
    outlier_indices = []
    for feature in features_list:
        if feature==target_name: continue
        if df[feature].nunique()<10: continue
        Q1 = np.percentile(df[feature], 25)
        Q3 = np.percentile(df[feature], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[feature] < Q1 - outlier_step) | (df[feature] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(key for key, value in outlier_indices.items() if value > n)
    return multiple_outliers


```


```python
shuffled = sss_resampled.sample(frac=1)
outliers_ind = detect_outliers(shuffled, 2, "outcome", shuffled.columns )
print(f"outliers_rate = {100*len(outliers_ind)/len(shuffled):.2f}%")
shuffled = shuffled.drop(outliers_ind, axis=0)
#plot_all_fetaures(shuffled, shuffled.columns, target="outcome", plot_all=1, skip_great_and_loss=0)
```


```python
def optimize_tresh(df, target):
    dd_tresh = {}
    for col_name in df.columns:
        if col_name==target: continue
        print(col_name)
        dd_tresh[col_name] = {}
        best_tresh, best_p1, best_p2 = 0, 50, 50
        use_p2 = df[col_name].nunique()==2
        for tresh in np.arange(0, 1, 0.05):
            bool_to_reduce = False
            df_ = df.copy()
            df_[col_name] = (df[col_name]>tresh).astype(int)
            a,p1 = ttest_ind(df_[df_[target]==0][col_name], df_[df_[target]==1][col_name], equal_var=False)
            #print(f"p1 = {p1}")
            bool_to_reduce = (p1<0.05) and (p1<best_p1)
            if not bool_to_reduce: continue
            if use_p2:
                _ = df_.groupby(col_name)
                N1, N2 = _[target].count().values
                p1, p2 = _[target].sum().values
                _ = proportion_comparison_test(p1/N1, p2/N2, N1, N2)
                p2 = _['p_value']
                #print(f"p2 = {p2}")
                bool_to_reduce = (p2<0.05) and (p2<best_p2)
                if not bool_to_reduce: continue
            else: p2 = None
            print(f":{best_tresh} -> {tresh} p1={p1:.2f} p2={p2}")
            best_tresh, best_p1, best_p2 = tresh, p1, p2
        dd_tresh[col_name] = {"tr": best_tresh, "p1":p1, "p2":p2}
    return dd_tresh

%matplotlib inline
def plot_all_fetaures(df, columns, target:str, gamma=True, plot_all=False, skip_great_and_loss=False):
  for col_name in columns:
    try:
      if col_name ==TARGET_COL_NAME: continue
      if skip_great_and_loss and (col_name in Great_cols + Lost_cols): continue
      print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {col_name} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
      if df[col_name].nunique()==2:
        _ = df.groupby(col_name)
        N1, N2 = _[target].count().values
        p1, p2 = _[target].sum().values
        print(proportion_comparison_test(p1/N1, p2/N2, N1, N2))
      a,b = ttest_ind(df[df[target]==0][col_name], df[df[target]==1][col_name], equal_var=False)
      print(f"p-val={b:.2f} -->", "means seem different with a good p-value" if b<0.05 else "means diff test: no enough evidence")
      if df[col_name].nunique()>7:
        sns.scatterplot(x=range(len(df)), y=col_name, hue=target, data=df)
        plt.show()

      if 0 and df[col_name].nunique()<7:
        univariate_double_plot(df,col_name,col_name,hue=target, ylabel=target)

      else:
        feature_data = univariate_numerical_plot(df,col_name,col_name, gamma=gamma, target_as_hue=target)
      if not plot_all: break
    except Exception as e:
      print(f"{col_name} failed with {e}")

def transform_data_from_feature_eng(df1, model_name, dict_tresh:dict, cols=None, return_last=False):
    if cols is None: cols = df1.columns
    df2 = df1.copy()
    Great_cols = []
    Lost_cols = []
    t = ''
    _ = None

    for col_name in dict_tresh:
      if col_name not in df1.columns: continue
      _ = col_name;df2[_] = df1[_].apply(lambda x: 0 if x<=dict_tresh[col_name]["tr"] else 1); Great_cols.append(_)

    df2 = df2[cols]
    _ = t or _
    if return_last: return df2,_ , Great_cols, Lost_cols
    else: return df2




```


```python
optimize_tresh(sss_resampled, "outcome")
```


```python
sss_resampled.describe()
```


```python
sss_resampled = sss_resampled[sorted(sss_resampled.columns)]
sss_resampled_sm = sss_resampled_sm[sorted(sss_resampled_sm.columns)]
```


```python
run_visualiation_on_res_column_instead_of_all = 0
use_resampled_dataset = 1
skip_great_and_loss = False
plot_all = True
rmv_outliers = True
df__ = sss_resampled if use_resampled_dataset else sss
if rmv_outliers:
  outliers_ind = detect_outliers(df__, 2, "outcome", df__.columns )
  print(f"outliers_rate = {100*len(outliers_ind)/len(df__):.2f}%")
  df__ = df__.drop(outliers_ind, axis=0)

dict_tresh = optimize_tresh(df__, target=TARGET_COL)
df_,_,Great_cols, Lost_cols = transform_data_from_feature_eng(df1=df__ , model_name="logit",dict_tresh=dict_tresh, return_last=True)
if run_visualiation_on_res_column_instead_of_all: plot_all_fetaures(df_, [_], target="outcome", skip_great_and_loss=0) #all==plot_all_feature
else: plot_all_fetaures(df_, df_.columns, target="outcome", plot_all=plot_all, skip_great_and_loss=skip_great_and_loss)

#plot_all_fetaures(sss, set(important_cols)-set(choosen_columns), target="outcome", all=all)
#plot_all_fetaures(DF, Great_cols, target="outcome", all=True)
#plot_all_fetaures(DF, choosen_columns, target="outcome", all=all)
#plot_all_fetaures(df1, df1.columns, target="outcome", all=all)
#
```

    outliers_rate = 18.52%
    my_agg_max
    :0 -> 0.0 p1=55.00 p2=7.651361294702674e-05
    my_agg_mean
    :0 -> 0.0 p1=55.00 p2=7.651361294702674e-05
    my_agg_std
    :0 -> 0.0 p1=55.00 p2=7.651361294702674e-05
    my_agg_sum
    :0 -> 0.0 p1=55.00 p2=7.651361294702674e-05
    nb_auction_x
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    nb_auction_y
    :0 -> 0.05 p1=0.00 p2=None
    :0.05 -> 0.1 p1=0.00 p2=None
    :0.1 -> 0.15000000000000002 p1=0.00 p2=None
    nb_bid_x
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    nb_bid_y
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    nb_country_x
    :0 -> 0.0 p1=0.01 p2=None
    nb_country_y
    :0 -> 0.05 p1=0.00 p2=None
    nb_device_x
    :0 -> 0.05 p1=0.02 p2=None
    nb_device_y
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    nb_ip_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_ip_y
    :0 -> 0.0 p1=0.00 p2=None
    nb_merchandise_1_x
    :0 -> 0.0 p1=0.01 p2=None
    nb_merchandise_1_y
    :0 -> 0.0 p1=71.00 p2=0.006833730265801696
    nb_merchandise_2_x
    :0 -> 0.05 p1=0.00 p2=None
    nb_merchandise_2_y
    nb_merchandise_3_x
    nb_merchandise_3_y
    nb_merchandise_4_x
    :0 -> 0.0 p1=0.02 p2=None
    nb_merchandise_4_y
    :0 -> 0.0 p1=71.00 p2=0.013915805621387722
    nb_merchandise_5_x
    nb_merchandise_5_y
    nb_merchandise_6_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_merchandise_6_y
    :0 -> 0.0 p1=45.00 p2=3.9726208756807324e-05
    nb_url_x
    nb_url_y
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    non_robot_merchandise
    :0 -> 0.0 p1=71.00 p2=1.9593566541020735e-05
    time
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> my_agg_max >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5258620689655172, 'std_stat_eval': 0.13295912253550995, 'Z': 3.9550657295069995, 'p_value': 7.651361294702674e-05, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_1.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_2.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> my_agg_mean >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5258620689655172, 'std_stat_eval': 0.13295912253550995, 'Z': 3.9550657295069995, 'p_value': 7.651361294702674e-05, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_4.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_5.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> my_agg_std >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5258620689655172, 'std_stat_eval': 0.13295912253550995, 'Z': 3.9550657295069995, 'p_value': 7.651361294702674e-05, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_7.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_8.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> my_agg_sum >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5258620689655172, 'std_stat_eval': 0.13295912253550995, 'Z': 3.9550657295069995, 'p_value': 7.651361294702674e-05, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_10.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_11.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_auction_x >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5597863942419317, 'std_stat_eval': 0.08728099536812643, 'Z': 6.413611484160002, 'p_value': 1.4211209986569884e-10, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_13.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_14.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_auction_y >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.9466666666666667, 'std_stat_eval': 0.08760705205479986, 'Z': 10.805827207546132, 'p_value': 0.0, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_16.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_17.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_bid_x >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5258620689655172, 'std_stat_eval': 0.13295912253550995, 'Z': 3.9550657295069995, 'p_value': 7.651361294702674e-05, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_19.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_20.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_bid_y >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5258620689655172, 'std_stat_eval': 0.13295912253550995, 'Z': 3.9550657295069995, 'p_value': 7.651361294702674e-05, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_22.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_23.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_country_x >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.27272727272727276, 'std_stat_eval': 0.10021496035998259, 'Z': 2.72142274713883, 'p_value': 0.006500157608642532, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.01 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_25.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_26.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_country_y >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.9726027397260274, 'std_stat_eval': 0.08728099536812643, 'Z': 11.143350687327356, 'p_value': 0.0, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_28.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_29.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_device_x >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5590551181102362, 'std_stat_eval': 0.22731090111284577, 'Z': 2.4594294218766923, 'p_value': 0.013915805621387722, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.02 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_31.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_32.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_device_y >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.9861111111111112, 'std_stat_eval': 0.08714957084928397, 'Z': 11.315157395513591, 'p_value': 0.0, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_34.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_35.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_ip_x >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5867768595041323, 'std_stat_eval': 0.15700666852043052, 'Z': 3.737273486748608, 'p_value': 0.0001860265488540236, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_37.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_38.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_ip_y >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5867768595041323, 'std_stat_eval': 0.15700666852043052, 'Z': 3.737273486748608, 'p_value': 0.0001860265488540236, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_40.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_41.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_merchandise_1_x >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5634920634920635, 'std_stat_eval': 0.20832732074055776, 'Z': 2.7048399676478976, 'p_value': 0.006833730265801696, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.01 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_43.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_44.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_merchandise_1_y >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5634920634920635, 'std_stat_eval': 0.20832732074055776, 'Z': 2.7048399676478976, 'p_value': 0.006833730265801696, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.01 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_46.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_47.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_merchandise_2_x >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5304347826086957, 'std_stat_eval': 0.12954890466953886, 'Z': 4.094475240541483, 'p_value': 4.2312562489987826e-05, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_49.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_50.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_merchandise_2_y >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.045454545454545414, 'std_stat_eval': 0.08678870151099544, 'Z': 0.523738051879791, 'p_value': 0.6004607454663171, 'reject_null': False, 'message': 'no enough evidence to support that p0!=p1'}
    p-val=0.60 --> means diff test: no enough evidence



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_52.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_53.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_merchandise_3_x >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5503875968992248, 'std_stat_eval': 0.29117336383189935, 'Z': 1.8902401979906904, 'p_value': 0.0587258427686812, 'reject_null': False, 'message': 'no enough evidence to support that p0!=p1'}
    p-val=0.08 --> means diff test: no enough evidence



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_55.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_56.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_merchandise_3_y >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5503875968992248, 'std_stat_eval': 0.29117336383189935, 'Z': 1.8902401979906904, 'p_value': 0.0587258427686812, 'reject_null': False, 'message': 'no enough evidence to support that p0!=p1'}
    p-val=0.08 --> means diff test: no enough evidence



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_58.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_59.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_merchandise_4_x >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5590551181102362, 'std_stat_eval': 0.22731090111284577, 'Z': 2.4594294218766923, 'p_value': 0.013915805621387722, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.02 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_61.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_62.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_merchandise_4_y >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.5590551181102362, 'std_stat_eval': 0.22731090111284577, 'Z': 2.4594294218766923, 'p_value': 0.013915805621387722, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.02 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_64.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_65.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_merchandise_5_x >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.045454545454545414, 'std_stat_eval': 0.1164392617545147, 'Z': 0.3903712954688413, 'p_value': 0.6962620103843555, 'reject_null': False, 'message': 'no enough evidence to support that p0!=p1'}
    p-val=0.70 --> means diff test: no enough evidence



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_67.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_68.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_merchandise_5_y >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.045454545454545414, 'std_stat_eval': 0.1164392617545147, 'Z': 0.3903712954688413, 'p_value': 0.6962620103843555, 'reject_null': False, 'message': 'no enough evidence to support that p0!=p1'}
    p-val=0.70 --> means diff test: no enough evidence



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_70.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_71.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_merchandise_6_x >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.4254901960784314, 'std_stat_eval': 0.10354912013746814, 'Z': 4.109066262596589, 'p_value': 3.9726208756807324e-05, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_73.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_74.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_merchandise_6_y >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.4254901960784314, 'std_stat_eval': 0.10354912013746814, 'Z': 4.109066262596589, 'p_value': 3.9726208756807324e-05, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_76.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_77.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_url_x >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.09027777777777779, 'std_stat_eval': 0.0974361823144665, 'Z': 0.9265323787667952, 'p_value': 0.3541693783403337, 'reject_null': False, 'message': 'no enough evidence to support that p0!=p1'}
    p-val=0.36 --> means diff test: no enough evidence



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_79.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_80.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> nb_url_y >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.9466666666666667, 'std_stat_eval': 0.08760705205479986, 'Z': 10.805827207546132, 'p_value': 0.0, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_82.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_83.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> non_robot_merchandise >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.6016949152542372, 'std_stat_eval': 0.14092956284320043, 'Z': 4.269472657938269, 'p_value': 1.9593566541020735e-05, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_85.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_86.png)
    


    
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> time >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {'p_hat': 0.6826923076923077, 'std_stat_eval': 0.10614790838673391, 'Z': 6.431519170448665, 'p_value': 1.2633494250735566e-10, 'reject_null': True, 'message': 'significant difference'}
    p-val=0.00 --> means seem different with a good p-value



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_88.png)
    



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_164_89.png)
    



```python
sns.histplot(x="outcome", data=sss)
```


```python
r_sss= (sss.outcome==1).sum()/len(sss) #poids à donner à 0
weights_sss = r_sss + (1-2*r_sss)*sss.outcome #associer r à 0 et (1-r) à 1
```


```python
sns.displot(x="nb_country", data=sss, hue="outcome", kde=True, weights=weights_sss)
```


```python
sns.scatterplot(x=range(len(sss)), y="nb_country", data=sss, hue="outcome")
```


```python
sns.displot(x="nb_device", data=sss, hue="outcome", kde=True, weights=weights_sss)
```


```python
sns.scatterplot(x=range(len(sss)), y="nb_device", data=sss, hue="outcome")
```


```python
sns.scatterplot(x="nb_device", y="nb_country", hue="outcome", data=sss)
```


```python
sns.scatterplot(x=range(len(sss)), y="time", hue="outcome", data=sss)
```


```python
sns.displot(x="time", hue="outcome", data=sss, kde=True)
```


```python
sns.displot(x="time", hue="outcome", data=sss, kde=True, weights=get_weight(sss.outcome))
```


```python

sns.scatterplot(x="nb_country", y="nb_auction", hue="outcome", data=sss)
```


```python
sns.displot(x="nb_country", hue="outcome", kde=True, data=sss)
```


```python
sns.heatmap(sss.corr().abs(), vmin=0, vmax=1)
```

ip, i_p_ et country sont evidemment correlées entres elles


```python
sns.histplot(x="outcome", data=sss)
```


```python
sns.histplot(x="nb_country", data=sss, hue="outcome")
```


```python
sns.scatterplot(x="nb_device", y="nb_country", hue="outcome", data=sss)
```

## test_env

Instead of accuracy, owe will use precision, recall and F1-score [2]

for later: i can test on [nb_device	nb_country	nb_auction	outcome	merchandise0	merchandise1	merchandise2] using the same outliers caps right ??


```
  list_fct = {
        "logit": logit_regression,
        "logistic": logistic_regression_sklearn,
        "svc": svc_classifier,
        "knn":knn_classifier,
        "sdg":sdg_classifier,
        "tree":decision_tree_classifier,
        "lda": lda_classifier,
        "forest":random_forest_classifier,
        "ada": ada_boost_classifier,
        "xgboost": gradient_boosting_classifier
        }

```




```python
added_cols = ['my_agg_max','my_agg_mean','my_agg_sum','my_agg_std']
```


```python
#important_cols_ = ["nb_merchandise_1","nb_merchandise_3", "merchandise_1", "merchandise_2", "merchandise_3", "nb_country", "nb_auction", "time", "nb_url", "nb_device"]
#classifier = M_Classifier_res(sss, "outcome", test_size=0.4, important_cols=important_cols_)
sss = sss[added_cols+list(set(sss.columns)-set(added_cols))]
classifier = M_Classifier_res(sss, "outcome", test_size=0.4, oversample=True, description="raw data modeling")
```

    MAIN_DF.shape: (87, 31)
    TARGET_NAME: outcome
    important_cols: ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    TEST_SIZE: 0.4
    oversample: True
    feature_eng: True
    my_agg_max
    :0 -> 0.05 p1=0.04 p2=None
    my_agg_mean
    my_agg_sum
    :0 -> 0.05 p1=0.04 p2=None
    my_agg_std
    nb_device_x
    :0 -> 0.55 p1=0.02 p2=None
    time
    :0 -> 0.05 p1=0.00 p2=None
    nb_merchandise_4_x
    :0 -> 0.0 p1=0.02 p2=None
    non_robot_merchandise
    nb_device_y
    :0 -> 0.05 p1=0.00 p2=None
    :0.05 -> 0.1 p1=0.00 p2=None
    nb_url_y
    :0 -> 0.05 p1=0.01 p2=None
    :0.05 -> 0.25 p1=0.00 p2=None
    nb_country_x
    nb_merchandise_5_x
    nb_merchandise_3_x
    nb_ip_y
    :0 -> 0.0 p1=0.00 p2=None
    nb_country_y
    :0 -> 0.05 p1=0.00 p2=None
    nb_bid_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_merchandise_1_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_merchandise_3_y
    nb_merchandise_5_y
    nb_auction_y
    :0 -> 0.1 p1=0.01 p2=None
    :0.1 -> 0.15000000000000002 p1=0.00 p2=None
    nb_merchandise_6_y
    nb_bid_y
    :0 -> 0.0 p1=0.00 p2=None
    nb_auction_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_merchandise_1_y
    nb_merchandise_2_x
    nb_merchandise_2_y
    nb_merchandise_6_x
    nb_merchandise_4_y
    nb_ip_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_url_x
    :0 -> 0.2 p1=0.04 p2=None



```python
sss_resampled = sss_resampled[added_cols+list(set(sss_resampled.columns)-set(added_cols))]
sss_resampled_sm = sss_resampled_sm[added_cols+list(set(sss_resampled_sm.columns)-set(added_cols))]
classifier1 = M_Classifier_res(sss_resampled, "outcome", test_size=0.5, oversample=False, description="duplic data modeling")
classifier2 = M_Classifier_res(sss_resampled_sm, "outcome", test_size=0.5, oversample=False, description="smote data modeling")
```

    MAIN_DF.shape: (162, 31)
    TARGET_NAME: outcome
    important_cols: ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    TEST_SIZE: 0.5
    oversample: False
    feature_eng: True
    my_agg_max
    :0 -> 0.0 p1=0.00 p2=None
    my_agg_mean
    :0 -> 0.0 p1=0.00 p2=None
    my_agg_sum
    :0 -> 0.0 p1=0.00 p2=None
    my_agg_std
    :0 -> 0.0 p1=0.00 p2=None
    nb_device_x
    :0 -> 0.05 p1=0.03 p2=None
    :0.05 -> 0.55 p1=0.02 p2=None
    time
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    nb_device_y
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    nb_merchandise_4_x
    :0 -> 0.0 p1=0.02 p2=None
    non_robot_merchandise
    :0 -> 0.0 p1=81.00 p2=9.015600541673052e-07
    nb_url_y
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    nb_country_x
    :0 -> 0.0 p1=0.01 p2=None
    nb_merchandise_5_x
    nb_merchandise_3_x
    nb_ip_y
    :0 -> 0.0 p1=0.00 p2=None
    nb_country_y
    :0 -> 0.05 p1=0.00 p2=None
    nb_bid_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_merchandise_1_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_merchandise_3_y
    nb_merchandise_5_y
    nb_auction_y
    :0 -> 0.05 p1=0.00 p2=None
    :0.05 -> 0.1 p1=0.00 p2=None
    :0.1 -> 0.15000000000000002 p1=0.00 p2=None
    nb_merchandise_6_y
    :0 -> 0.0 p1=55.00 p2=2.737060065283181e-05
    nb_bid_y
    :0 -> 0.0 p1=0.00 p2=None
    nb_auction_x
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    nb_merchandise_1_y
    :0 -> 0.0 p1=81.00 p2=0.0001702181723191032
    nb_merchandise_2_x
    :0 -> 0.05 p1=0.00 p2=None
    nb_merchandise_2_y
    nb_merchandise_6_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_merchandise_4_y
    :0 -> 0.0 p1=81.00 p2=0.023123071197780343
    nb_ip_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_url_x
    :0 -> 0.2 p1=0.04 p2=None
    MAIN_DF.shape: (162, 31)
    TARGET_NAME: outcome
    important_cols: ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    TEST_SIZE: 0.5
    oversample: False
    feature_eng: True
    my_agg_max
    :0 -> 0.0 p1=0.00 p2=None
    my_agg_mean
    :0 -> 0.0 p1=0.00 p2=None
    my_agg_sum
    :0 -> 0.0 p1=0.00 p2=None
    my_agg_std
    :0 -> 0.0 p1=0.00 p2=None
    nb_device_x
    :0 -> 0.0 p1=0.00 p2=None
    time
    :0 -> 0.05 p1=0.00 p2=None
    nb_device_y
    :0 -> 0.05 p1=0.00 p2=None
    :0.05 -> 0.1 p1=0.00 p2=None
    nb_merchandise_4_x
    :0 -> 0.0 p1=0.02 p2=None
    non_robot_merchandise
    :0 -> 0.0 p1=81.00 p2=9.015600541673052e-07
    nb_url_y
    :0 -> 0.05 p1=0.00 p2=None
    :0.05 -> 0.1 p1=0.00 p2=None
    nb_country_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_merchandise_5_x
    nb_merchandise_3_x
    nb_ip_y
    :0 -> 0.0 p1=0.00 p2=None
    nb_country_y
    :0 -> 0.05 p1=0.00 p2=None
    nb_bid_x
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    nb_merchandise_1_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_merchandise_3_y
    nb_merchandise_5_y
    :0 -> 0.45 p1=0.02 p2=None
    :0.45 -> 0.55 p1=0.01 p2=None
    :0.55 -> 0.6000000000000001 p1=0.00 p2=None
    :0.6000000000000001 -> 0.65 p1=0.00 p2=None
    :0.65 -> 0.7000000000000001 p1=0.00 p2=None
    :0.7000000000000001 -> 0.75 p1=0.00 p2=None
    nb_auction_y
    :0 -> 0.1 p1=0.00 p2=None
    :0.1 -> 0.15000000000000002 p1=0.00 p2=None
    nb_merchandise_6_y
    :0 -> 0.0 p1=0.00 p2=None
    nb_bid_y
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    nb_auction_x
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    nb_merchandise_1_y
    :0 -> 0.0 p1=81.00 p2=0.0001702181723191032
    nb_merchandise_2_x
    :0 -> 0.0 p1=0.00 p2=None
    :0.0 -> 0.05 p1=0.00 p2=None
    nb_merchandise_2_y
    :0 -> 0.0 p1=0.00 p2=None
    nb_merchandise_6_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_merchandise_4_y
    :0 -> 0.0 p1=81.00 p2=0.023123071197780343
    nb_ip_x
    :0 -> 0.0 p1=0.00 p2=None
    nb_url_x
    :0 -> 0.0 p1=0.00 p2=None



```python

```


```python
[print(elt) for elt in added_cols]
```


```python
classifier.compute_all("sdg", feature_eng=True, test_size=0.4)
```

    model loader: raw data modeling
    feature_eng =  True
    oversample =  True method= smote
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.4
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~sdg~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'average': 20, 'loss': 'hinge', 'max_iter': 15}
    y_train_proba: 2
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[48  0]
     [ 0 48]]
    - accuracy = 100.00%
    - f1 = 100.00%
    - roc(area under the curve) = 100.00%
    - precision = 100.00%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[31  2]
     [ 0  2]]
    - accuracy = 94.29%
    - f1 = 66.67%
    - roc(area under the curve) = 100.00%
    - precision = 50.00%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~sdg: nb_params = 30~~~~~~~~~~~~~~~~



```python
classifier2.compute_all('sdg')
```

    model loader: smote data modeling
    feature_eng =  True
    oversample =  False method= None
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.5
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~sdg~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'average': 1, 'loss': 'modified_huber', 'max_iter': 15}
    y_train_proba: 2
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[40  1]
     [ 0 40]]
    - accuracy = 98.77%
    - f1 = 98.77%
    - roc(area under the curve) = 100.00%
    - precision = 97.56%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[37  3]
     [ 0 41]]
    - accuracy = 96.30%
    - f1 = 96.47%
    - roc(area under the curve) = 100.00%
    - precision = 93.18%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~sdg: nb_params = 30~~~~~~~~~~~~~~~~



```python
classifier.important_cols
```


```python
sss.shape
```


```python
from pprint import pprint
pprint(classifier.DICT_RES)
```


```python
classifier.show_res()
```


```python
classifier.MAIN_DF.columns
```

linear feature importance


```python
# always has 100% recall and never 100% accuray --> good
cols = ['nb_merchandise_1', 'nb_bid', 'nb_country', 'time', 'nb_auction', 'nb_merchandise_2']
cols = cols + list(set(classifier.MAIN_DF.columns)-set(cols))
classifier.compute_all("logit", test_size=0.4, important_cols=cols)
```


```python
classifier1.compute_all("logit", oversample="duplic", test_size=0.4) #prone to overfitting?
```

N'oublie pas le jeu de tresh, ça donne du 100% partout si on s'y prends bien


```python
classifier2.compute_all("logit")
```


```python
#unstable; low on high test_size
#with the new class_weigh update in model + oversample, stable and good + all merchandises
classifier.compute_all("tree", oversample=True, test_size=0.4)
```


```python
t = [("ee",1),("rr",3),("tt",1)]

```


```python
_ = ['nb_country', 'my_agg_sum', 'nb_device', 'nb_merchandise_2'] if ADD_LEN_TO_GROUPBY else ['nb_merchandise_6', 'nb_merchandise_5','nb_auction', 'nb_device', 'nb_merchandise_2']
classifier2.compute_all("tree")
```


```python

```

#### prof


```python
# Ovr do not impact when it is just 3 merchandises
# As he will always proritize the deep forest, send less data (test_size=0.4) and add a oversampling. It worked on the 6-merchandises dataset
# i've added class_weigh but still need ovr on the 6-merchandises i guess
classifier.compute_all("forest", oversample=True, test_size=0.4)
```

    model loader: raw data modeling
    feature_eng =  True
    oversample =  True method= smote
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.4
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~forest~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'criterion': 'gini', 'max_depth': 3, 'max_features': 'sqrt', 'max_leaf_nodes': 3, 'n_estimators': 10}
    features:['nb_device_y', 'nb_url_y', 'nb_country_y', 'nb_auction_y'] ranking = [('nb_device_y', 1), ('nb_url_y', 1), ('nb_country_y', 1), ('nb_auction_y', 1), ('my_agg_sum', 2), ('nb_auction_x', 3), ('my_agg_mean', 4), ('nb_merchandise_2_x', 5), ('nb_merchandise_6_y', 6), ('nb_ip_y', 7), ('my_agg_std', 8), ('nb_bid_y', 9), ('nb_merchandise_5_x', 10), ('nb_merchandise_3_x', 11), ('nb_country_x', 12), ('my_agg_max', 13), ('non_robot_merchandise', 14), ('nb_merchandise_2_y', 15), ('nb_merchandise_1_y', 16), ('nb_bid_x', 17), ('nb_merchandise_6_x', 18), ('nb_merchandise_4_y', 19), ('nb_merchandise_4_x', 20), ('nb_ip_x', 21), ('time', 22), ('nb_merchandise_1_x', 23), ('nb_url_x', 24), ('nb_merchandise_3_y', 25), ('nb_merchandise_5_y', 26), ('nb_device_x', 27)]
    y_train_proba: 6
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[44  4]
     [ 0 48]]
    - accuracy = 95.83%
    - f1 = 96.00%
    - roc(area under the curve) = 100.00%
    - precision = 92.31%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[32  1]
     [ 0  2]]
    - accuracy = 97.14%
    - f1 = 80.00%
    - roc(area under the curve) = 100.00%
    - precision = 66.67%
    - recall = 100.00%
    ~~~~~~~~~~~~~~forest: nb_params = 4~~~~~~~~~~~~~~~



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_208_1.png)
    



```python
classifier1.compute_all("forest",
                        #remove_cols=['nb_merchandise_1', 'nb_merchandise_3','nb_merchandise_5'],
                        #important_cols=['nb_country'],
                        test_size=0.5)
```


```python
_ = ['nb_url','nb_country','nb_merchandise_2','nb_device'] if ADD_LEN_TO_GROUPBY else ['time', 'nb_merchandise_6', 'nb_merchandise_5', 'nb_merchandise_2']
classifier2.compute_all("forest",
                        #important_cols=_
                        ) #{'criterion': 'gini', 'max_depth': 5, 'max_features': 'sqrt', 'max_leaf_nodes': 5, 'n_estimators': 10}
```

    model loader: smote data modeling
    feature_eng =  True
    oversample =  False method= None
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.5
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~forest~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'criterion': 'entropy', 'max_depth': 5, 'max_features': 'sqrt', 'max_leaf_nodes': 5, 'n_estimators': 10}
    features:['nb_url_y', 'nb_country_y', 'nb_auction_y', 'nb_merchandise_2_y'] ranking = [('nb_url_y', 1), ('nb_country_y', 1), ('nb_auction_y', 1), ('nb_merchandise_2_y', 1), ('my_agg_sum', 2), ('nb_merchandise_6_y', 3), ('nb_merchandise_5_y', 4), ('nb_merchandise_1_x', 5), ('nb_bid_y', 6), ('nb_merchandise_1_y', 7), ('my_agg_std', 8), ('my_agg_max', 9), ('nb_bid_x', 10), ('nb_merchandise_5_x', 11), ('nb_ip_y', 12), ('nb_country_x', 13), ('nb_merchandise_3_x', 14), ('nb_auction_x', 15), ('nb_merchandise_6_x', 16), ('nb_merchandise_2_x', 17), ('non_robot_merchandise', 18), ('my_agg_mean', 19), ('nb_merchandise_4_x', 20), ('nb_device_y', 21), ('nb_merchandise_4_y', 22), ('nb_ip_x', 23), ('time', 24), ('nb_merchandise_3_y', 25), ('nb_url_x', 26), ('nb_device_x', 27)]
    y_train_proba: 7
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[40  1]
     [ 0 40]]
    - accuracy = 98.77%
    - f1 = 98.77%
    - roc(area under the curve) = 100.00%
    - precision = 97.56%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[37  3]
     [ 0 41]]
    - accuracy = 96.30%
    - f1 = 96.47%
    - roc(area under the curve) = 100.00%
    - precision = 93.18%
    - recall = 100.00%
    ~~~~~~~~~~~~~~forest: nb_params = 4~~~~~~~~~~~~~~~



    
![png](Projet_final_ML_v8%20%281%29_files/Projet_final_ML_v8%20%281%29_210_1.png)
    



```python
classifier.compute_all("logistic", oversample="duplic", test_size=0.4,
                       remove_cols=[],
                       )
```

    model loader: raw data modeling
    feature_eng =  True
    oversample used =  duplic
    oversample =  duplic method= duplic
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'non_robot_merchandise', 'nb_merchandise_4_x', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.4
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~logistic~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'C': 0.01, 'max_iter': 100, 'penalty': 'none', 'solver': 'lbfgs'}
    ranking = [('nb_device_x', 1), ('nb_device_y', 1), ('nb_country_x', 1), ('nb_merchandise_5_y', 1), ('nb_country_y', 2), ('nb_auction_x', 3), ('nb_auction_y', 4), ('nb_bid_x', 5), ('nb_merchandise_5_x', 6), ('my_agg_max', 7), ('nb_bid_y', 8), ('nb_url_y', 9), ('nb_merchandise_6_x', 10), ('nb_merchandise_2_y', 11), ('time', 12), ('nb_merchandise_6_y', 13), ('nb_merchandise_2_x', 14), ('nb_ip_y', 15), ('nb_ip_x', 16), ('non_robot_merchandise', 17), ('my_agg_sum', 18), ('nb_merchandise_1_y', 19), ('nb_url_x', 20), ('nb_merchandise_1_x', 21), ('my_agg_std', 22), ('my_agg_mean', 23), ('nb_merchandise_4_x', 24), ('nb_merchandise_4_y', 25), ('nb_merchandise_3_x', 26), ('nb_merchandise_3_y', 27)]
    y_train_proba: 9
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[48  0]
     [ 0 48]]
    - accuracy = 100.00%
    - f1 = 100.00%
    - roc(area under the curve) = 100.00%
    - precision = 100.00%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[30  3]
     [ 0  2]]
    - accuracy = 91.43%
    - f1 = 57.14%
    - roc(area under the curve) = 100.00%
    - precision = 40.00%
    - recall = 100.00%
    ~~~~~~~~~~~~~logistic: nb_params = 4~~~~~~~~~~~~~~



```python
classifier1.compute_all("logistic")
```

    model loader: duplic data modeling
    feature_eng =  True
    oversample =  False method= None
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.5
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~logistic~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'C': 0.01, 'max_iter': 100, 'penalty': 'none', 'solver': 'lbfgs'}
    ranking = [('nb_device_x', 1), ('nb_device_y', 1), ('nb_country_y', 1), ('nb_url_x', 1), ('my_agg_mean', 2), ('my_agg_sum', 3), ('nb_url_y', 4), ('nb_merchandise_2_y', 5), ('nb_auction_x', 6), ('nb_auction_y', 7), ('my_agg_std', 8), ('non_robot_merchandise', 9), ('nb_bid_y', 10), ('my_agg_max', 11), ('nb_merchandise_1_x', 12), ('nb_bid_x', 13), ('nb_merchandise_1_y', 14), ('nb_merchandise_2_x', 15), ('nb_ip_y', 16), ('time', 17), ('nb_merchandise_6_y', 18), ('nb_country_x', 19), ('nb_ip_x', 20), ('nb_merchandise_5_x', 21), ('nb_merchandise_5_y', 22), ('nb_merchandise_6_x', 23), ('nb_merchandise_3_x', 24), ('nb_merchandise_3_y', 25), ('nb_merchandise_4_x', 26), ('nb_merchandise_4_y', 27)]
    y_train_proba: 6
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[37  4]
     [ 0 40]]
    - accuracy = 95.06%
    - f1 = 95.24%
    - roc(area under the curve) = 100.00%
    - precision = 90.91%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[37  3]
     [ 0 41]]
    - accuracy = 96.30%
    - f1 = 96.47%
    - roc(area under the curve) = 100.00%
    - precision = 93.18%
    - recall = 100.00%
    ~~~~~~~~~~~~~logistic: nb_params = 4~~~~~~~~~~~~~~



```python
_ = ['nb_merchandise_2','my_agg_sum','nb_device', 'nb_url'] if ADD_LEN_TO_GROUPBY else ['time', 'nb_merchandise_6','nb_merchandise_5','nb_merchandise_2']
classifier2.compute_all("logistic",
                        #important_cols=_,
                        #test_size=0.2
                        )
```

    model loader: smote data modeling
    feature_eng =  True
    oversample =  False method= None
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.5
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~logistic~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'C': 0.01, 'max_iter': 100, 'penalty': 'none', 'solver': 'lbfgs'}
    ranking = [('nb_device_y', 1), ('nb_url_y', 1), ('my_agg_sum', 1), ('nb_merchandise_2_y', 1), ('nb_auction_x', 2), ('nb_auction_y', 3), ('nb_country_y', 4), ('my_agg_mean', 5), ('nb_bid_y', 6), ('nb_merchandise_2_x', 7), ('my_agg_std', 8), ('nb_bid_x', 9), ('nb_url_x', 10), ('my_agg_max', 11), ('non_robot_merchandise', 12), ('nb_ip_x', 13), ('time', 14), ('nb_ip_y', 15), ('nb_device_x', 16), ('nb_merchandise_5_y', 17), ('nb_merchandise_1_x', 18), ('nb_merchandise_5_x', 19), ('nb_merchandise_6_y', 20), ('nb_merchandise_1_y', 21), ('nb_country_x', 22), ('nb_merchandise_6_x', 23), ('nb_merchandise_3_x', 24), ('nb_merchandise_3_y', 25), ('nb_merchandise_4_x', 26), ('nb_merchandise_4_y', 27)]
    coeffs = [[-41.97799871 -41.19001266  25.11573221  27.48179125]]
    y_train_proba: 12
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[40  1]
     [ 0 40]]
    - accuracy = 98.77%
    - f1 = 98.77%
    - roc(area under the curve) = 100.00%
    - precision = 97.56%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[37  3]
     [ 0 41]]
    - accuracy = 96.30%
    - f1 = 96.47%
    - roc(area under the curve) = 100.00%
    - precision = 93.18%
    - recall = 100.00%
    ~~~~~~~~~~~~~logistic: nb_params = 4~~~~~~~~~~~~~~
    NOPE !! ~~~~ on test and ~~~~ on train



```python
classifier.compute_all("knn", oversample="smote", test_size=0.2)
```

    model loader: raw data modeling
    feature_eng =  True
    oversample used =  smote
    oversample =  smote method= smote
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.2
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~knn~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'n_neighbors': 2}
    y_train_proba: 3
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[62  2]
     [ 0 64]]
    - accuracy = 98.44%
    - f1 = 98.46%
    - roc(area under the curve) = 100.00%
    - precision = 96.97%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[16  1]
     [ 0  1]]
    - accuracy = 94.44%
    - f1 = 66.67%
    - roc(area under the curve) = 100.00%
    - precision = 50.00%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~knn: nb_params = 30~~~~~~~~~~~~~~~~
    NOPE !! ~~~~ on test and ~~~~ on train



```python
classifier2.compute_all("knn", test_size=0.4)
```

    model loader: smote data modeling
    feature_eng =  True
    oversample =  False method= None
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.4
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~knn~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'n_neighbors': 2}
    y_train_proba: 3
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[45  4]
     [ 0 48]]
    - accuracy = 95.88%
    - f1 = 96.00%
    - roc(area under the curve) = 100.00%
    - precision = 92.31%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[28  4]
     [ 0 33]]
    - accuracy = 93.85%
    - f1 = 94.29%
    - roc(area under the curve) = 100.00%
    - precision = 89.19%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~knn: nb_params = 30~~~~~~~~~~~~~~~~



```python
#because of the cost function
# very good on the 6-merchandise dataset (oversample=True, test_size from max=0.4(good rec) to best=0.2(good pres too))
# still need ovr after class_weigh !!
classifier.compute_all("svc", oversample=True, test_size=0.4,
                       #important_cols=['my_agg_max', 'my_agg_std', 'my_agg_mean', 'my_agg_sum', 'non_robot_merchandise', 'time']
                       )
```

    model loader: raw data modeling
    feature_eng =  True
    oversample =  True method= smote
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.4
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~svc~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ranking = [('nb_device_y', 1), ('nb_url_y', 1), ('nb_country_x', 1), ('nb_merchandise_5_y', 1), ('nb_country_y', 2), ('nb_auction_y', 3), ('my_agg_max', 4), ('nb_merchandise_5_x', 5), ('nb_device_x', 6), ('nb_merchandise_1_y', 7), ('nb_merchandise_1_x', 8), ('non_robot_merchandise', 9), ('time', 10), ('nb_merchandise_2_y', 11), ('nb_merchandise_6_x', 12), ('my_agg_std', 13), ('my_agg_mean', 14), ('nb_merchandise_2_x', 15), ('nb_merchandise_6_y', 16), ('nb_ip_y', 17), ('nb_bid_x', 18), ('my_agg_sum', 19), ('nb_auction_x', 20), ('nb_merchandise_4_y', 21), ('nb_merchandise_3_x', 22), ('nb_merchandise_4_x', 23), ('nb_ip_x', 24), ('nb_url_x', 25), ('nb_merchandise_3_y', 26), ('nb_bid_y', 27)]
    y_train_proba: 12
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[46  2]
     [ 0 48]]
    - accuracy = 97.92%
    - f1 = 97.96%
    - roc(area under the curve) = 100.00%
    - precision = 96.00%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[31  2]
     [ 0  2]]
    - accuracy = 94.29%
    - f1 = 66.67%
    - roc(area under the curve) = 100.00%
    - precision = 50.00%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~~svc: nb_params = 4~~~~~~~~~~~~~~~~
    NOPE !! ~~~~ on test and ~~~~ on train



```python
classifier1.compute_all("svc")
```

    model loader: duplic data modeling
    feature_eng =  True
    oversample =  False method= None
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.5
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~svc~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ranking = [('nb_device_y', 1), ('nb_country_y', 1), ('my_agg_sum', 1), ('nb_url_x', 1), ('nb_url_y', 2), ('nb_auction_y', 3), ('my_agg_mean', 4), ('nb_merchandise_2_y', 5), ('nb_auction_x', 6), ('my_agg_std', 7), ('nb_device_x', 8), ('nb_country_x', 9), ('nb_merchandise_5_y', 10), ('my_agg_max', 11), ('nb_merchandise_1_x', 12), ('nb_merchandise_5_x', 13), ('nb_merchandise_1_y', 14), ('non_robot_merchandise', 15), ('nb_merchandise_6_x', 16), ('nb_merchandise_6_y', 17), ('nb_merchandise_2_x', 18), ('nb_bid_x', 19), ('nb_ip_x', 20), ('nb_ip_y', 21), ('time', 22), ('nb_merchandise_3_x', 23), ('nb_bid_y', 24), ('nb_merchandise_4_x', 25), ('nb_merchandise_4_y', 26), ('nb_merchandise_3_y', 27)]
    y_train_proba: 9
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[40  1]
     [ 0 40]]
    - accuracy = 98.77%
    - f1 = 98.77%
    - roc(area under the curve) = 100.00%
    - precision = 97.56%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[37  3]
     [ 0 41]]
    - accuracy = 96.30%
    - f1 = 96.47%
    - roc(area under the curve) = 100.00%
    - precision = 93.18%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~~svc: nb_params = 4~~~~~~~~~~~~~~~~



```python
_ = ['nb_url','nb_merchandise_2','my_agg_mean', 'nb_device'] if ADD_LEN_TO_GROUPBY else ['time','nb_merchandise_6', 'nb_merchandise_5', 'nb_auction', 'nb_merchandise_2']
classifier2.compute_all("svc",
                        #important_cols=_ ,
                        test_size=0.2)
```

    model loader: smote data modeling
    feature_eng =  True
    oversample =  False method= None
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.2
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~svc~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ranking = [('nb_country_y', 1), ('nb_merchandise_5_y', 1), ('nb_bid_y', 1), ('my_agg_sum', 1), ('nb_device_y', 2), ('nb_auction_y', 3), ('nb_device_x', 4), ('nb_merchandise_1_x', 5), ('my_agg_mean', 6), ('nb_bid_x', 7), ('my_agg_std', 8), ('nb_url_y', 9), ('nb_auction_x', 10), ('nb_merchandise_2_y', 11), ('nb_url_x', 12), ('my_agg_max', 13), ('nb_country_x', 14), ('nb_merchandise_2_x', 15), ('nb_merchandise_1_y', 16), ('time', 17), ('non_robot_merchandise', 18), ('nb_merchandise_6_y', 19), ('nb_merchandise_6_x', 20), ('nb_merchandise_5_x', 21), ('nb_ip_x', 22), ('nb_ip_y', 23), ('nb_merchandise_4_x', 24), ('nb_merchandise_4_y', 25), ('nb_merchandise_3_x', 26), ('nb_merchandise_3_y', 27)]
    y_train_proba: 9
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[62  3]
     [ 1 63]]
    - accuracy = 96.90%
    - f1 = 96.92%
    - roc(area under the curve) = 100.00%
    - precision = 95.45%
    - recall = 98.44%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[14  2]
     [ 0 17]]
    - accuracy = 93.94%
    - f1 = 94.44%
    - roc(area under the curve) = 100.00%
    - precision = 89.47%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~~svc: nb_params = 4~~~~~~~~~~~~~~~~
    NOPE !! ~~~~ on test and ~~~~ on train



```python
#lda has always been good when row are removed for merchandises
# but not with all merchandises and all rows i guess
classifier.compute_all("lda", oversample=True, test_size=0.4, remove_cols=['non_robot_merchandise'])
```

    model loader: raw data modeling
    feature_eng =  True
    oversample =  True method= smote
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.4
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~lda~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'solver': 'svd'}
    y_train_proba: 33
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[47  1]
     [ 0 48]]
    - accuracy = 98.96%
    - f1 = 98.97%
    - roc(area under the curve) = 100.00%
    - precision = 97.96%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[30  3]
     [ 0  2]]
    - accuracy = 91.43%
    - f1 = 57.14%
    - roc(area under the curve) = 100.00%
    - precision = 40.00%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~lda: nb_params = 29~~~~~~~~~~~~~~~~



```python
classifier1.compute_all("lda", test_size=0.4) #important le test_size ici
```

    model loader: duplic data modeling
    feature_eng =  True
    oversample =  False method= None
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.4
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~lda~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'solver': 'svd'}
    y_train_proba: 34
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[48  1]
     [ 0 48]]
    - accuracy = 98.97%
    - f1 = 98.97%
    - roc(area under the curve) = 100.00%
    - precision = 97.96%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[29  3]
     [ 0 33]]
    - accuracy = 95.38%
    - f1 = 95.65%
    - roc(area under the curve) = 100.00%
    - precision = 91.67%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~lda: nb_params = 30~~~~~~~~~~~~~~~~



```python
classifier2.compute_all("lda")
```

    model loader: smote data modeling
    feature_eng =  True
    oversample =  False method= None
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.5
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~lda~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'solver': 'svd'}
    y_train_proba: 44
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[40  1]
     [ 0 40]]
    - accuracy = 98.77%
    - f1 = 98.77%
    - roc(area under the curve) = 100.00%
    - precision = 97.56%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[37  3]
     [ 0 41]]
    - accuracy = 96.30%
    - f1 = 96.47%
    - roc(area under the curve) = 100.00%
    - precision = 93.18%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~lda: nb_params = 30~~~~~~~~~~~~~~~~


knn + oversampling


```python
#oversample=True, test_size=0.2 best for the 3-merchandise: good
#oversample=True, test_size=0.4 best for the 3-merchandise: variable results
classifier.compute_all("knn", oversample=True, test_size=0.2)
```

    model loader: raw data modeling
    feature_eng =  True
    oversample =  True method= smote
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.2
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~knn~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'n_neighbors': 2}
    y_train_proba: 3
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[62  2]
     [ 0 64]]
    - accuracy = 98.44%
    - f1 = 98.46%
    - roc(area under the curve) = 100.00%
    - precision = 96.97%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[16  1]
     [ 0  1]]
    - accuracy = 94.44%
    - f1 = 66.67%
    - roc(area under the curve) = 100.00%
    - precision = 50.00%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~knn: nb_params = 30~~~~~~~~~~~~~~~~
    NOPE !! ~~~~ on test and ~~~~ on train



```python
classifier1.compute_all("knn")
```

    model loader: duplic data modeling
    feature_eng =  True
    oversample =  False method= None
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.5
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~knn~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'n_neighbors': 2}
    y_train_proba: 3
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[39  2]
     [ 0 40]]
    - accuracy = 97.53%
    - f1 = 97.56%
    - roc(area under the curve) = 100.00%
    - precision = 95.24%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[37  3]
     [ 0 41]]
    - accuracy = 96.30%
    - f1 = 96.47%
    - roc(area under the curve) = 100.00%
    - precision = 93.18%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~knn: nb_params = 30~~~~~~~~~~~~~~~~



```python
classifier2.compute_all("knn")
```

    model loader: smote data modeling
    feature_eng =  True
    oversample =  False method= None
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.5
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~knn~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'n_neighbors': 2}
    y_train_proba: 3
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[38  3]
     [ 0 40]]
    - accuracy = 96.30%
    - f1 = 96.39%
    - roc(area under the curve) = 100.00%
    - precision = 93.02%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[36  4]
     [ 0 41]]
    - accuracy = 95.06%
    - f1 = 95.35%
    - roc(area under the curve) = 100.00%
    - precision = 91.11%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~knn: nb_params = 30~~~~~~~~~~~~~~~~
    GREAT !! gain 1.00% on test and gain 0.40% on train



```python
# oublie le boosting hh
classifier.compute_all("ada", test_size=0.4)
```

    model loader: raw data modeling
    feature_eng =  True
    oversample =  True method= smote
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.4
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~ada~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'n_estimators': 25}
    y_train_proba: 17
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[48  0]
     [ 0 48]]
    - accuracy = 100.00%
    - f1 = 100.00%
    - roc(area under the curve) = 100.00%
    - precision = 100.00%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[30  3]
     [ 0  2]]
    - accuracy = 91.43%
    - f1 = 57.14%
    - roc(area under the curve) = 100.00%
    - precision = 40.00%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~ada: nb_params = 30~~~~~~~~~~~~~~~~



```python
classifier2.compute_all("ada", feature_eng=False)
```

    model loader: smote data modeling
    feature_eng =  False
    oversample =  False method= None
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.5
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~ada~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'n_estimators': 25}
    y_train_proba: 7
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[41  0]
     [ 0 40]]
    - accuracy = 100.00%
    - f1 = 100.00%
    - roc(area under the curve) = 100.00%
    - precision = 100.00%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[38  2]
     [ 0 41]]
    - accuracy = 97.53%
    - f1 = 97.62%
    - roc(area under the curve) = 100.00%
    - precision = 95.35%
    - recall = 100.00%
    ~~~~~~~~~~~~~~~ada: nb_params = 30~~~~~~~~~~~~~~~~



```python
# oublie le boosting hh
classifier.compute_all("xgboost", oversample=True, test_size=0.4)#, remove_cols=["time","nb_auction"])
```

    model loader: raw data modeling
    feature_eng =  True
    oversample =  True method= smote
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_device_y', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.4
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~xgboost~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'max_depth': 1, 'n_estimators': 10}
    y_train_proba: 14
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[47  1]
     [ 0 48]]
    - accuracy = 98.96%
    - f1 = 98.97%
    - roc(area under the curve) = 100.00%
    - precision = 97.96%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[30  3]
     [ 0  2]]
    - accuracy = 91.43%
    - f1 = 57.14%
    - roc(area under the curve) = 100.00%
    - precision = 40.00%
    - recall = 100.00%
    ~~~~~~~~~~~~~xgboost: nb_params = 30~~~~~~~~~~~~~~



```python
classifier2.compute_all("xgboost", feature_eng=False)
```

    model loader: smote data modeling
    feature_eng =  False
    oversample =  False method= None
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    important_cols =  ['nb_device_x', 'time', 'nb_device_y', 'nb_merchandise_4_x', 'non_robot_merchandise', 'nb_url_y', 'my_agg_max', 'nb_country_x', 'nb_merchandise_5_x', 'nb_merchandise_3_x', 'nb_ip_y', 'nb_country_y', 'nb_bid_x', 'nb_merchandise_1_x', 'nb_merchandise_3_y', 'nb_merchandise_5_y', 'nb_auction_y', 'my_agg_std', 'nb_merchandise_6_y', 'nb_bid_y', 'my_agg_mean', 'nb_auction_x', 'my_agg_sum', 'nb_merchandise_1_y', 'nb_merchandise_2_x', 'nb_merchandise_2_y', 'nb_merchandise_6_x', 'nb_merchandise_4_y', 'nb_ip_x', 'nb_url_x']
    test_size =  0.5
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~xgboost~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {'max_depth': 5, 'n_estimators': 10}
    y_train_proba: 3
    
    >>>> metriques sur la base de données d'entrainement
    - confusion_matrix
     [[41  0]
     [ 0 40]]
    - accuracy = 100.00%
    - f1 = 100.00%
    - roc(area under the curve) = 100.00%
    - precision = 100.00%
    - recall = 100.00%
    
    >>>> metriques sur la base de données de test
    - confusion_matrix
     [[39  1]
     [ 0 41]]
    - accuracy = 98.77%
    - f1 = 98.80%
    - roc(area under the curve) = 100.00%
    - precision = 97.62%
    - recall = 100.00%
    ~~~~~~~~~~~~~xgboost: nb_params = 30~~~~~~~~~~~~~~



```python
assert 0, "end of slideshow ! by hermann :)"
```


```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
print(X.shape)
knn = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
sfs.fit(X, y)


sfs.get_support()

sfs.transform(X).shape

```


```python
import pandas as pd
print(pd.Series([0,0,0, 1,2,5, 7,8,9]).describe())
```

# Méthode 2 (depreciated)


```python
TARGET_COL_NAME = "outcome"
ORIGINAL_COLS = ["bidder_id"]
CATEGORICAL_COLS = new_cols
MULTI_CAT_COLS = ["auction", "device", "country", "ip", "url"]
CONTINOUS_COLS = ["time"]
#df = df[ORIGINAL_COLS+CATEGORICAL_COLS + MULTI_CAT_COLS + CONTINOUS_COLS + [TARGET_COL_NAME]]
assert set(ORIGINAL_COLS+CATEGORICAL_COLS + MULTI_CAT_COLS + CONTINOUS_COLS + [TARGET_COL_NAME]) == set(df.columns)
```


```python
# set aside 20% of train and test data for evaluation
X = df.drop(TARGET_COL_NAME,axis=1)
y = df[TARGET_COL_NAME]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state=10)
```


```python
df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)
df_train.head()
```

## bidder_id


```python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_train.bidder_id = le.fit_transform(df_train.bidder_id)
```

## merchandise


```python
df_train.describe()
```

## country

### traitement des cellules vides

### histogramme

histogramme de country


```python
sns.histplot(x="country", data=df_train, hue="outcome")
```


```python
sns.histplot(x="country", data=df_train, hue="outcome", cumulative=True)
```


```python
sorted([(elt,len(df_train[df_train.country==elt])) for elt in df_train.country.unique()], key=lambda x: x[1] ,reverse=True)
```

histogramme du nombre de valeurs par pays


```python
ss = df_train.groupby("country").country.count() #ss = df.country.value_counts()
sns.histplot(ss.to_numpy())
```


```python
ss_ = (ss>70).astype('int')
sns.histplot(ss_)
np.corrcoef(df_train.country.apply(lambda x: ss_[x]), df_train.outcome)[0,1]
```


```python
dd = {}
for i in range(0,40000,1000):
  ss_ = (ss>i).astype('int')
  dd[i] = np.corrcoef(df_train.country.apply(lambda x: ss_[x]), df_train.outcome)[0,1]
  if i%1000 ==0: print(f"{100*i/40000}% --> {dd[i]}")

dd
```

no corr !

another solution: mean encoding

## mean encoding


```python
df_train_enc = df_train.copy()
for col_name in MULTI_CAT_COLS:
  dd = df_train.groupby(col_name)[TARGET_COL_NAME].mean()
  print(col_name, ((dd*2 - 1).abs()!=1).sum(), ((dd*2 - 1).abs()==1).sum())
  df_train_enc[col_name] = df_train[col_name].apply(lambda x: dd[x] )
  df_test_enc[col_name] = df_test[col_name].apply(lambda x: dd[x] )

df_train_enc
```


```python
dd = df.groupby("country").agg({"outcome":lambda x: len(x[x==1])/len(x) }).outcome
df["country_meanu_enc"] = df.country.apply(dd.get)
del df["country_meanu_enc"]
```

# load model into filestorage


```python
import joblib

# save
joblib.dump(log_reg, "log_reg.pkl")

# load
clf2 = joblib.load("log_reg.pkl")
```


```python
df_train.head()
```


```python
df_temp = df_train
for elt in df_temp.bidder_id.unique():
  print(f" isbot:{df_temp[df_temp.bidder_id==elt].outcome.unique()[0]}: id:{elt} -> ndevice:{df_temp[df_temp.bidder_id==elt].device.nunique()}-> nmerchandise:{df_temp[df_temp.merchandise==elt].country.nunique()}-> npays:{df_temp[df_temp.bidder_id==elt].country.nunique()}")
```

# memory

- Standardisation (important. instead for doing custom ennoying pipelines)
- Stratified Shuffle Split:
- upsampling: SMOTE (4 neighbors, test_size<0.4) vs duplication vs none [[1]](https://medium.com/@arch.mo2men/imbalanced-dataset-here-are-5-regularization-methods-which-can-help-5acdb8d324e3), [[5]](https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28)
- no downsampling (Delete nearest majority using Tomlinks or ...)
-  metrics for umbalances ( precision, recall, f1, ...): i follow the recall [[4](https://towardsdatascience.com/how-to-deal-with-unbalanced-data-d1d5bad79e72)]
- best treshold for any model[[2]](https://towardsdatascience.com/how-to-deal-with-unbalanced-data-d1d5bad79e72)
- cost sentivite classification through class weight ? [3](https://resources.experfy.com/ai-ml/most-useful-techniques-handle-imbalanced-datasets/) SVC, decision_tree or cost function ? svc, decision_tree
- replace nb_bid with nb_bide_at_the_same_time
- smote on dataset + binarisation (sur test_size=0.5!!!)  is all you need LOL
- still, What about FLE !


```python

```

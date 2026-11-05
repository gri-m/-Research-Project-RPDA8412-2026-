#!/usr/bin/env python3
"""Rebuild the ST10212348 two-source research analysis.

Python 3.10+; install dependencies with:
    python -m pip install numpy pandas scipy scikit-learn matplotlib
Run:
    python Research_Analysis_ST10212348.py
Offline:
    python Research_Analysis_ST10212348.py --data-dir source_data --offline
Quick validation (reduced resampling and forests; not final inference):
    python Research_Analysis_ST10212348.py --quick --out quick_check



Sources:
SohaibDevv (2026), Kaggle, AI and Student Life 2026, version 1, CC0.
https://www.kaggle.com/datasets/sohaibdevv/ai-and-student-life-2026-the-new-normal
Correa, C. (2026), Mendeley Data, version 1, CC BY 4.0.
https://doi.org/10.17632/mcwb2ppdsw.1
Survey transformations: adults only, ordinal codes, binary endorsements.
Source bytes are preserved; exact SHA-256 checks prevent silent source drift.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import logging
import platform
import sys
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, confusion_matrix, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 8412
A_NAME = 'AI_Impact_Student_Life_2026.csv'
B_NAME = 'survey_cleaned_english.csv'
HASHES = {A_NAME:'69940571eb565f86359a29115e6b135f41b8fbd176ca5acd96eec2efd55958df',
 B_NAME:'84e3f83cad10a3e7939e4ffc9b795750afe3d910c2cd1736debbe02a46f0d1cc'}
FREQ = ['Never','Rarely','Occasionally','Almost always','Always']
DEP = ['Not at all','Probably not','Not sure','Probably yes','A lot']
HELP = ['Definitely no','Probably no','Neutral','Probably yes','Definitely yes']
FRAUD = ['Strongly disagree','Disagree','Undecided','Agree','Strongly agree']
GROUPS = ['Low','Moderate','High']
TASK = {'Brainstorming':1,'Exam Prep':1,'Literature Review':2,'Code Debugging':2,'Essay Drafting':3}


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent':'ST10212348 research reproducibility/1.0'})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


def extract_csvs(blob, destination):
    # Do not trust archive paths or extract arbitrary source scripts.
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for member in z.namelist():
            if member.lower().endswith(('.csv','.txt','.r')) and not member.endswith('/'):
                target = destination / Path(member).name
                if target.exists() and target.read_bytes() != z.read(member):
                    raise ValueError(f'Conflicting archive filename: {target.name}')
                target.write_bytes(z.read(member))


def acquire(root, offline):
    root.mkdir(parents=True, exist_ok=True)
    for name in [A_NAME, B_NAME]:
        found = list(root.rglob(name))
        if not found:
            if offline:
                raise FileNotFoundError(f'Place {name} under {root}; or omit --offline to download.')
            logging.info('Downloading %s', name)
            if name == A_NAME:
                blob = fetch('https://www.kaggle.com/api/v1/datasets/download/sohaibdevv/ai-and-student-life-2026-the-new-normal?datasetVersionNumber=1')
                extract_csvs(blob, root)
            else:
                entries = json.loads(fetch('https://data.mendeley.com/public-api/datasets/mcwb2ppdsw/files?folder_id=root&version=1'))
                matches = [e for e in entries if e['filename'].endswith('.zip')]
                if len(matches) != 1:
                    raise ValueError('Expected one survey archive. Download source version 1 manually.')
                extract_csvs(fetch(matches[0]['content_details']['download_url']), root)
            found = list(root.rglob(name))
        if len(found) != 1:
            raise ValueError(f'Expected exactly one {name}; found {len(found)}')
        digest = hashlib.sha256(found[0].read_bytes()).hexdigest()
        if digest != HASHES[name]:
            raise ValueError(f'Source hash mismatch: {name}\nExpected {HASHES[name]}\nActual {digest}')
    return {name:next(root.rglob(name)) for name in HASHES}


def ordinal(s, levels):
    unknown = set(s.dropna()) - set(levels)
    if unknown or s.isna().any():
        raise ValueError(f'Invalid or missing labels in {s.name}: {unknown}')
    return s.map({v:i+1 for i,v in enumerate(levels)}).astype(int)


def prepare(paths):
    a, raw = pd.read_csv(paths[A_NAME]), pd.read_csv(paths[B_NAME])
    if a.shape != (1500,11) or raw.shape != (357,31):
        raise ValueError(f'Unexpected source shapes {a.shape}, {raw.shape}')
    if a.isna().any().any() or not a.Student_ID.is_unique or a.duplicated().any():
        raise ValueError('Synthetic source failed missingness/identifier/duplicate checks')
    for col, lo, hi in [('GPA_Baseline',0,4),('GPA_Post_AI',0,4),('Task_Frequency_Daily',1,10),('Time_Saved_Hours_Weekly',2,15)]:
        if not a[col].between(lo,hi).all(): raise ValueError(f'Out of range: {col}')
    if set(a.Main_Usage_Case) != set(TASK):
        raise ValueError(f'Task labels differ from documented mapping: {set(a.Main_Usage_Case)}')
    a['source_row'] = np.arange(len(a))
    a['GPA_Change'] = (a.GPA_Post_AI-a.GPA_Baseline).round(8)
    a['Use_Group'] = pd.cut(a.Task_Frequency_Daily,[0,3,7,10],labels=GROUPS)
    a['Task_Score'] = a.Main_Usage_Case.map(TASK)
    a['Profile_Score'] = .6*(a.Task_Frequency_Daily-1)/9+.4*(a.Task_Score-1)/2
    a['Profile'] = pd.cut(a.Profile_Score,[-np.inf,.33,.66,np.inf],labels=GROUPS)
    if raw.age_years.isna().any() or raw.duplicated().any(): raise ValueError('Survey failed age/duplicate checks')
    raw['source_row'] = np.arange(len(raw))
    b = raw.loc[raw.age_years>=18].copy()
    if not b.cumulative_gpa_papa.between(0,5).all(): raise ValueError('Invalid survey GPA')
    for original, new, levels in [('ai_use_frequency_for_study','Frequency',FREQ),('perceived_ai_dependence','Dependence',DEP),('ai_helped_understand_topics','Understanding',HELP),('ai_use_in_academic_tasks_is_fraud','Fraud',FRAUD)]:
        b[new] = ordinal(b[original],levels)
    b['Endorsement'] = (b.Dependence>=4).astype(int)
    b['Understands'] = (b.Understanding>=4).astype(int)
    b['Working'] = (b.current_university_condition=='Study and work').astype(int)
    return a,b,raw


class Output:
    def __init__(self, root):
        self.root=root
        for folder in ['tables','figures','processed']:(root/folder).mkdir(parents=True,exist_ok=True)
        self.inventory=[]
    def table(self,name,frame):
        pd.DataFrame(frame).to_csv(self.root/'tables'/f'{name}.csv',index=True)
    def figure(self,number,title,fig=None):
        fig=fig or plt.gcf();fig.suptitle(f'Figure {number}  {title}',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.95])
        for ext in ['png','svg']: fig.savefig(self.root/'figures'/f'figure_{number.replace(".","_")}.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig);self.inventory.append({'number':number,'title':title})


def summary(d, columns):
    s=d[columns].describe().T
    s['mean_ci_low']=s['mean']-stats.t.ppf(.975,s['count']-1)*s['std']/np.sqrt(s['count'])
    s['mean_ci_high']=s['mean']+stats.t.ppf(.975,s['count']-1)*s['std']/np.sqrt(s['count'])
    return s


def holm(p):
    p=np.asarray(p);order=np.argsort(p);ans=np.empty(len(p))
    ans[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return ans


def correlation(x,y,nboot,nperm,seed):
    x,y=np.asarray(x),np.asarray(y);n=len(x);rng=np.random.default_rng(seed)
    rho,p=stats.spearmanr(x,y);boot=[]
    for _ in range(nboot):
        idx=rng.integers(0,n,n)
        if np.unique(x[idx]).size>1 and np.unique(y[idx]).size>1:boot.append(stats.spearmanr(x[idx],y[idx]).statistic)
    ci=np.quantile(boot,[.025,.975])
    if nperm:
        rx,ry=stats.rankdata(x),stats.rankdata(y)
        rx=(rx-rx.mean())/np.linalg.norm(rx-rx.mean());ry=(ry-ry.mean())/np.linalg.norm(ry-ry.mean())
        hits=sum(abs(np.dot(rx,rng.permutation(ry)))>=abs(rho)-1e-12 for _ in range(nperm));p=(hits+1)/(nperm+1)
    return {'rho':rho,'p':p,'ci_low':ci[0],'ci_high':ci[1],'n':n}


def chi(d,x,y):
    table=pd.crosstab(d[x],d[y]);stat,p,df,exp=stats.chi2_contingency(table)
    return {'statistic':stat,'p':p,'df':df,'cramers_v':np.sqrt(stat/(table.values.sum()*min(np.array(table.shape)-1))),'min_expected':exp.min()}


def ols(d,target,numeric,categorical,out,prefix):
    X=pd.concat([d[numeric].astype(float),pd.get_dummies(d[categorical],drop_first=True,dtype=float)],axis=1)
    X.insert(0,'Intercept',1.)
    if X.isna().any().any() or d[target].isna().any(): raise ValueError('OLS fields contain missing data')
    x,y=X.to_numpy(),d[target].to_numpy(float);n,k=x.shape
    beta=np.linalg.lstsq(x,y,rcond=None)[0];pinv=np.linalg.pinv(x.T@x);resid=y-x@beta
    h=np.sum((x@pinv)*x,axis=1)
    if np.linalg.matrix_rank(x)<k or np.any(h>=1-1e-10):raise ValueError('OLS singular or unit leverage design')
    cov=pinv@(x.T@((resid/(1-h))[:,None]**2*x))@pinv
    se=np.sqrt(np.maximum(np.diag(cov),0));crit=stats.t.ppf(.975,n-k)
    check=LinearRegression(fit_intercept=False).fit(x,y).coef_
    np.testing.assert_allclose(beta,check,rtol=1e-7,atol=1e-8)
    out.table(prefix+'_coefficients',pd.DataFrame({'term':X.columns,'coefficient':beta,'HC3_se':se,'ci_low':beta-crit*se,'ci_high':beta+crit*se,'p_t':2*stats.t.sf(abs(beta/se),n-k)}))
    vifs=[]
    for j in range(1,k):
        aux=LinearRegression().fit(np.delete(x,j,axis=1),x[:,j]);r=aux.score(np.delete(x,j,axis=1),x[:,j])
        vifs.append({'term':X.columns[j],'VIF':1/(1-r) if r<1 else np.inf})
    out.table(prefix+'_VIF',vifs)
    r2=1-resid@resid/np.sum((y-y.mean())**2)
    diag={'n':n,'parameters':k,'rank':np.linalg.matrix_rank(x),'r_squared':r2,'adjusted_r_squared':1-(1-r2)*(n-1)/(n-k),'max_leverage':h.max()}
    out.table(prefix+'_diagnostics',[diag]);return x@beta,resid,diag


def pipeline(model,X):
    num=X.select_dtypes(include='number').columns.tolist();cat=[c for c in X if c not in num]
    pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
    return Pipeline([('preprocess',pre),('model',model)])


def model_analysis(d,classification,out,nboot,quick):
    prefix='B' if classification else 'A'
    if classification:
        columns=['age_years','current_program_term','cumulative_gpa_papa','n_ai_tools','gender','academic_program','current_university_condition','independent_study_time','procrastination_frequency','ai_use_frequency_for_study','prompt_engineering_knowledge']
        target='Endorsement';cv=StratifiedKFold(5,shuffle=True,random_state=SEED);scoring='roc_auc'
        models={'Prevalence baseline':(DummyClassifier(strategy='prior'),{}),'Logistic regression':(LogisticRegression(max_iter=3000),{'model__C':[.1,1,10]}),'Random forest':(RandomForestClassifier(n_estimators=30 if quick else 300,random_state=SEED,n_jobs=1),{'model__max_depth':[4,None],'model__min_samples_leaf':[5,10]})}
    else:
        columns=['Task_Frequency_Daily','Time_Saved_Hours_Weekly','GPA_Baseline','Age','Main_Usage_Case','Major','Primary_AI_Tool','AI_Ethics_Concern'];target='GPA_Change';cv=KFold(5,shuffle=True,random_state=SEED);scoring='neg_root_mean_squared_error'
        models={'Mean baseline':(DummyRegressor(strategy='mean'),{}),'Linear regression':(LinearRegression(),{}),'Ridge':(Ridge(),{'model__alpha':[.1,1,10,100]}),'Random forest':(RandomForestRegressor(n_estimators=30 if quick else 250,random_state=SEED,n_jobs=1),{'model__max_depth':[4,None],'model__min_samples_leaf':[5,15]})}
    X,y=d[columns],d[target]
    train,test=train_test_split(np.arange(len(d)),test_size=.25 if classification else .2,random_state=SEED,stratify=y if classification else None)
    memberships=pd.DataFrame({'source_row':d.source_row.to_numpy(),'split':'train'});memberships.loc[test,'split']='test';out.table(prefix+'_split',memberships)
    rows=[];preds={};folds=[]
    for name,(model,grid) in models.items():
        pipe=pipeline(model,X)
        if grid:
            fit=GridSearchCV(pipe,grid,scoring=scoring,cv=cv,n_jobs=1,error_score='raise').fit(X.iloc[train],y.iloc[train]);est=fit.best_estimator_;cvscore=fit.best_score_;params=fit.best_params_
        else:
            scores=cross_val_score(pipe,X.iloc[train],y.iloc[train],scoring=scoring,cv=cv,error_score='raise');cvscore=scores.mean();est=pipe.fit(X.iloc[train],y.iloc[train]);params={}
        scores=cross_val_score(est,X.iloc[train],y.iloc[train],scoring=scoring,cv=cv,error_score='raise')
        folds.extend({'model':name,'fold':i+1,'score':v if classification else -v} for i,v in enumerate(scores))
        pred=est.predict_proba(X.iloc[test])[:,1] if classification else est.predict(X.iloc[test]);preds[name]=pred
        row={'model':name,'cv_score':cvscore if classification else -cvscore,'parameters':json.dumps(params)}
        truth=y.iloc[test].to_numpy()
        if classification:
            hard=(pred>=.5).astype(int)
            row.update(AUC=roc_auc_score(truth,pred),balanced_accuracy=balanced_accuracy_score(truth,hard),precision=precision_score(truth,hard,zero_division=0),recall=recall_score(truth,hard),F1=f1_score(truth,hard),Brier=brier_score_loss(truth,pred))
        else:row.update(RMSE=np.sqrt(mean_squared_error(truth,pred)),MAE=mean_absolute_error(truth,pred),R_squared=r2_score(truth,pred))
        rows.append(row)
    scores=pd.DataFrame(rows);out.table(prefix+'_model_performance',scores);out.table(prefix+'_cv_folds',folds)
    out.table(prefix+'_heldout_predictions',pd.DataFrame({'source_row':d.iloc[test].source_row.to_numpy(),'actual':truth,**preds}))
    rng=np.random.default_rng(SEED+91);intervals=[]
    for name,pred in preds.items():
        values=[]
        for _ in range(nboot):
            idx=rng.integers(0,len(test),len(test))
            if classification:
                if np.unique(truth[idx]).size<2:continue
                values.append(roc_auc_score(truth[idx],pred[idx]))
            else:values.append(np.sqrt(np.mean((truth[idx]-pred[idx])**2))-np.sqrt(np.mean((truth[idx]-preds['Mean baseline'][idx])**2)))
        low,high=np.quantile(values,[.025,.975]);intervals.append({'model':name,'metric':'AUC' if classification else 'RMSE minus mean baseline','low':low,'high':high})
    out.table(prefix+'_heldout_bootstrap_intervals',intervals)
    if classification:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for name,p in preds.items():
            fpr,tpr,_=roc_curve(truth,p);axes[0].plot(fpr,tpr,label=name)
        axes[0].plot([0,1],[0,1],'k--');axes[0].legend(fontsize=8);axes[0].set(xlabel='False positive rate',ylabel='True positive rate')
        # Choose the non-baseline model using training CV only.
        chosen=scores.loc[scores.model!='Prevalence baseline'].sort_values('cv_score',ascending=False).iloc[0]['model']
        cm=confusion_matrix(truth,(preds[chosen]>=.5).astype(int),labels=[0,1]);axes[1].imshow(cm,cmap='Blues')
        for (i,j),value in np.ndenumerate(cm):axes[1].text(j,i,str(value),ha='center',va='center')
        axes[1].set(xticks=[0,1],yticks=[0,1],xticklabels=['Other','Endorse'],yticklabels=['Other','Endorse'],xlabel='Predicted',ylabel='Observed',title=chosen)
        out.figure('4.20','Held-out ROC curves and selected confusion matrix',fig)
    else:
        fig,ax=plt.subplots(figsize=(8,4));pos=np.arange(len(scores));ax.bar(pos-.18,scores.cv_score,.36,label='Training CV RMSE');ax.bar(pos+.18,scores.RMSE,.36,label='Held-out RMSE');ax.set_xticks(pos,scores.model,rotation=15);ax.set_ylabel('GPA change RMSE');ax.legend();out.figure('4.14','Cross-validation and held-out regression errors',fig)
    return scores.to_dict('records')


def bars(out,number,title,counts,ylabel='Records'):
    fig,ax=plt.subplots(figsize=(8,4));counts.plot.bar(ax=ax,color='#3576a8');ax.set_ylabel(ylabel);ax.tick_params(axis='x',rotation=20);out.figure(number,title,fig)


def group_plot(out,number,title,d,group,value):
    cols=[g for g in GROUPS if g in set(d[group].astype(str))]
    fig,ax=plt.subplots(figsize=(7,4));ax.boxplot([d.loc[d[group]==g,value] for g in cols],tick_labels=cols,showmeans=True);ax.set(xlabel=group,ylabel=value);out.figure(number,title,fig)


def workflow(out,number,title,labels):
    fig,ax=plt.subplots(figsize=(9,4));ax.set_axis_off()
    for i,label in enumerate(labels):
        y=.88-i*.24;ax.text(.5,y,label,ha='center',va='center',bbox={'boxstyle':'round,pad=.5','fc':'#eef3f7','ec':'#3576a8'},transform=ax.transAxes)
        if i<len(labels)-1:ax.annotate('',xy=(.5,y-.17),xytext=(.5,y-.06),xycoords='axes fraction',arrowprops={'arrowstyle':'->'})
    out.figure(number,title,fig)


def run(args):
    out=Output(args.out);paths=acquire(args.data_dir,args.offline);a,b,raw=prepare(paths)
    nboot=100 if args.quick else 2000;nperm=199 if args.quick else 9999
    provenance={name:{'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in paths.items()}
    out.table('source_audit',[{'dataset':'A synthetic','rows':len(a),'original_columns':11},{'dataset':'B original survey','rows':len(raw),'original_columns':31},{'dataset':'B adults','rows':len(b),'original_columns':31}])
    for name,d in [('A',a),('B',raw)]:out.table(name+'_missingness',d.isna().sum().to_frame('missing'))
    a.to_csv(args.out/'processed'/'A_derived.csv',index=False);b.to_csv(args.out/'processed'/'B_adults_derived.csv',index=False)
    workflow(out,'1.1','Support and substitution as contextual modes',['AI supports a task: explanation, feedback, checking','Task context and independent student effort matter','AI substitutes for task performance: requires direct evidence'])
    workflow(out,'2.1','Conceptual relationships and measurement limits',['Recorded AI use and task context','Time savings, grades and perceived dependence','Unaided learning and retention remain unmeasured'])
    workflow(out,'3.1','Separate analytical roles of the two sources',['Dataset A: 1,500 synthetic profiles; Dataset B: adult survey','Separate validation, measures and statistical analyses','Conceptual comparison; no pooling or causal inference'])
    workflow(out,'3.2','Training and evaluation workflow',['Validate data and define outcomes','Reserve test records; preprocess and tune within training folds','Evaluate fixed models on held-out records'])
    out.table('A_numerical_summary',summary(a,['Age','Task_Frequency_Daily','GPA_Baseline','GPA_Post_AI','GPA_Change','Time_Saved_Hours_Weekly','Career_Confidence_Score']))
    for field,num,title in [('Use_Group','4.1','Daily AI use groups'),('Main_Usage_Case','4.2','Primary academic AI use cases')]:
        counts=a[field].value_counts(sort=False);out.table('A_'+field,counts.to_frame('n').assign(percentage=100*counts/len(a)));bars(out,num,title,counts)
    for group,field,num,title in [('Use_Group','Time_Saved_Hours_Weekly','4.3','Weekly time saved by use group'),('Use_Group','GPA_Change','4.4','GPA change by use group'),('Profile','GPA_Change','4.6','GPA change by exploratory usage profile'),('Profile','Time_Saved_Hours_Weekly','4.7','Weekly time saved by exploratory usage profile'),('Profile','Career_Confidence_Score','4.8','Career confidence by exploratory usage profile')]:
        out.table('A_'+group+'_'+field,a.groupby(group,observed=False)[field].agg(['count','mean','std','median','min','max']));group_plot(out,num,title,a,group,field)
    for x,y,num,title in [('Use_Group','Task_Score','4.5','Task orientation by use group'),('Profile','AI_Ethics_Concern','4.9','Ethics concern by exploratory usage profile')]:
        tab=pd.crosstab(a[x],a[y]);out.table('A_'+x+'_'+y,tab);fig,ax=plt.subplots(figsize=(8,4));tab.div(tab.sum(axis=1),axis=0).plot.bar(stacked=True,ax=ax);ax.set_ylabel('Proportion within group');out.figure(num,title,fig)
    tests=[]
    for x,y in [('Task_Frequency_Daily','Time_Saved_Hours_Weekly'),('Task_Frequency_Daily','GPA_Change')]:tests.append({'test':x+' vs '+y,**correlation(a[x],a[y],nboot,0,SEED)})
    for x,y in [('Use_Group','Task_Score'),('Use_Group','AI_Ethics_Concern'),('Main_Usage_Case','AI_Ethics_Concern')]:tests.append({'test':x+' vs '+y,**chi(a,x,y)})
    t=pd.DataFrame(tests);t['p_Holm']=holm(t.p);out.table('A_primary_tests',t)
    group_tests=[]
    for group in ['Use_Group','Profile']:
        for val in ['GPA_Change','Time_Saved_Hours_Weekly','Career_Confidence_Score']:
            groups=[g[val].to_numpy() for _,g in a.groupby(group,observed=True)];h,p=stats.kruskal(*groups);group_tests.append({'group':group,'outcome':val,'H':h,'p':p,'epsilon_squared':max(0,(h-len(groups)+1)/(len(a)-len(groups)))})
    out.table('A_exploratory_group_tests',group_tests)
    out.table('A_original_task_outcomes',a.groupby('Main_Usage_Case')[['GPA_Change','Time_Saved_Hours_Weekly','Career_Confidence_Score']].agg(['count','mean','std']))
    fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].hist(a.Task_Frequency_Daily,bins=np.arange(.5,11.5));ax[0].set_xlabel('Daily tasks');ax[1].hist(a.GPA_Change,bins=25);ax[1].set_xlabel('GPA change');out.figure('4.10','Daily frequency and GPA change distributions',fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].scatter(a.GPA_Baseline,a.GPA_Change,s=7,alpha=.3);ax[0].set(xlabel='Baseline GPA',ylabel='GPA change');ax[1].hist(a.GPA_Post_AI,bins=25);ax[1].set_xlabel('Post-AI-labelled GPA');out.figure('4.11','GPA headroom and upper-bound concentration',fig)
    sens=[]
    for name,w,cuts,mapping in [('Original',.6,[.33,.66],TASK),('Exact thirds',.6,[1/3,2/3],TASK),('Equal weights',.5,[.33,.66],TASK),('Frequency weight 0.4',.4,[.33,.66],TASK),('Alternative tasks',.6,[.33,.66],{**TASK,'Exam Prep':2,'Essay Drafting':2})]:
        score=w*(a.Task_Frequency_Daily-1)/9+(1-w)*(a.Main_Usage_Case.map(mapping)-1)/2;cats=pd.cut(score,[-np.inf,*cuts,np.inf],labels=GROUPS);h,p=stats.kruskal(*[a.loc[cats==g,'GPA_Change'] for g in GROUPS]);sens.append({'scheme':name,'agreement':np.mean(cats==a.Profile),'H':h,'p':p,**cats.value_counts().to_dict()})
    out.table('A_index_sensitivity',sens)
    cols=['Task_Frequency_Daily','GPA_Change','Time_Saved_Hours_Weekly','Career_Confidence_Score','GPA_Baseline'];corr=a[cols].corr(method='spearman');out.table('A_rank_correlations',corr)
    fig,ax=plt.subplots(figsize=(8,6));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_xticks(range(len(cols)),cols,rotation=50,ha='right');ax.set_yticks(range(len(cols)),cols)
    for (i,j),v in np.ndenumerate(corr):ax.text(j,i,f'{v:.2f}',ha='center',va='center')
    fig.colorbar(im,ax=ax);out.figure('4.12','Selected rank correlations in synthetic profiles',fig)
    fitted,resid,diagA=ols(a,'GPA_Change',['Task_Frequency_Daily','Time_Saved_Hours_Weekly','GPA_Baseline','Age'],['Main_Usage_Case','Major'],out,'A_OLS')
    fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].scatter(fitted,resid,s=6,alpha=.3);ax[0].axhline(0,color='black');ax[0].set(xlabel='Fitted GPA change',ylabel='Residual');stats.probplot(resid,plot=ax[1]);out.figure('4.13','OLS residual diagnostics for synthetic profiles',fig)
    modelA=model_analysis(a,False,out,nboot,args.quick)
    out.table('B_numerical_summary',summary(b,['age_years','current_program_term','cumulative_gpa_papa','n_ai_tools']))
    freq=b.ai_use_frequency_for_study.value_counts().reindex(FREQ,fill_value=0);dep=b.perceived_ai_dependence.value_counts().reindex(DEP,fill_value=0)
    out.table('B_frequency_counts',freq.to_frame('n'));out.table('B_dependence_counts',dep.to_frame('n'))
    fig,ax=plt.subplots(1,2,figsize=(11,4));freq.plot.bar(ax=ax[0]);dep.plot.bar(ax=ax[1]);[x.tick_params(axis='x',rotation=35) for x in ax];out.figure('4.15','Use and perceived dependence in the adult survey',fig)
    tab=pd.crosstab(b.ai_use_frequency_for_study,b.perceived_ai_dependence).reindex(index=FREQ,columns=DEP,fill_value=0);out.table('B_frequency_dependence',tab)
    fig,ax=plt.subplots(figsize=(9,4));tab.div(tab.sum(axis=1),axis=0).plot.bar(stacked=True,ax=ax);ax.set_ylabel('Proportion');ax.legend(fontsize=7);out.figure('4.16','Dependence responses across use frequencies',fig)
    pairs=[('Frequency','Dependence'),('Frequency','cumulative_gpa_papa'),('Dependence','cumulative_gpa_papa'),('Frequency','Fraud'),('Dependence','Fraud')]
    testsB=pd.DataFrame([{'test':x+' vs '+y,**correlation(b[x],b[y],nboot,nperm,SEED+i)} for i,(x,y) in enumerate(pairs)]);testsB['p_Holm']=holm(testsB.p);out.table('B_primary_tests',testsB)
    out.table('B_exploratory_frequency_understanding',[correlation(b.Frequency,b.Understanding,nboot,nperm,SEED+10)])
    fig,ax=plt.subplots(figsize=(8,4));ax.errorbar(testsB.rho,np.arange(5),xerr=[testsB.rho-testsB.ci_low,testsB.ci_high-testsB.rho],fmt='o');ax.set_yticks(range(5),testsB.test);ax.axvline(0,color='gray');ax.set_xlabel('Spearman correlation and percentile bootstrap 95% interval');out.figure('4.17','Adult survey associations and uncertainty',fig)
    joint=pd.crosstab(b.Understands,b.Endorsement).reindex(index=[0,1],columns=[0,1],fill_value=0);out.table('B_joint_understanding_dependence',joint)
    fig,ax=plt.subplots(figsize=(6,4));ax.imshow(joint,cmap='Blues')
    for (i,j),v in np.ndenumerate(joint):ax.text(j,i,str(v),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Other','Endorse'],yticklabels=['Other','Endorse'],xlabel='Dependence',ylabel='Improved understanding');out.figure('4.18','Joint endorsement of understanding and dependence',fig)
    out.table('B_GPA_by_dependence',b.groupby('perceived_ai_dependence').cumulative_gpa_papa.agg(['count','mean','std','median']))
    fig,ax=plt.subplots(figsize=(8,4));ax.boxplot([b.loc[b.perceived_ai_dependence==g,'cumulative_gpa_papa'] for g in DEP],tick_labels=DEP,showmeans=True);ax.set_ylabel('Current GPA, source scale 0–5');out.figure('4.19','Current GPA by perceived dependence',fig)
    _,_,diagB=ols(b,'cumulative_gpa_papa',['Frequency','Dependence','age_years','current_program_term','Working'],['academic_program'],out,'B_OLS')
    modelB=model_analysis(b,True,out,nboot,args.quick)
    rng=np.random.default_rng(SEED+20);means=[a.GPA_Change.iloc[rng.integers(0,len(a),len(a))].mean() for _ in range(nboot)]
    paired={'mean_change':a.GPA_Change.mean(),'bootstrap_ci':np.quantile(means,[.025,.975]).tolist(),'paired_t':dict(zip(['statistic','p'],stats.ttest_rel(a.GPA_Post_AI,a.GPA_Baseline)[:2])),'wilcoxon':dict(zip(['statistic','p'],stats.wilcoxon(a.GPA_Change)[:2]))}
    subset=a[a.GPA_Baseline<=3.7];out.table('A_headroom_sensitivity',[{'n':len(subset),'mean_change':subset.GPA_Change.mean(),**correlation(subset.Task_Frequency_Daily,subset.GPA_Change,nboot,0,SEED)}])
    out.table('figure_inventory',out.inventory)
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__}
    result={'reconstructed_script':True,'quick_run':args.quick,'seed':SEED,'bootstrap_resamples':nboot,'permutations':nperm,'versions':versions,'provenance':provenance,'sample_sizes':{'A':len(a),'B_raw':len(raw),'B_adults':len(b)},'synthetic_paired_summary':paired,'OLS_A':diagA,'OLS_B':diagB,'models_A':modelA,'models_B':modelB,'survey_dependence_endorsements':int(b.Endorsement.sum()),'joint_endorsements':int(((b.Endorsement==1)&(b.Understands==1)).sum()),'figures':len(out.inventory)}
    def convert(v):
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    (args.out/'results.json').write_text(json.dumps(result,indent=2,default=convert),encoding='utf-8')
    (args.out/'requirements_used.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items() if k!='python')+'\n',encoding='utf-8')
    logging.info('Completed: %s figures, %s CSV tables. Outputs: %s',len(out.inventory),len(list((args.out/'tables').glob('*.csv'))),args.out.resolve())


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir',type=Path,default=Path('source_data'))
    p.add_argument('--out',type=Path,default=Path('research_outputs'))
    p.add_argument('--offline',action='store_true');p.add_argument('--quick',action='store_true')
    args=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    try:run(args)
    except Exception:logging.exception('Analysis stopped; incomplete outputs must not be treated as final.');sys.exit(1)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Reproduce Chapter 4. Run in this directory: python train_models.py.
Fixed group-disjoint held-out fold; tuning/selection use only development CV.
"""
from pathlib import Path
import json, hashlib, os, platform, sys, warnings
import importlib.metadata as md
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score,balanced_accuracy_score,cohen_kappa_score,accuracy_score,confusion_matrix,classification_report
from xgboost import XGBClassifier
from mord import LogisticAT
from feature_engineering import EngineerFeatures,NUMERIC,CATEGORICAL,RAW_FEATURES

ROOT=Path(__file__).resolve().parent
# Pipeline override: read cleaned CSVs from GMPP_CLEANED_DIR and write artefacts
# to GMPP_OUTPUT_DIR. If those are unset, behaviour matches the original script.
DATA_DIR=Path(os.environ.get('GMPP_CLEANED_DIR', str(ROOT.parent/'gmpp-data')))
OUT=Path(os.environ.get('GMPP_OUTPUT_DIR', str(ROOT)))
OUT.mkdir(parents=True, exist_ok=True)
FIG=OUT/'figures'; SHOTS=OUT/'screenshots'
for p in [FIG,SHOTS]:p.mkdir(exist_ok=True)
SEED=42
LABELS=['Red','Amber','Green']; MAP={v:i for i,v in enumerate(LABELS)}
COLORS=['#DC3545','#E7A400','#159A70']  # DCA Red/Amber/Green only
METRIC_COLORS=['#1F4E79','#4C78A8','#6C757D','#7B2D8E']  # metrics: not DCA colours
plt.rcParams.update({'font.size':12,'axes.titlesize':17,'axes.labelsize':13,'figure.dpi':130,'savefig.dpi':220,'axes.spines.top':False,'axes.spines.right':False})

def savefig(path):
    plt.tight_layout();plt.savefig(path,bbox_inches='tight',facecolor=plt.gcf().get_facecolor());plt.close()

def output_image(title,lines,path):
    fig,ax=plt.subplots(figsize=(13,max(4,1.2+len(lines)*.35)))
    fig.patch.set_facecolor('#11203A');ax.set_facecolor('#11203A');ax.axis('off')
    ax.text(.02,.97,title,transform=ax.transAxes,va='top',color='#49E3BD',fontsize=20,weight='bold')
    ax.text(.02,.85,'\n'.join(lines),transform=ax.transAxes,va='top',color='white',fontsize=12,fontfamily='monospace',linespacing=1.7)
    savefig(path)

def scores(y,p):
    return {'macro_f1':float(f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0)),
      'balanced_accuracy':float(balanced_accuracy_score(y,p)), 'cohen_kappa':float(cohen_kappa_score(y,p)),
      'accuracy':float(accuracy_score(y,p)), 'confusion_matrix':confusion_matrix(y,p,labels=[0,1,2]).tolist(),
      'classification_report':classification_report(y,p,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0)}

def counts(y):return {v:int(np.sum(np.asarray(y)==i)) for i,v in enumerate(LABELS)}

def pipeline(estimator):
    nums=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())])
    preprocess=ColumnTransformer([('num',nums,NUMERIC),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL)])
    return Pipeline([('engineer',EngineerFeatures()),('preprocess',preprocess),('model',estimator)])

def main():
    inputs=sorted(DATA_DIR.glob('cleaned-gmpp-*.csv'))
    D=pd.concat([pd.read_csv(p).assign(source_file=p.name,source_row=lambda d:np.arange(len(d))+2) for p in inputs],ignore_index=True)
    D['IPA Delivery Confidence']=D['IPA Delivery Confidence'].str.strip().str.title()
    assert set(D['IPA Delivery Confidence'])==set(LABELS)
    initial=len(D); missing=D['Major Projects ID'].isna() | D['Major Projects ID'].fillna('').str.strip().eq('')
    D.loc[missing].to_csv(OUT/'excluded_missing_project_id.csv',index=False)
    E=EngineerFeatures().transform(D)
    audit={'input_rows':initial,'all_input_class_counts':D['IPA Delivery Confidence'].value_counts().to_dict(),
      'rows_per_source':D['source_file'].value_counts().sort_index().to_dict(),
      'raw_missing_counts':D[RAW_FEATURES+['Major Projects ID']].isna().sum().to_dict(),
      'engineered_missing_counts_all_rows':E.isna().sum().to_dict(),
      'category_counts':E['Annual Report Category'].value_counts().to_dict(),
      'excluded_missing_id':int(missing.sum()),
      'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    D=D.loc[~missing].copy().reset_index(drop=True)
    # Remove the reporting/vintage suffix, retain original department and project number.
    D['project_group']=D['Major Projects ID'].str.strip().str.upper().str.replace(r'_\d{4}-Q\d+$','',regex=True)
    assert D['project_group'].str.fullmatch(r'[A-Z]+_\d+').all()
    X=D[RAW_FEATURES];y=D['IPA Delivery Confidence'].map(MAP).to_numpy();g=D['project_group'].to_numpy()
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    train,test=next(outer.split(X,y,g))
    assert not set(g[train])&set(g[test])
    inner=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED+1).split(X.iloc[train],y[train],g[train]))
    for a,b in inner: assert not set(g[train][a])&set(g[train][b])
    for a,b in inner: assert len(set(y[train][a]))==3 and len(set(y[train][b]))==3
    audit.update({'eligible_rows':len(D),'unique_project_groups':len(set(g)), 'train_rows':len(train),'test_rows':len(test),
      'train_groups':len(set(g[train])),'test_groups':len(set(g[test])),'group_overlap':0,
      'train_class_counts':counts(y[train]),'test_class_counts':counts(y[test]),
      'eligible_class_counts':counts(y),'group_definition':'Original ID prefix department_projectnumber; remove _YYYY-QN suffix',
      'train_majority_class':LABELS[int(np.bincount(y[train]).argmax())]})
    D['split']='train';D.loc[test,'split']='test'
    D[['source_file','source_row','Major Projects ID','project_group','IPA Delivery Confidence','split']].to_csv(OUT/'split_manifest.csv',index=False)
    output_image('01 | Real dataset audit',[
      f"Input rows: {initial} across {len(inputs)} annual cleaned files",
      f"All rows, Red / Amber / Green: {list(audit['all_input_class_counts'].get(k,0) for k in LABELS)}",
      f"Missing project IDs excluded: {missing.sum()}; eligible rows: {len(D)}",
      f"Unique project groups: {len(set(g))}",
      f"Development: {len(train)} rows; held out: {len(test)} rows; overlap: 0 groups",
      f"Development classes: {counts(y[train])}",f"Held-out classes: {counts(y[test])}",
      'Costs/benefits: GBP millions; exempt text -> missing, never FOI numbers.',
      'Mixed dates parsed explicitly; named ranges use Mid when available.',
      'All learned imputation/encoding/scaling fitted inside CV pipelines.'
    ],SHOTS/'01_data_and_split.png')
    print(json.dumps(audit,indent=2))
    configs={
      'Random Forest':(RandomForestClassifier(n_estimators=350,class_weight='balanced',random_state=SEED,n_jobs=1),{'model__max_depth':[None,6,12],'model__min_samples_leaf':[2,5]}),
      'XGBoost':(XGBClassifier(objective='multi:softprob',num_class=3,n_estimators=250,learning_rate=.05,subsample=.85,colsample_bytree=.85,eval_metric='mlogloss',random_state=SEED,n_jobs=1),{'model__max_depth':[2,4],'model__min_child_weight':[1,5]}),
      'Ordinal Logistic':(LogisticAT(max_iter=3000),{'model__alpha':[.1,1.,10.,100.]})}
    results={};fitted={}; cvrows=[]
    baseline=pipeline(DummyClassifier(strategy='constant',constant=MAP['Amber'])).fit(X.iloc[train],y[train])
    results['Always Amber']=scores(y[test],baseline.predict(X.iloc[test]))
    results['Always Amber']['cv_macro_f1_mean']=float(np.mean([f1_score(y[train][b],np.full(len(b),1),average='macro',labels=[0,1,2],zero_division=0) for a,b in inner]))
    fitted['Always Amber']=baseline
    for name,(est,grid) in configs.items():
        print('TUNING',name,flush=True)
        search=GridSearchCV(pipeline(est),grid,scoring='f1_macro',cv=inner,n_jobs=2,refit=True,error_score='raise',return_train_score=False)
        search.fit(X.iloc[train],y[train]); fitted[name]=search.best_estimator_
        pred=search.predict(X.iloc[test]);results[name]=scores(y[test],pred)
        results[name].update({'cv_macro_f1_mean':float(search.best_score_),'cv_macro_f1_std':float(search.cv_results_['std_test_score'][search.best_index_]),'best_parameters':search.best_params_})
        for params,mean,std in zip(search.cv_results_['params'],search.cv_results_['mean_test_score'],search.cv_results_['std_test_score']):
            cvrows.append({'model':name,'parameters':json.dumps(params),'mean_macro_f1':mean,'std_macro_f1':std})
        print(name,results[name],flush=True)
    # Choose on development CV, never on held-out test scores.
    selected=max(configs,key=lambda name:results[name]['cv_macro_f1_mean'])
    test_best=max(configs,key=lambda name:results[name]['macro_f1'])
    results_meta={'selection_metric':'mean macro F1 on 3-fold project-group-disjoint development CV',
      'selected_model':selected,'highest_heldout_macro_f1_model':test_best,'models':results,'data_audit':audit,
      'seed':SEED,'outer_split':'first fold of 5-fold StratifiedGroupKFold(shuffle=True, random_state=42)',
      'label_mapping':MAP,'ordinal_order':'Red (0) < Amber (1) < Green (2)',
      'artifact_fit_scope':'Development rows only; held-out rows not refitted into artifacts',
      'versions':{p:md.version(p) for p in ['scikit-learn','xgboost','mord','numpy','pandas','shap','joblib']},
      'limitations':['All GMPP sectors included, not an IT-only cohort; ICT category counts given in audit.',
       'Contemporaneous DCA classification, not a future-risk forecast or causal analysis.',
       'Project IDs group by original department/number; changed identities or renamed IDs not reconciled.',
       '2025-26 source financial columns explicitly use 2024/25 real prices; earlier source price bases are not harmonised.',
       'Stratified grouped split is not chronological; future-year external validation is still needed.',
       'Same-year rows may be correlated beyond projects; group splitting reduces project leakage, not every dependency.',
       'Minority labels scarce; probabilities are not separately calibrated.',
       '2021-22 Benefits is entirely absent; missingness can encode reporting-year effects.',
       'Cleaning provenance/selection bias not revalidated; 896 are supplied cleaned records, 894 are identifiable.',
       'Ordinal logit uses proportional-odds/all-threshold structure; assumption not formally tested.']}
    (OUT/'metrics.json').write_text(json.dumps(results_meta,indent=2))
    pd.DataFrame(cvrows).to_csv(OUT/'cv_search_results.csv',index=False)
    best=fitted[selected];joblib.dump(best,OUT/'best_model.joblib')
    joblib.dump(best[:-1],OUT/'feature_pipeline.joblib');joblib.dump(best[-1],OUT/'estimator.joblib')
    joblib.dump(fitted[test_best],OUT/'highest_test_macro_f1_model.joblib')
    (OUT/'label_mapping.json').write_text(json.dumps({'class_to_index':MAP,'probability_order':LABELS,'raw_features':RAW_FEATURES,'currency_unit':'GBP millions'},indent=2))
    pred_frame=D.iloc[test][['source_file','source_row','project_group','IPA Delivery Confidence']].copy()
    for name,model in fitted.items():pred_frame[name]=[LABELS[int(i)] for i in model.predict(X.iloc[test])]
    pred_frame.to_csv(OUT/'heldout_predictions.csv',index=False)
    summary=pd.DataFrame({n:{k:s[k] for k in ['macro_f1','balanced_accuracy','cohen_kappa','accuracy','cv_macro_f1_mean']} for n,s in results.items()}).T
    summary.to_csv(OUT/'metrics_table.csv')
    print('\nFINAL METRICS\n',summary.to_string(), '\nCV-selected:',selected,'Test-highest:',test_best,flush=True)
    output_image('02 | Training and model selection',[
      'Development-only 3-fold stratified grouped cross-validation; seed 43.',
      'Random Forest: 6 combinations; XGBoost: 4; Ordinal logistic: 4.',
      'RF uses balanced class weights; XGBoost/ordinal use unweighted training.',
      'Ordinal class order: Red < Amber < Green (all-threshold logistic).',
      'Selection: mean validation macro F1, not test accuracy.',
      *[f"{n:18s} CV macro F1 = {results[n]['cv_macro_f1_mean']:.4f}" for n in results],
      f'Frozen selected model: {selected}',
      'The final estimator is refit on development data only.'
    ],SHOTS/'02_training_and_selection.png')
    output_image('03 | Actual held-out results',[
      'MODEL                 MACRO F1  BAL.ACC  KAPPA    ACCURACY',
      *[f"{n:21s} {s['macro_f1']:.4f}    {s['balanced_accuracy']:.4f}   {s['cohen_kappa']:.4f}   {s['accuracy']:.4f}" for n,s in results.items()],
      f'CV-selected deployment candidate: {selected}',
      f'Highest held-out macro F1 among learned models: {test_best}',
      'Test rows are unseen projects, but not a future-only year cohort.',
      'These are rendered program outputs, not browser/desktop photographs.'
    ],SHOTS/'03_heldout_metrics.png')
    # Figures, with full cohort and exclusion clearly distinguished.
    plt.figure(figsize=(8,5)); values=[audit['all_input_class_counts'][c] for c in LABELS]
    bars=plt.bar(LABELS,values,color=COLORS[:3]);plt.bar_label(bars,labels=[f'{v}\n({v/initial:.1%})' for v in values],padding=6,fontsize=13)
    plt.ylim(0,max(values)*1.22);plt.ylabel('Annual project records');plt.title(f'DCA distribution | supplied cohort (n={initial})');savefig(FIG/'dca_distribution.png')
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    for ax,(name,s) in zip(axes.flat,results.items()):
        cm=np.array(s['confusion_matrix']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(cm.max(),1))
        ax.set(xticks=[0,1,2],xticklabels=LABELS,yticks=[0,1,2],yticklabels=LABELS,xlabel='Predicted',ylabel='Actual',title=name)
        for i in range(3):
            for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=19,color='white' if cm[i,j]>cm.max()/2 else '#11203A',weight='bold')
    fig.suptitle(f'Project-disjoint held-out confusion matrices (n={len(test)})',fontsize=19)
    savefig(FIG/'confusion_matrices.png')
    fig,ax=plt.subplots(figsize=(13,6)); positions=np.arange(len(results));width=.19
    for i,key in enumerate(['macro_f1','balanced_accuracy','cohen_kappa','accuracy']):
        bars=ax.bar(positions+(i-1.5)*width,[s[key] for s in results.values()],width,label=key.replace('_',' ').title(),color=METRIC_COLORS[i]);ax.bar_label(bars,fmt='%.2f',padding=2,fontsize=10)
    ax.set_xticks(positions,list(results));ax.set_ylim(min(0,min(s['cohen_kappa'] for s in results.values())-.05),1);ax.set_ylabel('Score');ax.set_title('Held-out evaluation | accuracy alone hides class imbalance');ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.09));savefig(FIG/'metric_comparison.png')
    # SHAP explanations concern best tree selected by development CV, not necessarily overall winner.
    tree_name=max(['Random Forest','XGBoost'],key=lambda name:results[name]['cv_macro_f1_mean'])
    shap_status={}
    try:
        import shap
        tree=fitted[tree_name]; Xt=tree[:-1].transform(X.iloc[test]); names=tree.named_steps['preprocess'].get_feature_names_out()
        explainer=shap.TreeExplainer(tree[-1]); values=np.asarray(explainer.shap_values(Xt))
        # Modern SHAP returns (rows, features, classes).
        if values.ndim==3:
            if values.shape[:2]==Xt.shape: importance=np.abs(values).mean(axis=(0,2))
            else: importance=np.abs(values).mean(axis=(0,1))
        else: importance=np.abs(values).mean(axis=0)
        order=np.argsort(importance)[-15:];fig,ax=plt.subplots(figsize=(11,8))
        ax.barh(np.array(names)[order],[float(importance[i]) for i in order],color='#7254BD');ax.set_xlabel('Mean absolute SHAP value (all held-out rows and classes)');ax.set_title(f'SHAP global importance | {tree_name}');savefig(FIG/'shap_summary.png')
        pd.DataFrame({'feature':names,'mean_absolute_shap':importance}).sort_values('mean_absolute_shap',ascending=False).to_csv(OUT/'shap_importance.csv',index=False)
        shap_status={'success':True,'tree_model':tree_name,'values_shape':list(values.shape),'path':'figures/shap_summary.png','interpretation':'Global magnitude, not effect direction, causality or probability change.'}
    except Exception as e:
        shap_status={'success':False,'tree_model':tree_name,'reason':repr(e)}
    results_meta['shap']=shap_status
    (OUT/'metrics.json').write_text(json.dumps(results_meta,indent=2))
    # Fixed, de-identified real example for API smoke test; no target included.
    example={k:(None if pd.isna(v) else str(v)) for k,v in X.iloc[test[0]].to_dict().items()}
    (OUT/'example_request.json').write_text(json.dumps(example,indent=2))
    print('SHAP:',shap_status,flush=True)

if __name__=='__main__':main()

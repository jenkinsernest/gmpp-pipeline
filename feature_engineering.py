"""Deterministic, target-free transforms shared by training and inference."""
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

COSTS=['Whole Life Cost','Financial Year Baseline','Financial Year Forecast','Benefits']
RAW_FEATURES=['Department','Annual Report Category',*COSTS,'Start Date','End Date']
CATEGORICAL=['Department','Annual Report Category']
NUMERIC=[*COSTS,'Start year','End year','Planned duration years','Forecast minus baseline','Forecast relative deviation']

def midpoint(value):
    s=str(value).strip()
    m=re.search(r'\bMid\s*:\s*([^,]+(?:,\d{3})*(?:\.\d+)?)',s,re.I)
    # Numeric ranges use commas as thousands separators: delimit by the next named bound.
    if re.search(r'\bMid\s*:',s,re.I):
        s=re.split(r'\bMid\s*:\s*',s,flags=re.I)[1]
        s=re.split(r',?\s*(?:High|Low)\s*:',s,flags=re.I)[0].strip(' ,')
    return s

def money(value):
    s=midpoint(value).replace('£','').replace(',','').replace('\u200b','').strip()
    if re.fullmatch(r'\([-+]?\d+(?:\.\d+)?\)',s): s='-'+s[1:-1]
    if not re.fullmatch(r'[-+]?\d+(?:\.\d+)?',s): return np.nan
    return float(s)

def date_value(value):
    s=midpoint(value)
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}(?: 00:00:00)?',s):
        return pd.to_datetime(s,errors='coerce')
    if re.fullmatch(r'\d{1,2}/\d{1,2}/\d{4}',s):
        return pd.to_datetime(s,format='%d/%m/%Y',errors='coerce')
    return pd.NaT

class EngineerFeatures(TransformerMixin,BaseEstimator):
    def fit(self,X,y=None): return self
    def transform(self,X):
        X=pd.DataFrame(X).copy()
        out=pd.DataFrame(index=X.index)
        for col in CATEGORICAL:
            out[col]=X[col].fillna('Unknown').astype(str).str.strip()
        out['Department']=out['Department'].str.upper()
        out['Annual Report Category']=out['Annual Report Category'].replace({'Information and Communications Technology (ICT)':'ICT'})
        for col in COSTS: out[col]=X[col].map(money)
        start=X['Start Date'].map(date_value); end=X['End Date'].map(date_value)
        out['Start year']=start.map(lambda d: d.year+(d.dayofyear-1)/365.25 if pd.notna(d) else np.nan)
        out['End year']=end.map(lambda d: d.year+(d.dayofyear-1)/365.25 if pd.notna(d) else np.nan)
        duration=(end-start).dt.total_seconds()/(86400*365.25)
        out['Planned duration years']=duration.where(duration>=0)
        out['Forecast minus baseline']=out['Financial Year Forecast']-out['Financial Year Baseline']
        baseline=out['Financial Year Baseline'].abs().replace(0,np.nan)
        out['Forecast relative deviation']=out['Forecast minus baseline']/baseline
        return out[NUMERIC+CATEGORICAL].replace([np.inf,-np.inf],np.nan)

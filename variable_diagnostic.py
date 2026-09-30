#!/usr/bin/env python3
"""Generic CSV variable diagnostic and selection engine.

Produces one diagnostic report for a supplied CSV. Supports scalar boolean,
categorical, ordinal (via overrides), discrete/continuous numeric and percentage
columns. List/tuple/matrix cells are expanded into conservative numeric summary
features. The engine combines univariate evidence, mutual information, model
permutation importance, redundancy, pairwise interactions (bounded), synthetic
noise controls and optional stability estimates.

COMMENTED EDITION
-----------------
This file preserves the v1 logic established for the project. Comments and
section headers were added in English to explain what each part does.
No new statistical method, score component, threshold, or decision category
has been intentionally introduced in this commented edition.

CSV loading is located in SECTION 12, inside main():
    df=pd.read_csv(a.csv,sep=a.delimiter or None,engine='python')
"""

# =============================================================================
# SECTION 1 - IMPORTS AND GLOBAL CONSTANTS
# =============================================================================

from __future__ import annotations
import argparse, ast, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Suppress warnings so the generated diagnostic output stays readable.
warnings.filterwarnings('ignore')

# Fixed random seed used by the repeatable stochastic procedures.
RNG_SEED = 42

# Very small positive constant used to protect divisions from zero.
EPS = 1e-12


# =============================================================================
# SECTION 2 - GENERIC STATISTICAL HELPERS
# =============================================================================

def clip01(x):
    """Convert x to float and clamp it to the interval [0, 1]."""
    try: return float(np.clip(float(x), 0.0, 1.0))
    except Exception: return 0.0


def strength_abs(x):
    """Return |x| limited to [0,1]; return 0 when x is missing."""
    return clip01(abs(float(x))) if pd.notna(x) else 0.0


def cramers_v(x, y):
    """Return bias-corrected Cramer's V and the chi-square p-value.

    This is used when both variables are categorical. Cramer's V provides a
    bounded association-strength measure while chi-square supplies the
    corresponding test p-value.
    """
    tab = pd.crosstab(x, y)
    if tab.size == 0 or min(tab.shape) < 2: return 0.0, 1.0
    chi2, p, _, _ = stats.chi2_contingency(tab)
    n = tab.values.sum(); r, k = tab.shape
    phi2 = chi2 / max(n, 1)
    phi2c = max(0, phi2 - ((k-1)*(r-1))/max(n-1,1))
    rc = r - ((r-1)**2)/max(n-1,1); kc = k - ((k-1)**2)/max(n-1,1)
    den = max(min(kc-1, rc-1), EPS)
    return clip01(math.sqrt(phi2c/den)), float(p)


def eta_squared(groups):
    """Calculate eta-squared for numeric observations divided into groups."""
    vals = [np.asarray(g, float) for g in groups if len(g)]
    if len(vals) < 2: return 0.0
    allv = np.concatenate(vals); gm = np.mean(allv)
    ssb = sum(len(g)*(np.mean(g)-gm)**2 for g in vals); sst = sum((allv-gm)**2)
    return clip01(ssb/max(sst,EPS))


def correlation_ratio(categories, values):
    """Measure how much a numeric variable differs across categorical groups."""
    df = pd.DataFrame({'c': categories, 'v': values}).dropna()
    return eta_squared([g.v.values for _, g in df.groupby('c')]) if len(df) else 0.0


# =============================================================================
# SECTION 3 - LIST / TUPLE / VECTOR / MATRIX SUPPORT
# =============================================================================

def parse_structured(v):
    """Try to parse one cell as a numeric list, tuple, vector, or matrix."""
    if isinstance(v, (list, tuple, np.ndarray)): return np.asarray(v, dtype=float)
    if not isinstance(v, str): return None
    s=v.strip()
    if not (s.startswith('[') or s.startswith('(')): return None
    try:
        a=np.asarray(ast.literal_eval(s), dtype=float)
        return a if a.size else None
    except Exception: return None


def structured_features(series, name):
    """Convert a structured column into seven scalar summary features.

    The established v1 summaries are:
        mean, std, min, max, median, range, size.

    The column is considered structured only when at least 80% of cells can be
    parsed. Vector/matrix structure is flattened before the summaries are made.
    """
    parsed=series.map(parse_structured)
    if parsed.notna().mean() < 0.8: return None
    rows=[]
    for a in parsed:
        if a is None: rows.append([np.nan]*7); continue
        z=np.asarray(a,float).ravel(); z=z[np.isfinite(z)]
        if not len(z): rows.append([np.nan]*7); continue
        rows.append([z.mean(), z.std(), z.min(), z.max(), np.median(z), np.ptp(z), len(z)])
    return pd.DataFrame(rows, index=series.index, columns=[f'{name}__{q}' for q in ['mean','std','min','max','median','range','size']])


# =============================================================================
# SECTION 4 - AUTOMATIC VARIABLE-TYPE INFERENCE
# =============================================================================

def infer_kind(s, name=''):
    """Infer the internal type used to choose the analysis methods.

    Possible automatic results in v1 include:
        constant, identifier, binary, vector, matrix, datetime,
        percentage, numeric_discrete, numeric_continuous, categorical.

    Ordinal variables are accepted through the optional override JSON because
    their semantic ordering cannot be inferred reliably from values alone.
    """
    non=s.dropna(); n=len(non); u=non.nunique(dropna=True)
    if n==0 or u<=1: return 'constant'
    low=name.lower()

    # High-cardinality columns whose names resemble identifiers are excluded.
    if any(k in low for k in ['id','uuid','serial','serie']) and u/max(n,1)>.8: return 'identifier'

    # Detect common binary forms.
    if pd.api.types.is_bool_dtype(s) or (u==2 and set(map(str,non.unique())).issubset({'0','1','True','False','true','false','OK','NOK','ok','nok'})): return 'binary'

    # Detect textual list/tuple/matrix values.
    if non.map(lambda z: parse_structured(z) is not None).mean()>.8:
        first=next(a for a in non.map(parse_structured) if a is not None)
        return 'matrix' if np.asarray(first).ndim>1 else 'vector'

    if pd.api.types.is_datetime64_any_dtype(s): return 'datetime'

    # Try numeric interpretation even if pandas initially read the column as text.
    num=pd.to_numeric(non, errors='coerce')
    if num.notna().mean()>.95:
        vals=num.dropna();
        if ('%' in low) or (vals.between(0,1).all() and any(k in low for k in ['pct','percent','porcentaje'])): return 'percentage'
        if u<=max(12, int(math.sqrt(n))) and np.allclose(vals, np.round(vals)): return 'numeric_discrete'
        return 'numeric_continuous'

    # Very high-cardinality remaining text is treated as a likely identifier.
    if u/max(n,1)>.9: return 'identifier'
    return 'categorical'


def is_numeric_kind(k):
    """Return True when the internal type is handled numerically."""
    return k in {'numeric_discrete','numeric_continuous','percentage','datetime'}


def is_categorical_kind(k):
    """Return True when the internal type is handled categorically."""
    return k in {'binary','categorical','ordinal'}


# =============================================================================
# SECTION 5 - BUILD THE WORKING X PREDICTOR FRAME
# =============================================================================

def prepare_frame(df, ycols, overrides):
    """Prepare all predictors X while keeping their source-column identity.

    Targets are excluded from X. Constants and likely identifiers are not used.
    Vectors/matrices are expanded into summary features. Dates are converted to
    numeric Unix-time seconds. Numeric data are coerced to numeric values.

    Returns:
        X      -> working predictor DataFrame
        meta   -> original column -> inferred/overridden type
        source -> working feature -> original source column
    """
    Xparts=[]; meta={}; source={}
    for c in df.columns:
        if c in ycols: continue
        k=overrides.get(c) or infer_kind(df[c],c); meta[c]=k
        if k in {'constant','identifier'}: continue
        if k in {'vector','matrix'}:
            sf=structured_features(df[c],c)
            if sf is not None:
                Xparts.append(sf); source.update({q:c for q in sf.columns})
        else:
            q=df[[c]].copy()
            if k=='datetime': q[c]=pd.to_datetime(q[c],errors='coerce').astype('int64').replace(-9223372036854775808,np.nan)/1e9
            elif is_numeric_kind(k): q[c]=pd.to_numeric(q[c],errors='coerce')
            Xparts.append(q); source[c]=c
    X=pd.concat(Xparts,axis=1) if Xparts else pd.DataFrame(index=df.index)
    return X, meta, source


# =============================================================================
# SECTION 6 - LEVEL 1: INDIVIDUAL X -> Y EVIDENCE
# =============================================================================

def univariate_evidence(x, y, kx, ky):
    """Calculate type-appropriate individual association evidence for X and Y.

    Numeric/numeric:
        Pearson + Spearman.
    Categorical/categorical:
        chi-square through cramers_v() + Cramer's V strength.
    Numeric/categorical:
        eta-squared strength + Kruskal-Wallis p-value when possible.
    Categorical/numeric:
        correlation-ratio / eta-squared strength.

    The strongest applicable association magnitude becomes the normalized
    univariate evidence score for this stage.
    """
    d=pd.DataFrame({'x':x,'y':y}).dropna(); out={'pearson':None,'spearman':None,'cramers_v':None,'eta2':None,'p_min':None}
    if len(d)<5 or d.x.nunique()<2 or d.y.nunique()<2: return out,0.0
    scores=[]; ps=[]
    if is_numeric_kind(kx) and is_numeric_kind(ky):
        try:
            r,p=stats.pearsonr(pd.to_numeric(d.x),pd.to_numeric(d.y)); out['pearson']=float(r); scores.append(abs(r)); ps.append(p)
        except: pass
        try:
            r,p=stats.spearmanr(pd.to_numeric(d.x),pd.to_numeric(d.y)); out['spearman']=float(r); scores.append(abs(r)); ps.append(p)
        except: pass
    elif is_categorical_kind(kx) and is_categorical_kind(ky):
        v,p=cramers_v(d.x.astype(str),d.y.astype(str)); out['cramers_v']=v; scores.append(v); ps.append(p)
    elif is_numeric_kind(kx) and is_categorical_kind(ky):
        groups=[pd.to_numeric(g.x,errors='coerce').dropna().values for _,g in d.groupby('y')]
        e=eta_squared(groups); out['eta2']=e; scores.append(math.sqrt(e))
        try:
            if len(groups)>=2 and all(len(g)>=2 for g in groups): ps.append(stats.kruskal(*groups).pvalue)
        except: pass
    elif is_categorical_kind(kx) and is_numeric_kind(ky):
        e=correlation_ratio(d.x.astype(str),pd.to_numeric(d.y,errors='coerce')); out['eta2']=e; scores.append(math.sqrt(e))
    out['p_min']=float(min(ps)) if ps else None
    return out, clip01(max(scores) if scores else 0)


def encode_for_mi(x,kx):
    """Encode one X variable for Mutual Information estimation."""
    if is_numeric_kind(kx): return pd.to_numeric(x,errors='coerce').fillna(pd.to_numeric(x,errors='coerce').median()).to_numpy().reshape(-1,1), False
    codes=pd.Series(x).fillna('__MISSING__').astype('category').cat.codes.to_numpy().reshape(-1,1)
    return codes, True


def mi_evidence(x,y,kx,ky):
    """Calculate the established normalized Mutual Information evidence."""
    d=pd.DataFrame({'x':x,'y':y}).dropna()
    if len(d)<20 or d.x.nunique()<2 or d.y.nunique()<2: return 0.0
    X,disc=encode_for_mi(d.x,kx)
    try:
        if is_categorical_kind(ky):
            yy=d.y.astype('category').cat.codes
            mi=mutual_info_classif(X,yy,discrete_features=[disc],random_state=RNG_SEED)[0]
            hy=stats.entropy(pd.Series(yy).value_counts(normalize=True)); return clip01(mi/max(hy,EPS))
        yy=pd.to_numeric(d.y,errors='coerce').to_numpy()
        mi=mutual_info_regression(X,yy,discrete_features=[disc],random_state=RNG_SEED)[0]
        return clip01(1-math.exp(-max(mi,0)))
    except: return 0.0


# =============================================================================
# SECTION 7 - LEVEL 2: JOINT MODEL AND PERMUTATION IMPORTANCE
# =============================================================================

def build_model(X, y, kinds, classification):
    """Create the established preprocessing + Extra Trees joint model.

    Numeric columns:
        median imputation.
    Categorical columns:
        most-frequent imputation + OrdinalEncoder.
    Target mode:
        ExtraTreesClassifier or ExtraTreesRegressor.
    """
    num=[c for c in X if is_numeric_kind(kinds.get(c,'numeric_continuous'))]
    cat=[c for c in X if c not in num]
    pre=ColumnTransformer([
        ('num',Pipeline([('imp',SimpleImputer(strategy='median'))]),num),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('enc',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]),cat)
    ],remainder='drop')
    est=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',random_state=RNG_SEED,n_jobs=-1) if classification else ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,random_state=RNG_SEED,n_jobs=-1)
    return Pipeline([('pre',pre),('model',est)])


def aggregate_feature_importance(cols, source, values):
    """Aggregate derived-feature importance back to each original CSV column."""
    out={}
    for c,v in zip(cols,values): out[source.get(c,c)]=out.get(source.get(c,c),0.0)+max(0,float(v))
    return out


# =============================================================================
# SECTION 8 - X <-> X REDUNDANCY ANALYSIS
# =============================================================================

def redundancy_scores(X, source, kinds):
    """Find the strongest information overlap between every X and another X.

    Numeric/numeric uses absolute Spearman.
    Categorical/categorical uses Cramer's V.
    Mixed pairs use the correlation-ratio formulation established for v1.
    """
    originals=sorted(set(source.values())); best={o:(0.0,None) for o in originals}
    reps={o:[c for c,s in source.items() if s==o][0] for o in originals}
    for i,a in enumerate(originals):
        for b in originals[i+1:]:
            xa,xb=X[reps[a]],X[reps[b]]; ka,kb=kinds.get(a,'numeric_continuous'),kinds.get(b,'numeric_continuous')
            try:
                if is_numeric_kind(ka) and is_numeric_kind(kb): v=abs(stats.spearmanr(pd.to_numeric(xa,errors='coerce'),pd.to_numeric(xb,errors='coerce'),nan_policy='omit').statistic)
                elif is_categorical_kind(ka) and is_categorical_kind(kb): v=cramers_v(xa.fillna('NA').astype(str),xb.fillna('NA').astype(str))[0]
                elif is_numeric_kind(ka): v=math.sqrt(correlation_ratio(xb.fillna('NA').astype(str),pd.to_numeric(xa,errors='coerce')))
                else: v=math.sqrt(correlation_ratio(xa.fillna('NA').astype(str),pd.to_numeric(xb,errors='coerce')))
                v=clip01(v)
            except: v=0
            if v>best[a][0]: best[a]=(v,b)
            if v>best[b][0]: best[b]=(v,a)
    return best


# =============================================================================
# SECTION 9 - BOUNDED PAIRWISE INTERACTION ANALYSIS
# =============================================================================

def interaction_gain(df, x1, x2, y, classification):
    """Estimate the added predictive value of the simple interaction X1*X2.

    The df parameter is preserved exactly from the established v1 function
    signature, although the current function body does not use it directly.
    """
    if not (pd.api.types.is_numeric_dtype(x1) and pd.api.types.is_numeric_dtype(x2)): return 0.0
    d=pd.DataFrame({'a':x1,'b':x2,'y':y}).dropna()
    if len(d)<50: return 0.0
    a=d[['a','b']].to_numpy(); yi=d.y
    try:
        if classification:
            yi=yi.astype('category').cat.codes; base=ExtraTreesClassifier(n_estimators=120,min_samples_leaf=3,random_state=7)
            score=lambda yy,pp: balanced_accuracy_score(yy,pp)
        else:
            yi=pd.to_numeric(yi); base=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=3,random_state=7); score=lambda yy,pp:r2_score(yy,pp)
        tr,te=train_test_split(np.arange(len(d)),test_size=.3,random_state=9,stratify=yi if classification and pd.Series(yi).value_counts().min()>2 else None)
        m1=clone(base).fit(a[tr],np.asarray(yi)[tr]); s1=score(np.asarray(yi)[te],m1.predict(a[te]))
        z=np.column_stack([a,a[:,0]*a[:,1]])
        m2=clone(base).fit(z[tr],np.asarray(yi)[tr]); s2=score(np.asarray(yi)[te],m2.predict(z[te]))
        return clip01(max(0,s2-s1))
    except: return 0.0


# =============================================================================
# SECTION 10 - CENTRAL ANALYSIS, SCORING, NOISE, STABILITY, AND DECISIONS
# =============================================================================

def analyze(df, ycols, overrides, noise_reps=8, stability_reps=5, interaction_top=8):
    """Run the complete established v1 diagnostic for every selected target Y."""

    # Prepare the common X frame once for the requested target set.
    X,meta,source=prepare_frame(df,ycols,overrides)
    results=[]

    # Analyze every requested Y independently.
    for yname in ycols:
        ky=overrides.get(yname) or infer_kind(df[yname],yname); y=df[yname]; classification=is_categorical_kind(ky) or (ky=='numeric_discrete' and y.nunique()<=12)

        # If Y itself is a vector/matrix, summarize it and analyze each summary.
        if ky in {'vector','matrix'}:
            yf=structured_features(y,yname)
            if yf is None: continue
            for yc in yf.columns:
                tmp=df.copy(); tmp[yc]=yf[yc]; results.extend(analyze(tmp,[yc],{**overrides,yc:'numeric_continuous'},noise_reps,stability_reps,interaction_top))
            continue

        # ---------------------------------------------------------------------
        # LEVEL 1: individual X -> Y evidence.
        # ---------------------------------------------------------------------
        rows=[]
        for orig,kx in meta.items():
            if orig not in source.values(): continue
            cols=[c for c,s in source.items() if s==orig]
            evs=[]; mis=[]
            for c in cols:
                ev,u=univariate_evidence(X[c],y,'numeric_continuous' if c!=orig else kx,ky); evs.append((ev,u)); mis.append(mi_evidence(X[c],y,'numeric_continuous' if c!=orig else kx,ky))
            bestev,maxuni=max(evs,key=lambda z:z[1]); mi=max(mis) if mis else 0
            rows.append({'variable':orig,'target':yname,'type_x':kx,'type_y':ky,'univariate':maxuni,'mutual_info':mi,**bestev})
        if not rows: continue
        rmap={r['variable']:r for r in rows}

        # ---------------------------------------------------------------------
        # LEVEL 2: multivariable model + permutation importance.
        # ---------------------------------------------------------------------
        valid=y.notna(); XX=X.loc[valid].copy(); yy=y.loc[valid].copy()
        kinds_exp={c:('numeric_continuous' if c!=source.get(c,c) else meta.get(source.get(c,c),'numeric_continuous')) for c in XX.columns}
        perm={v:0 for v in rmap}; stability={v:0 for v in rmap}; noise_baseline=0
        if len(XX)>=60 and yy.nunique()>=2:
            if classification: yy=yy.astype(str)
            else: yy=pd.to_numeric(yy,errors='coerce'); keep=yy.notna(); XX=XX.loc[keep]; yy=yy.loc[keep]
            strat=yy if classification and yy.value_counts().min()>=3 else None
            tr,te=train_test_split(np.arange(len(XX)),test_size=.3,random_state=RNG_SEED,stratify=strat)
            model=build_model(XX,yy,kinds_exp,classification); model.fit(XX.iloc[tr],yy.iloc[tr])
            scoring='balanced_accuracy' if classification else 'r2'
            pi=permutation_importance(model,XX.iloc[te],yy.iloc[te],n_repeats=6,random_state=RNG_SEED,scoring=scoring,n_jobs=-1)
            raw=aggregate_feature_importance(list(XX.columns),source,pi.importances_mean)
            scale=max(max(raw.values(),default=0),EPS); perm={k:clip01(v/scale) for k,v in raw.items()}

            # -----------------------------------------------------------------
            # Synthetic noise baseline: shuffle real columns in the holdout and
            # compare the resulting score loss against the fitted model.
            # -----------------------------------------------------------------
            noise_vals=[]; rng=np.random.default_rng(RNG_SEED)
            base_score=model.score(XX.iloc[te],yy.iloc[te])
            for _ in range(noise_reps):
                col=rng.choice(list(XX.columns)); altered=XX.iloc[te].copy(); altered[col]=rng.permutation(altered[col].values)
                noise_vals.append(max(0,base_score-model.score(altered,yy.iloc[te])))
            noise_baseline=float(np.quantile(noise_vals,.95)) if noise_vals else 0

            # -----------------------------------------------------------------
            # Stability annex: repeated random holdouts and selection frequency
            # above the median importance for each repeated model.
            # -----------------------------------------------------------------
            counts={v:0 for v in rmap}
            for j in range(stability_reps):
                tr2,te2=train_test_split(np.arange(len(XX)),test_size=.3,random_state=100+j,stratify=strat)
                m=build_model(XX,yy,kinds_exp,classification); m.fit(XX.iloc[tr2],yy.iloc[tr2])
                p=permutation_importance(m,XX.iloc[te2],yy.iloc[te2],n_repeats=3,random_state=200+j,scoring=scoring,n_jobs=-1)
                rr=aggregate_feature_importance(list(XX.columns),source,p.importances_mean); med=np.median(list(rr.values())) if rr else 0
                for v,z in rr.items(): counts[v]+=int(z>max(0,med))
            stability={v:counts[v]/max(stability_reps,1) for v in counts}

        # X <-> X redundancy is a separate evidence channel.
        red=redundancy_scores(X,source,meta)

        # ---------------------------------------------------------------------
        # Interaction search is intentionally bounded to the strongest
        # preliminary candidates and numeric original variables.
        # ---------------------------------------------------------------------
        candidates=sorted(rmap,key=lambda v:max(rmap[v]['univariate'],rmap[v]['mutual_info'],perm.get(v,0)),reverse=True)[:interaction_top]
        inter={v:(0.0,None) for v in rmap}; reps={o:[c for c,s in source.items() if s==o][0] for o in rmap}
        for i,a in enumerate(candidates):
            for b in candidates[i+1:]:
                if is_numeric_kind(meta.get(a,'')) and is_numeric_kind(meta.get(b,'')):
                    g=interaction_gain(df,X[reps[a]],X[reps[b]],y,classification)
                    if g>inter[a][0]: inter[a]=(g,b)
                    if g>inter[b][0]: inter[b]=(g,a)

        # ---------------------------------------------------------------------
        # SCORE CALCULATION + ORDERED DECISION TABLE.
        # These weights and thresholds are the established v1 specification.
        # ---------------------------------------------------------------------
        for v,r in rmap.items():
            # Relevance combines individual, MI, joint-model, and interaction evidence.
            p=perm.get(v,0); relevance=clip01(.30*r['univariate']+.30*r['mutual_info']+.30*p+.10*inter[v][0])

            # Quality penalizes missing X values and small complete X/Y samples.
            missing=float(df[v].isna().mean()); nobs=int(df[[v,yname]].dropna().shape[0]); quality=clip01((1-missing)*min(1,nobs/100))

            # Confidence combines cross-method agreement, data quality, and stability.
            evid=[r['univariate'],r['mutual_info'],p]; agreement=clip01(1-np.std(evid)/.5); model_available=(len(XX)>=60 and yy.nunique()>=2); stab=stability.get(v,0) if model_available else 0.5; confidence=clip01(.45*agreement+.30*quality+.25*stab)

            # Compare joint-model contribution with the established noise reference.
            rv,partner=red.get(v,(0,None)); noise_ok=(p>max(.05, noise_baseline)) if perm else False
            flags=[]

            # DECISION RULE 1 - Quality veto.
            if quality<.5: decision='REVISAR'; flags.append('calidad_datos_baja')

            # DECISION RULE 2 - Relevant evidence that is unstable across splits.
            elif model_available and stability.get(v,1)<.4 and p>=.15 and relevance>=.45: decision='INESTABLE'; flags.append('evidencia_inestable')

            # DECISION RULE 3 - Relevant variable with strong X-X information overlap.
            elif rv>=.90 and relevance>=.45 and ((p<.45) or partner): decision='REDUNDANTE'; flags.append(f'solapamiento_con:{partner}')

            # DECISION RULE 4 - Sufficiently strong combined evidence.
            elif relevance>=.55 and (noise_ok or p>=.25 or (r['univariate']>=.70 and r['mutual_info']>=.50)): decision='USAR'

            # DECISION RULE 5 - Consistently weak evidence with adequate confidence.
            elif r['univariate']<.20 and r['mutual_info']<.20 and p<.15 and not noise_ok and confidence>=.55: decision='POSIBLE RUIDO'

            # DECISION RULE 6 - Conservative fallback for unresolved evidence.
            else: decision='REVISAR'

            # Secondary flags preserve interaction / redundancy context.
            if inter[v][0]>=.08: flags.append(f'interaccion_posible:{inter[v][1]}')
            if rv>=.85: flags.append(f'redundancia_alta:{partner}')

            # Preserve raw evidence, sub-scores, final decision, and explanation.
            r.update({'permutation':p,'interaction':inter[v][0],'interaction_with':inter[v][1],'redundancy':rv,'redundant_with':partner,'stability':stability.get(v,0),'noise_baseline':noise_baseline,'beats_noise':bool(noise_ok),'relevance':relevance,'confidence':confidence,'quality':quality,'decision':decision,'flags':'|'.join(flags)})
            results.append(r)
    return pd.DataFrame(results)


# =============================================================================
# SECTION 11 - DATASET-SPECIFIC HTML REPORT
# =============================================================================

def generate_html(res, csvname, outfile):
    """Generate the established HTML report for the CSV that was analyzed."""
    if res.empty: body='<p>No se pudieron producir diagnósticos.</p>'
    else:
        show=res[['target','variable','type_x','relevance','confidence','quality','univariate','mutual_info','permutation','redundancy','interaction','stability','beats_noise','decision','flags']].copy()
        for c in ['relevance','confidence','quality','univariate','mutual_info','permutation','redundancy','interaction','stability']: show[c]=show[c].map(lambda x:f'{x:.3f}')
        body=show.to_html(index=False,escape=True)
    html=f'''<!doctype html><meta charset="utf-8"><title>Diagnóstico de variables</title><style>body{{font-family:Arial,sans-serif;margin:32px;color:#172033}}h1{{color:#17365d}}table{{border-collapse:collapse;font-size:12px;width:100%}}th{{background:#17365d;color:white;padding:7px}}td{{padding:6px;border-bottom:1px solid #ddd}}tr:nth-child(even){{background:#f5f7fa}}.note{{background:#eef5ff;padding:12px;border-left:4px solid #3b82f6}}</style><h1>Diagnóstico automático de variables</h1><p><b>Archivo:</b> {csvname}</p><div class="note">USAR, REDUNDANTE, INESTABLE, REVISAR y POSIBLE RUIDO son diagnósticos de evidencia predictiva/asociativa; no demuestran causalidad.</div>{body}'''
    Path(outfile).write_text(html,encoding='utf-8')


# =============================================================================
# SECTION 12 - COMMAND LINE, CSV LOADING, EXECUTION, AND OUTPUT FILES
# =============================================================================

def main():
    """Read CLI arguments, load the CSV, run analyze(), and save the reports."""

    # -------------------------------------------------------------------------
    # Command-line arguments.
    # The positional 'csv' argument is the path to the input dataset.
    # Repeat -y / --target to analyze more than one Y.
    # -------------------------------------------------------------------------
    ap=argparse.ArgumentParser(description='Diagnóstico y selección automática de variables en CSV')
    ap.add_argument('csv'); ap.add_argument('--target','-y',action='append',required=True,help='Columna objetivo; repetir para varias Y')
    ap.add_argument('--delimiter',default=None); ap.add_argument('--types',help='JSON opcional {"columna":"ordinal",...}')
    ap.add_argument('--output',default='diagnostico_variables.csv'); ap.add_argument('--html',default='diagnostico_variables.html')
    ap.add_argument('--noise-reps',type=int,default=8); ap.add_argument('--stability-reps',type=int,default=5); ap.add_argument('--interaction-top',type=int,default=8)

    # Parse the values supplied by the user.
    a=ap.parse_args(); overrides=json.loads(Path(a.types).read_text(encoding='utf-8')) if a.types else {}

    # =========================================================================
    # THE CSV IS LOADED HERE.
    # =========================================================================
    # Example:
    #     python variable_diagnostic_v1_commented_en.py process.csv -y Status
    #
    # Then:
    #     a.csv == 'process.csv'
    #
    # pandas reads that file and stores the complete table in df.
    df=pd.read_csv(a.csv,sep=a.delimiter or None,engine='python')

    # Confirm that all requested target columns actually exist in the loaded CSV.
    missing=[y for y in a.target if y not in df.columns]
    if missing: raise SystemExit(f'Objetivos inexistentes: {missing}')

    # Run the complete diagnostic engine on the loaded DataFrame.
    res=analyze(df,a.target,overrides,a.noise_reps,a.stability_reps,a.interaction_top)

    # Save the detailed result table and the readable HTML report.
    res.to_csv(a.output,index=False); generate_html(res,Path(a.csv).name,a.html)
    print(f'OK: {a.output}\nOK: {a.html}')


# =============================================================================
# SECTION 13 - PROGRAM ENTRY POINT
# =============================================================================

if __name__=='__main__': main()

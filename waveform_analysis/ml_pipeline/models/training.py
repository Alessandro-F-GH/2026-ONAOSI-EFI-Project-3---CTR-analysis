from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle
import numpy as np

@dataclass
class FittedModel:
    model_type:str; payload:Any; mean:np.ndarray|float; std:np.ndarray|float; params:dict[str,Any]; device:str='cpu'
    def predict(self,X:np.ndarray,batch_size:int=2048)->np.ndarray:
        X=np.asarray(X,dtype=np.float32)
        if self.model_type=='linear_svr':
            diff=(X[:,0,:]-X[:,1,:]-self.mean)/self.std
            return np.asarray(self.payload.predict(diff),dtype=np.float64)
        import torch
        model=self.payload; dev=torch.device(self.device); model.eval(); out=[]
        with torch.no_grad():
            for start in range(0,len(X),batch_size):
                z=(X[start:start+batch_size]-self.mean)/self.std
                out.append(model(torch.from_numpy(np.asarray(z,np.float32)).to(dev)).cpu().numpy())
        return np.concatenate(out).astype(np.float64) if out else np.empty(0)
    def save(self,path:Path):
        path.parent.mkdir(parents=True,exist_ok=True)
        if self.model_type=='linear_svr':
            with path.open('wb') as f: pickle.dump({'model_type':self.model_type,'payload':self.payload,'mean':self.mean,'std':self.std,'params':self.params},f)
        else:
            import torch
            torch.save({'model_type':self.model_type,'state_dict':self.payload.state_dict(),'input_length':self.payload.input_length,'mean':np.asarray(self.mean),'std':np.asarray(self.std),'params':self.params},path)

def _activation(name:str):
    import torch.nn as nn
    key=str(name).lower()
    if key=='silu':return nn.SiLU()
    if key=='tanh':return nn.Tanh()
    if key=='gelu':return nn.GELU()
    return nn.ReLU()

class SharedMLP(__import__('torch').nn.Module):
    def __init__(self,input_length,hidden_size=64,depth=2,activation='relu'):
        import torch.nn as nn
        super().__init__(); self.input_length=int(input_length)
        layers=[]; d=self.input_length
        for _ in range(max(1,int(depth))): layers += [nn.Linear(d,int(hidden_size)),_activation(activation)]; d=int(hidden_size)
        layers.append(nn.Linear(d,1)); self.branch=nn.Sequential(*layers)
    def forward(self,x): return self.branch(x[:,0,:]).squeeze(-1)-self.branch(x[:,1,:]).squeeze(-1)

class SharedCNN(__import__('torch').nn.Module):
    def __init__(self,input_length,params):
        import torch.nn as nn
        super().__init__(); self.input_length=int(input_length)
        channels=params.get('channels',[16,32,64]); channels=[int(channels)] if np.isscalar(channels) else [int(x) for x in channels]
        kernels=params.get('kernel_sizes',params.get('kernel_size',[9]*len(channels))); kernels=[int(kernels)]*len(channels) if np.isscalar(kernels) else [int(x) for x in kernels]
        strides=params.get('strides',[1]*len(channels)); strides=[int(strides)]*len(channels) if np.isscalar(strides) else [int(x) for x in strides]
        dilations=params.get('dilations',[1]*len(channels)); dilations=[int(dilations)]*len(channels) if np.isscalar(dilations) else [int(x) for x in dilations]
        while len(kernels)<len(channels):kernels.append(kernels[-1])
        while len(strides)<len(channels):strides.append(strides[-1])
        while len(dilations)<len(channels):dilations.append(dilations[-1])
        layers=[]; cin=1; act=str(params.get('activation','relu')); dropout=float(params.get('conv_dropout',0.0)); normalization=str(params.get('normalization','none')).lower()
        for cout,k,s,d in zip(channels,kernels,strides,dilations):
            padding=((k-1)*d)//2; layers.append(nn.Conv1d(cin,cout,k,stride=s,padding=padding,dilation=d))
            if normalization in {'batch','batchnorm','batch_norm'}: layers.append(nn.BatchNorm1d(cout))
            elif normalization in {'group','groupnorm','group_norm'}: layers.append(nn.GroupNorm(1,cout))
            layers.append(_activation(act))
            if dropout>0: layers.append(nn.Dropout(dropout))
            cin=cout
        pool=max(1,int(params.get('adaptive_pool_length',8))); layers += [nn.AdaptiveAvgPool1d(pool),nn.Flatten()]; self.features=nn.Sequential(*layers)
        dense=params.get('dense_units',[32]); dense=[int(dense)] if np.isscalar(dense) else [int(x) for x in dense]; head=[]; dim=channels[-1]*pool
        for width in dense: head += [nn.Linear(dim,width),_activation(act)]; dim=width
        head.append(nn.Linear(dim,1)); self.head=nn.Sequential(*head)
    def score(self,x): return self.head(self.features(x[:,None,:])).squeeze(-1)
    def forward(self,x): return self.score(x[:,0,:])-self.score(x[:,1,:])

def build_torch_pair_model(model_type,input_length,params):
    if model_type=='constructive_mlp_encoder':
        hidden=int(params.get('hidden_size',params.get('max_units',64))); depth=int(params.get('depth',1 if 'max_units' in params else 2)); return SharedMLP(input_length,hidden,depth,params.get('activation','relu'))
    if model_type=='cnn_regressor': return SharedCNN(input_length,params)
    raise ValueError(f'{model_type} is not a Torch waveform model')

def _device(name):
    import torch
    if name=='auto': return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name

def fit_model(model_type:str,X:np.ndarray,y:np.ndarray,params:dict[str,Any],*,seed:int,training:dict[str,Any],X_val=None,y_val=None)->FittedModel:
    X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.float64)
    if len(X)!=len(y) or X.ndim!=3 or X.shape[1]!=2: raise ValueError('Expected X [events,2,length] and matching y')
    if model_type=='linear_svr':
        from sklearn.svm import LinearSVR
        diff=np.asarray(X[:,0,:]-X[:,1,:],dtype=np.float64); mean=np.mean(diff,axis=0); std=np.std(diff,axis=0); std=np.where(std>1e-8,std,1.0); z=(diff-mean)/std
        est=LinearSVR(C=float(params.get('C',1.0)),epsilon=float(params.get('epsilon_ps',params.get('epsilon',10.0))),loss=str(params.get('svm_loss','epsilon_insensitive')),tol=float(params.get('tolerance',1e-4)),max_iter=int(params.get('max_iterations',20000)),dual=params.get('dual','auto'),random_state=int(seed))
        est.fit(z,y); return FittedModel(model_type,est,mean,std,dict(params))
    import torch
    torch.manual_seed(seed); np.random.seed(seed); dev=_device(str(training.get('device','auto')))
    flat=X.reshape(-1,X.shape[-1]); mean=np.mean(flat,axis=0).astype(np.float32); std=np.std(flat,axis=0).astype(np.float32); std=np.where(std>1e-6,std,1.0).astype(np.float32)
    if X_val is None or y_val is None:
        rng=np.random.default_rng(seed); order=rng.permutation(len(X)); nv=min(max(1,int(round(len(X)*float(training.get('early_stop_fraction',.15))))),max(1,len(X)-2)); vi=order[:nv]; ti=order[nv:]; Xtr,ytr=X[ti],y[ti]; Xv,yv=X[vi],y[vi]
    else: Xtr,ytr=X,np.asarray(y); Xv,yv=np.asarray(X_val,dtype=np.float32),np.asarray(y_val,dtype=np.float64)
    model=build_torch_pair_model(model_type,X.shape[-1],params).to(dev); opt=torch.optim.Adam(model.parameters(),lr=float(params.get('learning_rate',1e-3)),weight_decay=float(params.get('weight_decay',1e-5)))
    loss_fn=torch.nn.MSELoss(); batch=int(training.get('batch_size',256)); epochs=int(training.get('max_epochs',120)); patience=int(training.get('patience',12)); best=None; best_loss=np.inf; stale=0; rng=np.random.default_rng(seed)
    for _epoch in range(epochs):
        model.train(); order=rng.permutation(len(Xtr))
        for start in range(0,len(order),batch):
            idx=order[start:start+batch]; xb=torch.from_numpy(((Xtr[idx]-mean)/std).astype(np.float32)).to(dev); yb=torch.from_numpy(np.asarray(ytr[idx],np.float32)).to(dev)
            opt.zero_grad(set_to_none=True); pred=model(xb); loss=loss_fn(pred,yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vals=[]
            for start in range(0,len(Xv),batch): vals.append(model(torch.from_numpy(((Xv[start:start+batch]-mean)/std).astype(np.float32)).to(dev)).cpu().numpy())
        pred=np.concatenate(vals) if vals else np.empty(0); vl=float(np.mean((np.asarray(yv)-pred)**2)) if pred.size else np.inf
        if vl<best_loss-1e-6: best_loss=vl; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=patience: break
    if best is not None: model.load_state_dict(best)
    return FittedModel(model_type,model,mean,std,dict(params),dev)

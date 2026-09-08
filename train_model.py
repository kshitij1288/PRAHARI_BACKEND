import pandas as pd, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv("data/wayanad_sensor_prototype.csv")
features=["s1_pct","s2_pct","dht_temp","dht_humidity","bmp_pressure","rain_detected","vibration","magnitude","pitch","roll"]
X=df[features].values
y=df["risk"].values
scaler=StandardScaler()
Xs=scaler.fit_transform(X)
Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
model=MLPClassifier(hidden_layer_sizes=(16,12),max_iter=800,random_state=42)
model.fit(Xtr,ytr)
acc=accuracy_score(yte,model.predict(Xte))
out={
 "prototype_accuracy":float(acc),
 "rows":int(len(df)),
 "features":features,
 "warning":"This is prototype training data, not measured field data."
}
open("model/metrics.json","w").write(json.dumps(out,indent=2))
print(out)

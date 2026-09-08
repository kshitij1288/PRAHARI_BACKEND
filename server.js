const express = require("express");
const fs = require("fs");
const path = require("path");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json({limit:"100kb"}));

const PORT = process.env.PORT || 5000;
const HOST = process.env.HOST || "0.0.0.0";
const DATA_DIR = path.join(__dirname, "data");
const LIVE_FILE = path.join(DATA_DIR, "live_data.csv");
const MODEL = JSON.parse(fs.readFileSync(path.join(__dirname,"model","model.json"),"utf8"));

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR,{recursive:true});
if (!fs.existsSync(LIVE_FILE)) {
  fs.writeFileSync(LIVE_FILE,
    "timestamp,s1_pct,s2_pct,dht_temp,dht_humidity,bmp_pressure,rain_detected,vibration,magnitude,pitch,roll,risk,confidence,risk_code,sample_id\n"
  );
}

function relu(x){ return Math.max(0,x); }
function softmax(arr){
  const m=Math.max(...arr);
  const ex=arr.map(v=>Math.exp(v-m));
  const s=ex.reduce((a,b)=>a+b,0);
  return ex.map(v=>v/s);
}
function dense(x, layer, activation){
  const out=[];
  for(let j=0;j<layer.bias.length;j++){
    let z=layer.bias[j];
    for(let i=0;i<x.length;i++) z += x[i]*layer.weights[i][j];
    out.push(activation==="relu" ? relu(z) : z);
  }
  return out;
}
function predict(features){
  const x=MODEL.features.map((k,i)=>(Number(features[k]) - MODEL.mean[i]) / MODEL.scale[i]);
  let h=dense(x,MODEL.layers[0],"relu");
  h=dense(h,MODEL.layers[1],"relu");
  const logits=dense(h,MODEL.layers[2],"linear");
  const probs=softmax(logits);
  let best=0;
  for(let i=1;i<probs.length;i++) if(probs[i]>probs[best]) best=i;
  const risk=MODEL.classes[best];
  const code={LOW:0,MEDIUM:1,HIGH:2}[risk] ?? 0;
  const probability={};
  MODEL.classes.forEach((c,i)=>probability[c]=Number((probs[i]*100).toFixed(2)));
  return {risk, confidence:Number((probs[best]*100).toFixed(2)), risk_code:code, probabilities:probability};
}

function n(v,d=0){ const x=Number(v); return Number.isFinite(x)?x:d; }

function normalize(body){
  return {
    s1_pct:n(body.s1_pct),
    s2_pct:n(body.s2_pct),
    dht_temp:n(body.dht_temp),
    dht_humidity:n(body.dht_humidity),
    bmp_pressure:n(body.bmp_pressure),
    rain_detected:n(body.rain_detected),
    vibration:n(body.vibration),
    magnitude:n(body.magnitude,1),
    pitch:n(body.pitch),
    roll:n(body.roll)
  };
}
function nextSampleId(){
  try {
    const txt=fs.readFileSync(LIVE_FILE,"utf8").trim().split("\n");
    return Math.max(0,txt.length-1)+1;
  } catch { return 1; }
}
function append(row){
  const values=[
    row.timestamp,row.s1_pct,row.s2_pct,row.dht_temp,row.dht_humidity,row.bmp_pressure,
    row.rain_detected,row.vibration,row.magnitude,row.pitch,row.roll,
    row.risk,row.confidence,row.risk_code,row.sample_id
  ];
  fs.appendFileSync(LIVE_FILE,values.join(",")+"\n");
}

app.get("/api/status",(req,res)=>res.json({
  ok:true,
  service:"PRAHARI Wayanad backend",
  model:"Wayanad-informed prototype",
  model_warning:"Prototype sensor training data; field validation required.",
  time:new Date().toISOString()
}));

app.get("/api/latest",(req,res)=>{
  const txt=fs.readFileSync(LIVE_FILE,"utf8").trim().split("\n");
  if(txt.length<2) return res.json({ok:true,latest:null});
  const cols=txt[0].split(",");
  const vals=txt[txt.length-1].split(",");
  const obj={}; cols.forEach((c,i)=>obj[c]=vals[i]);
  res.json({ok:true,latest:obj});
});

app.get("/api/history",(req,res)=>{
  const limit=Math.min(Math.max(Number(req.query.limit)||100,1),1000);
  const lines=fs.readFileSync(LIVE_FILE,"utf8").trim().split("\n");
  const cols=lines.shift().split(",");
  const data=lines.slice(-limit).map(line=>{
    const vals=line.split(","); const o={};
    cols.forEach((c,i)=>o[c]=vals[i]); return o;
  });
  res.json({ok:true,data});
});

app.get("/api/predict",(req,res)=>{
  const input=normalize(req.query);
  res.json({ok:true,input,prediction:predict(input)});
});

app.post(["/data","/api/data"],(req,res)=>{
  const input=normalize(req.body);
  const p=predict(input);
  const row={...input,...p,timestamp:new Date().toISOString(),sample_id:nextSampleId()};
  append(row);
  res.json(row);
});

app.get("/api/export.csv",(req,res)=>{
  res.download(LIVE_FILE,"prahari_live_data.csv");
});

app.listen(PORT,HOST,()=>console.log(`PRAHARI Wayanad backend listening on ${HOST}:${PORT}`));

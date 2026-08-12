const LABELS={random:"Random",zscore:"Z-Score",pca:"PCA",iforest:"Isolation Forest",
  lstm_autoencoder:"LSTM Autoencoder"};
const ORDER=["random","zscore","pca","iforest","lstm_autoencoder"];
const DARK={paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",
  font:{color:"#94a3b8",family:"Inter"},margin:{l:55,r:20,t:10,b:45}};

let current=null, machine=null, evt=null;

async function getJSON(u){const r=await fetch(u);if(!r.ok)throw new Error(u+" "+r.status);return r.json();}

async function init(){
  const {machines}=await getJSON("/api/machines");
  const sel=document.getElementById("machine-select");
  sel.innerHTML=machines.map(m=>`<option>${m}</option>`).join("");
  sel.onchange=()=>{machine=sel.value;stop();resetChart();};
  machine=machines[0];
  const results=await getJSON(`/api/results/${machine}`);
  const dets=Object.keys(results).sort((a,b)=>(ORDER.indexOf(a)+1||99)-(ORDER.indexOf(b)+1||99));
  const tabs=document.getElementById("tabs");
  tabs.innerHTML=dets.map(d=>`<div class="tab" data-det="${d}">${LABELS[d]||d}</div>`).join("")
    +`<div class="tab total" data-det="__total">All detectors</div>`;
  tabs.querySelectorAll(".tab").forEach(t=>t.onclick=()=>selectTab(t.dataset.det));
  selectTab(dets[2]||dets[0]); // default to PCA-ish
}

function selectTab(det){
  current=det; stop();
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t.dataset.det===det));
  const title=det==="__total"?"All detectors — live":`${LABELS[det]||det} — live detection`;
  document.getElementById("view-title").textContent=title;
  resetChart();
}

function resetChart(){
  Plotly.newPlot("chart",[{y:[],x:[],mode:"lines",line:{color:"#38bdf8",width:1.4},name:"score"},
    {y:[],x:[],mode:"markers",marker:{color:"#ef4444",size:5},name:"flag"}],
    {...DARK,height:420,xaxis:{title:"timestep",gridcolor:"#1e293b"},
     yaxis:{title:"anomaly score",gridcolor:"#1e293b"},shapes:[],showlegend:false},
    {displayModeBar:false});
  document.getElementById("caught-num").textContent="0 / 0";
  document.getElementById("alarm-num").textContent="0";
  document.getElementById("score-num").textContent="–";
  document.getElementById("progress").textContent="";
}

function stop(){
  if(evt){evt.close();evt=null;}
  const b=document.getElementById("play-btn");
  b.textContent="Start stream";b.disabled=false;b.classList.remove("pulse");
}

function startStream(){
  if(current==="__total"){startTotal();return;}
  stop(); resetChart();
  const b=document.getElementById("play-btn");
  b.textContent="Streaming…";b.disabled=true;b.classList.add("pulse");

  const xs=[],ys=[],fx=[],fy=[],shapes=[];
  let threshold=null,nSeg=0,caught=0,alarms=0,segStart=null,inCaught=false;

  evt=new EventSource(`/api/stream/${machine}/${current}`);
  evt.onmessage=e=>{
    const m=JSON.parse(e.data);
    if(m.meta){threshold=m.threshold;nSeg=m.n_segments;
      document.getElementById("caught-num").textContent=`0 / ${nSeg}`;return;}
    if(m.done){stop();document.getElementById("progress").textContent=`done — ${xs.length} steps`;return;}

    // anomaly band tracking
    if(m.label===1&&segStart===null){segStart=m.t;inCaught=false;}
    if(m.label===1&&m.score!==null&&threshold!==null&&m.score>=threshold&&!inCaught){
      inCaught=true;caught++;document.getElementById("caught-num").textContent=`${caught} / ${nSeg}`;
    }
    if(m.label===0&&segStart!==null){
      shapes.push({type:"rect",xref:"x",yref:"paper",x0:segStart,x1:m.t,y0:0,y1:1,
        fillcolor:"#ef4444",opacity:0.14,line:{width:0}});segStart=null;
    }
    // score line + flags
    if(m.score!==null){
      xs.push(m.t);ys.push(m.score);
      document.getElementById("score-num").textContent=m.score.toFixed(3);
      if(threshold!==null&&m.score>=threshold){fx.push(m.t);fy.push(m.score);alarms++;}
    }
    if(m.t%6===0){
      Plotly.update("chart",{x:[xs,fx],y:[ys,fy]},{shapes},[0,1]);
      document.getElementById("alarm-num").textContent=alarms;
      document.getElementById("progress").textContent=`t = ${m.t}`;
    }
  };
  evt.onerror=()=>{stop();document.getElementById("progress").textContent="stream ended";};
}

// "All detectors" — stream each sequentially into overlaid faint lines
async function startTotal(){
  stop();resetChart();
  const results=await getJSON(`/api/results/${machine}`);
  const dets=Object.keys(results).sort((a,b)=>(ORDER.indexOf(a)+1||99)-(ORDER.indexOf(b)+1||99));
  const colors=["#64748b","#38bdf8","#a78bfa","#f59e0b","#22c55e"];
  Plotly.newPlot("chart",dets.map((d,i)=>({y:[],x:[],mode:"lines",
    line:{color:colors[i%colors.length],width:1.2},name:LABELS[d]||d})),
    {...DARK,height:420,xaxis:{title:"timestep",gridcolor:"#1e293b"},
     yaxis:{title:"anomaly score",gridcolor:"#1e293b"},showlegend:true,
     legend:{orientation:"h",y:1.1,font:{size:11}}},{displayModeBar:false});
  const b=document.getElementById("play-btn");b.textContent="Streaming…";b.disabled=true;b.classList.add("pulse");
  document.getElementById("view-desc").textContent="Every detector streamed together — compare how each responds to the same anomalies.";
  for(let i=0;i<dets.length;i++){await streamInto(dets[i],i);}
  stop();
}
function streamInto(det,idx){
  return new Promise(res=>{
    const xs=[],ys=[];const s=new EventSource(`/api/stream/${machine}/${det}`);
    s.onmessage=e=>{const m=JSON.parse(e.data);
      if(m.meta)return; if(m.done){s.close();res();return;}
      if(m.score!==null){xs.push(m.t);ys.push(m.score);}
      if(m.t%8===0)Plotly.update("chart",{x:[xs],y:[ys]},{},[idx]);};
    s.onerror=()=>{s.close();res();};
  });
}

document.getElementById("play-btn").onclick=startStream;
init().catch(e=>console.error(e));
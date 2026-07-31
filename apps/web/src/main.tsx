import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, Brain, Cpu, Download, FileJson, Gauge, ImageUp, Layers, Play, RefreshCw, Settings, TerminalSquare } from "lucide-react";
import "./styles.css";

type Api<T> = { success: boolean; data: T };
type Weight = { checkpoint_id: string; display_name: string; status: string; size_bytes?: number; sha256?: string; domain?: string; default?: boolean; auto_route?: boolean; path?: string };
type ModelStatus = { model_id: string; display_name: string; description: string; available: boolean; status_message: string; license_name: string; repository_url: string; capabilities: { weights?: Weight[]; auto_route?: boolean; manual_only_note?: string } };
type Task = { task_id: string; status: string; progress: number; logs: string[]; analysis?: any; user_intent?: any; hardware_info?: any; model_candidates: any[]; checkpoint_candidates: any[]; plan?: any; candidates: any[]; best_result?: any; error?: string; report_file_id?: string; final_recommendation?: string };

const api = async <T,>(url: string, init?: RequestInit): Promise<T> => {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(await res.text());
  const body = await res.json() as Api<T>;
  return body.data;
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [imageId, setImageId] = useState("");
  const [goal, setGoal] = useState("自然增强暗部，保持原始颜色，路灯不要过曝，质量优先，这是一张室外城市夜景。");
  const [mode, setMode] = useState<"auto" | "manual" | "compare">("auto");
  const [priority, setPriority] = useState<"quality" | "balanced" | "speed">("quality");
  const [model, setModel] = useState("retinexformer");
  const [checkpoint, setCheckpoint] = useState("lol_v2_real");
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [system, setSystem] = useState<any>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [intent, setIntent] = useState<any>(null);
  const [task, setTask] = useState<Task | null>(null);
  const [history, setHistory] = useState<Task[]>([]);
  const [tab, setTab] = useState("candidates");
  const [busy, setBusy] = useState(false);

  const refresh = () => {
    api<ModelStatus[]>("/api/v1/models").then(setModels).catch(console.error);
    api<any>("/api/v1/system").then(setSystem).catch(console.error);
    api<Task[]>("/api/v1/history").then(setHistory).catch(console.error);
  };
  useEffect(refresh, []);
  useEffect(() => {
    if (!task || ["completed", "failed", "cancelled"].includes(task.status)) return;
    const id = window.setInterval(async () => {
      const next = await api<Task>(`/api/v1/tasks/${task.task_id}`);
      setTask(next);
      setAnalysis(next.analysis || analysis);
      setIntent(next.user_intent || intent);
    }, 1000);
    return () => window.clearInterval(id);
  }, [task]);

  const selectedModel = models.find(m => m.model_id === model);
  const weights = selectedModel?.capabilities?.weights || [];
  useEffect(() => { if (weights.length && !weights.some(w => w.checkpoint_id === checkpoint)) setCheckpoint(weights[0].checkpoint_id); }, [model, models]);

  const onFile = (f: File | null) => {
    setFile(f); setAnalysis(null); setImageId(""); setTask(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(f ? URL.createObjectURL(f) : "");
  };
  const parseIntent = async () => setIntent(await api<any>("/api/v1/intent/parse", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: goal, priority, mode, model_id: mode === "manual" ? model : undefined, checkpoint_id: mode === "manual" ? checkpoint : undefined }) }));
  const uploadAnalyze = async () => {
    if (!file) return;
    setBusy(true);
    const fd = new FormData(); fd.append("file", file);
    const data = await api<any>("/api/v1/images/analyze", { method: "POST", body: fd });
    setImageId(data.file.file_id); setAnalysis(data.analysis); await parseIntent(); setBusy(false);
  };
  const start = async () => {
    if (!file) return;
    let id = imageId;
    if (!id) {
      const fd = new FormData(); fd.append("file", file);
      const uploaded = await api<any>("/api/v1/images/upload", { method: "POST", body: fd });
      id = uploaded.file_id; setImageId(id);
    }
    const created = await api<Task>("/api/v1/tasks", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_id: id, user_goal: goal, mode, priority, model_id: mode === "manual" ? model : undefined, checkpoint_id: mode === "manual" ? checkpoint : undefined }) });
    setTask(created); setTab("logs");
  };
  const installedWeights = models.flatMap(m => m.capabilities?.weights || []).filter(w => w.status === "found").length;
  const resultUrl = task?.best_result?.output_url;

  return <div className="shell">
    <header className="topbar"><div className="brand"><Brain/><b>VisionRestore Agent</b><span>本地模式</span></div><div className="sys"><span>API: online</span><span>{system?.hardware?.gpu_name || "GPU unknown"}</span><span>CUDA: {String(system?.hardware?.cuda_available ?? false)}</span><span>显存: {system?.hardware?.gpu_memory_mb ?? "-"} MB</span><span>模型: {models.filter(m=>m.available).length}</span><span>权重: {installedWeights}</span></div></header>
    <aside className="nav"><a href="#workbench"><ImageUp/>增强工作台</a><a href="#models"><Layers/>模型与权重</a><a href="#history"><Activity/>历史任务</a><a href="#settings"><Settings/>系统设置</a></aside>
    <main id="workbench" className="mainPanel">
      <section className="inputPane glass"><h2>输入图像与需求</h2><div className="drop" onDrop={e=>{e.preventDefault(); onFile(e.dataTransfer.files[0]);}} onDragOver={e=>e.preventDefault()}><input id="file" type="file" accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff" onChange={e=>onFile(e.target.files?.[0] || null)}/><label htmlFor="file"><ImageUp/> {file?.name || "拖拽或选择 RGB 静态图像"}</label>{preview && <img src={preview}/>}</div><textarea value={goal} onChange={e=>setGoal(e.target.value)} /><div className="seg">{(["quality","balanced","speed"] as const).map(x=><button key={x} className={priority===x?"on":""} onClick={()=>setPriority(x)}>{x}</button>)}</div><div className="seg">{(["auto","manual","compare"] as const).map(x=><button key={x} className={mode===x?"on":""} onClick={()=>setMode(x)}>{x}</button>)}</div>{mode==="manual" && <div className="manual"><select value={model} onChange={e=>setModel(e.target.value)}>{models.map(m=><option key={m.model_id} value={m.model_id}>{m.display_name}</option>)}</select><select value={checkpoint} onChange={e=>setCheckpoint(e.target.value)}>{weights.map(w=><option key={w.checkpoint_id} value={w.checkpoint_id}>{w.display_name}</option>)}</select></div>}<div className="row"><button onClick={uploadAnalyze} disabled={!file || busy}><Gauge/>分析/解析</button><button className="primary" onClick={start} disabled={!file}><Play/>开始智能增强</button>{task && !["completed","failed","cancelled"].includes(task.status) && <button onClick={()=>api(`/api/v1/tasks/${task.task_id}/cancel`, {method:"POST"}).then(setTask as any)}>取消任务</button>}</div></section>
      <section className="comparePane glass"><h2>增强结果对比</h2><div className="images"><div>{preview ? <img src={preview}/> : <span>原始图像</span>}</div><div>{resultUrl ? <img src={resultUrl}/> : <span>{task?.error || "推荐增强结果"}</span>}</div></div><div className="facts"><span>输入: {analysis ? `${analysis.width}×${analysis.height}` : "-"}</span><span>输出: {task?.best_result?.metrics ? `${analysis?.width}×${analysis?.height}` : "-"}</span><span>推荐模型: {task?.best_result?.model_id || "-"}</span><span>推荐权重: {task?.best_result?.checkpoint_id || "-"}</span><span>评分: {task?.best_result?.score ?? "-"}</span><span>耗时: {task?.best_result?.runtime_ms ?? "-"} ms</span><span>显存: {task?.best_result?.peak_memory_mb?.toFixed?.(1) ?? "-"} MB</span></div><div className="row">{resultUrl && <a className="button" href={resultUrl} download><Download/>保存图像</a>}{task?.task_id && <a className="button" href={`/api/v1/tasks/${task.task_id}/report`}><FileJson/>导出报告</a>}</div></section>
      <section className="bottom glass"><div className="tabs">{["candidates","metrics","logs","params"].map(x=><button key={x} className={tab===x?"on":""} onClick={()=>setTab(x)}>{x}</button>)}</div>{tab==="candidates" && <pre>{JSON.stringify(task?.candidates || [], null, 2)}</pre>}{tab==="metrics" && <pre>{JSON.stringify(task?.best_result?.metrics || { note: "无参考指标只能作为辅助判断，不能完全替代人工主观评价。" }, null, 2)}</pre>}{tab==="logs" && <div className="log">{(task?.logs || []).map((l,i)=><p key={i}>{l}</p>)}</div>}{tab==="params" && <pre>{JSON.stringify(task?.plan || {}, null, 2)}</pre>}</section>
    </main>
    <aside className="decision glass"><h2>Agent 决策中心</h2><Block title="图像分析" data={analysis}/><Block title="用户意图" data={intent || task?.user_intent}/><Block title="模型架构路由" data={task?.model_candidates}/><Block title="权重路由" data={task?.checkpoint_candidates}/><Block title="执行计划" data={task?.plan}/><Block title="当前状态" data={{status:task?.status, progress:task?.progress, recommendation:task?.final_recommendation}}/></aside>
    <section id="models" className="modelsPage glass"><h2>模型与权重</h2><button onClick={refresh}><RefreshCw/>重新扫描</button>{models.map(m=><div className="modelCard" key={m.model_id}><h3>{m.display_name}</h3><p>{m.description}</p><code>{m.status_message}</code><small>{m.license_name}</small>{(m.capabilities.weights||[]).map(w=><div className="weight" key={w.checkpoint_id}><b>{w.display_name}</b><span>{w.status}</span><span>{fmtBytes(w.size_bytes)}</span><span>{w.domain}</span><small>{w.sha256?.slice(0,16)}...</small></div>)}</div>)}</section>
    <section id="history" className="historyPage glass"><h2>历史任务</h2>{history.slice(0,20).map(h=><p key={h.task_id}>{h.task_id.slice(0,8)} · {h.status} · {h.best_result?.model_id || "-"}</p>)}</section>
    <section id="settings" className="settingsPage glass"><h2>系统设置</h2><pre>{JSON.stringify(system, null, 2)}</pre></section>
  </div>;
}

function Block({title,data}:{title:string;data:any}) { return <div className="block"><h3>{title}</h3><pre>{JSON.stringify(data || {}, null, 2)}</pre></div> }
function fmtBytes(n?: number) { if (!n) return "-"; return `${(n/1024/1024).toFixed(2)} MB`; }

createRoot(document.getElementById("root")!).render(<App />);

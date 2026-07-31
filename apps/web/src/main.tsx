import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, Brain, Download, FileJson, ImageUp, Play, RotateCcw, Settings, SlidersHorizontal } from "lucide-react";
import "./styles.css";

type Api<T> = { success: boolean; data: T };
type ModelStatus = { model_id: string; display_name: string; description: string; available: boolean; installed: boolean; status_message: string; license_name: string; repository_url: string; supported_devices: string[]; supported_precisions: string[]; install_hint: string };
type Task = { task_id: string; status: string; progress: number; logs: string[]; analysis?: any; plan?: any; candidates: any[]; best_result?: any; error?: string; report_file_id?: string };

const api = async <T,>(url: string, init?: RequestInit): Promise<T> => {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(await res.text());
  const body = await res.json() as Api<T>;
  return body.data;
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [imageId, setImageId] = useState("");
  const [goal, setGoal] = useState("自然增强这张夜景图，不要让高光过曝。");
  const [mode, setMode] = useState<"auto" | "manual" | "compare">("auto");
  const [priority, setPriority] = useState<"quality" | "balanced" | "speed">("balanced");
  const [model, setModel] = useState("zero_dce");
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [task, setTask] = useState<Task | null>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [system, setSystem] = useState<any>(null);
  const [history, setHistory] = useState<Task[]>([]);
  const [busy, setBusy] = useState(false);
  const resultUrl = task?.best_result?.output_url;

  useEffect(() => {
    api<ModelStatus[]>("/api/v1/models").then(setModels).catch(console.error);
    api<any>("/api/v1/system").then(setSystem).catch(console.error);
    api<Task[]>("/api/v1/history").then(setHistory).catch(console.error);
  }, []);

  useEffect(() => {
    if (!task || ["completed", "failed", "cancelled"].includes(task.status)) return;
    const id = window.setInterval(async () => {
      const next = await api<Task>(`/api/v1/tasks/${task.task_id}`);
      setTask(next);
      api<Task[]>("/api/v1/history").then(setHistory).catch(console.error);
    }, 900);
    return () => window.clearInterval(id);
  }, [task]);

  const onFile = (f: File | null) => {
    setFile(f);
    setAnalysis(null);
    setImageId("");
    setTask(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(f ? URL.createObjectURL(f) : "");
  };

  const uploadAnalyze = async () => {
    if (!file) return;
    setBusy(true);
    const fd = new FormData();
    fd.append("file", file);
    const data = await api<any>("/api/v1/images/analyze", { method: "POST", body: fd });
    setImageId(data.file.file_id);
    setAnalysis(data.analysis);
    setBusy(false);
  };

  const start = async () => {
    let id = imageId;
    if (!id) {
      await uploadAnalyze();
      id = imageId;
    }
    if (!id && !file) return;
    if (!id) {
      const fd = new FormData();
      fd.append("file", file!);
      const uploaded = await api<any>("/api/v1/images/upload", { method: "POST", body: fd });
      id = uploaded.file_id;
      setImageId(id);
    }
    const created = await api<Task>("/api/v1/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: id, user_goal: goal, mode, priority, model_id: mode === "manual" ? model : undefined })
    });
    setTask(created);
  };

  const availableSummary = useMemo(() => models.filter(m => m.available).map(m => m.display_name).join(", ") || "暂无真实权重可用", [models]);

  return <div className="app">
    <aside>
      <div className="brand"><Brain size={26}/><div><h1>VisionRestore Agent</h1><span>多模型协同低照度图像增强</span></div></div>
      <nav>
        <a href="#workbench"><ImageUp size={18}/>工作台</a>
        <a href="#models"><SlidersHorizontal size={18}/>模型管理</a>
        <a href="#history"><Activity size={18}/>历史记录</a>
        <a href="#settings"><Settings size={18}/>系统设置</a>
      </nav>
      <div className="statusBox">
        <b>硬件</b>
        <span>{system?.hardware?.gpu_name || "未检测到 CUDA GPU"}</span>
        <span>CUDA: {String(system?.hardware?.cuda_available ?? false)}</span>
      </div>
    </aside>
    <main>
      <section id="workbench" className="band">
        <div className="sectionHead"><h2>图像增强工作台</h2><p>只处理 RGB 静态图像；不会提供视频、事件相机或音频功能。</p></div>
        <div className="workgrid">
          <div className="panel uploader" onDrop={e => { e.preventDefault(); onFile(e.dataTransfer.files[0]); }} onDragOver={e => e.preventDefault()}>
            <input id="file" type="file" accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff" onChange={e => onFile(e.target.files?.[0] || null)} />
            <label htmlFor="file"><ImageUp size={28}/><b>拖放或选择 RGB 图像</b><span>{file?.name || "png / jpg / jpeg / bmp / tif / tiff"}</span></label>
            {preview && <img className="preview" src={preview} />}
            <button onClick={uploadAnalyze} disabled={!file || busy}><Activity size={18}/>分析图像</button>
          </div>
          <div className="panel controls">
            <label>自然语言要求<textarea value={goal} onChange={e => setGoal(e.target.value)} /></label>
            <div className="seg">
              {(["auto","manual","compare"] as const).map(x => <button className={mode===x ? "on" : ""} onClick={() => setMode(x)} key={x}>{x}</button>)}
            </div>
            <div className="seg">
              {(["quality","balanced","speed"] as const).map(x => <button className={priority===x ? "on" : ""} onClick={() => setPriority(x)} key={x}>{x}</button>)}
            </div>
            {mode === "manual" && <select value={model} onChange={e => setModel(e.target.value)}>{models.map(m => <option key={m.model_id} value={m.model_id}>{m.display_name}</option>)}</select>}
            <button className="primary" onClick={start} disabled={!file}><Play size={18}/>开始增强</button>
            {task && !["completed","failed","cancelled"].includes(task.status) && <button onClick={() => api(`/api/v1/tasks/${task.task_id}/cancel`, {method:"POST"}).then(setTask as any)}><RotateCcw size={18}/>取消任务</button>}
          </div>
        </div>
      </section>
      <section className="band metrics">
        <div className="sectionHead"><h2>图像分析区</h2><p>检测结果为统计估计值。</p></div>
        <Metric title="分辨率" value={analysis ? `${analysis.width}×${analysis.height}` : "-"} />
        <Metric title="平均亮度" value={fmt(analysis?.mean_luminance)} />
        <Metric title="暗像素比例" value={pct(analysis?.dark_pixel_ratio)} />
        <Metric title="过曝比例" value={pct(analysis?.overexposed_pixel_ratio)} />
        <Metric title="噪声估计" value={fmt(analysis?.noise_estimate)} />
        <Metric title="清晰度估计" value={fmt(analysis?.laplacian_sharpness)} />
        <Metric title="色偏估计" value={analysis?.color_cast_label || "-"} />
        <Metric title="推荐可用模型" value={availableSummary} />
      </section>
      <section className="band">
        <div className="sectionHead"><h2>任务与结果</h2><p>{task ? `${task.status} · ${Math.round((task.progress || 0) * 100)}%` : "等待任务"}</p></div>
        {task && <div className="progress"><span style={{width: `${Math.round((task.progress || 0) * 100)}%`}} /></div>}
        <div className="resultGrid">
          <div className="panel imageFrame">{preview ? <img src={preview}/> : <span>原图预览</span>}</div>
          <div className="panel imageFrame">{resultUrl ? <img src={resultUrl}/> : <span>{task?.error || "增强图将在真实模型完成推理后显示"}</span>}</div>
        </div>
        <div className="log">{(task?.logs || []).map((l,i)=><p key={i}>{l}</p>)}</div>
        <div className="actions">
          {resultUrl && <a className="button" href={resultUrl} download><Download size={18}/>保存增强图像</a>}
          {task?.task_id && <a className="button" href={`/api/v1/tasks/${task.task_id}/report`}><FileJson size={18}/>导出 Markdown 报告</a>}
        </div>
      </section>
      <section id="models" className="band">
        <div className="sectionHead"><h2>模型管理</h2><p>页面显示真实安装状态，未安装不会伪装可用。</p></div>
        <div className="modelList">{models.map(m => <div className="model" key={m.model_id}><b>{m.display_name}</b><span>{m.description}</span><code>{m.available ? "available" : "not installed"}</code><small>{m.status_message}</small><a href={m.repository_url} target="_blank">官方仓库</a></div>)}</div>
      </section>
      <section id="history" className="band">
        <div className="sectionHead"><h2>历史记录</h2><p>SQLite 仅保存元数据，不保存图像二进制。</p></div>
        <table><tbody>{history.map(h => <tr key={h.task_id}><td>{h.task_id.slice(0,8)}</td><td>{h.status}</td><td>{h.plan?.selected_model || "-"}</td><td>{h.best_result?.score ?? "-"}</td></tr>)}</tbody></table>
      </section>
      <section id="settings" className="band">
        <div className="sectionHead"><h2>系统设置</h2><p>API 认证默认关闭，仅允许本地前端来源。</p></div>
        <pre>{JSON.stringify(system, null, 2)}</pre>
      </section>
    </main>
  </div>
}

function Metric({title,value}:{title:string;value:any}) { return <div className="metric"><span>{title}</span><b>{value ?? "-"}</b></div> }
const fmt = (n:any) => typeof n === "number" ? n.toFixed(2) : "-";
const pct = (n:any) => typeof n === "number" ? `${(n*100).toFixed(1)}%` : "-";

createRoot(document.getElementById("root")!).render(<App />);

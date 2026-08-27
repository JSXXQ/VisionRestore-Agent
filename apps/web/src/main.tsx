import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  Brain,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Cpu,
  Database,
  Download,
  Eye,
  FileImage,
  Gauge,
  History,
  ImageUp,
  Layers,
  ListChecks,
  Monitor,
  Play,
  RefreshCw,
  RotateCcw,
  Settings,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import "./styles.css";

type Api<T> = { success: boolean; data: T };
type Priority = "quality" | "balanced" | "speed";
type Mode = "auto" | "manual" | "compare";

type Weight = {
  checkpoint_id: string;
  display_name: string;
  status: string;
  size_bytes?: number;
  sha256?: string;
  domain?: string;
  default?: boolean;
  auto_route?: boolean;
  path?: string;
};

type ModelStatus = {
  model_id: string;
  display_name: string;
  description: string;
  available: boolean;
  status_message: string;
  license_name: string;
  repository_url: string;
  capabilities: { weights?: Weight[]; auto_route?: boolean; manual_only_note?: string; task_type?: string };
};

type Analysis = {
  width: number;
  height: number;
  channels: number;
  format: string;
  bit_depth: number;
  mean_luminance: number;
  median_luminance: number;
  grayscale_histogram: number[];
  dark_pixel_ratio: number;
  bright_pixel_ratio: number;
  overexposed_pixel_ratio: number;
  color_cast_index: number;
  color_cast_label: string;
  dynamic_range: number;
  rms_contrast: number;
  laplacian_sharpness: number;
  noise_estimate: number;
  image_entropy: number;
  local_luminance_non_uniformity: number;
  suggest_tile_inference: boolean;
  estimated_memory_mb: number;
  notes: string[];
};

type Intent = {
  priority: Priority | string;
  scene: string;
  preferences: Record<string, boolean>;
  raw_text: string;
  evidence: string[];
};

type Candidate = {
  candidate_id?: string;
  model_id: string;
  checkpoint_id?: string;
  role?: string;
  output_file_id?: string;
  output_url?: string;
  status: string;
  score: number;
  final_score?: number;
  planning_score?: number;
  metrics: Record<string, unknown>;
  parameters: Record<string, unknown>;
  runtime_ms: number;
  peak_memory_mb: number;
  is_mock: boolean;
  adapter_class?: string;
  checkpoint_path?: string;
  checkpoint_sha256?: string;
  worker_python?: string;
  device?: string;
  precision?: string;
  input_sha256?: string;
  output_sha256?: string;
  error?: string;
};

type Plan = {
  selected_model: string;
  selected_checkpoint?: string;
  selection_reason: string;
  parameters: Record<string, unknown>;
  fallback_models: string[];
  fallback_checkpoints: string[];
  evaluation_strategy: string;
  max_retries: number;
  expected_memory_mb: number;
  expected_runtime_ms: number;
  expected_risks: string[];
};

type Task = {
  task_id: string;
  status: string;
  image_id: string;
  user_goal: string;
  analysis_mode?: AnalysisMode;
  progress: number;
  task_mode?: string;
  logs: string[];
  analysis?: Analysis;
  ai_analysis?: AIAnalysis;
  user_intent?: Intent;
  hardware_info?: Record<string, unknown>;
  model_candidates: Array<Record<string, unknown>>;
  checkpoint_candidates: Array<Record<string, unknown>>;
  region_constraints?: RegionConstraint[];
  plan?: Plan;
  candidate_plan?: Record<string, unknown>;
  candidate_ranking?: Record<string, unknown>;
  postprocess_recommendation?: PostprocessRecommendation;
  postprocess_history?: Array<Record<string, unknown>>;
  candidates: Candidate[];
  best_result?: Candidate;
  error?: string;
  report_file_id?: string;
  final_recommendation?: string;
};

type PostprocessRecommendation = {
  denoise_recommended: boolean;
  denoise_confidence: number;
  denoise_reason: string[];
  denoise_risk: string[];
  preferred_denoiser: string;
  super_resolution_recommended: boolean;
  sr_confidence: number;
  sr_reason: string[];
  sr_risk: string[];
  preferred_sr_model: string;
  preferred_scale: number;
};

type PostprocessResult = {
  task_id: string;
  operation: string;
  decision: string;
  accepted: boolean;
  executed: boolean;
  next_status?: string;
  model_id?: string;
  message: string;
  metadata?: { candidate?: Candidate; error?: string };
};

type SystemInfo = {
  hardware?: Record<string, unknown>;
  settings?: Record<string, unknown>;
};

type RealESRGANStatus = {
  pytorch_backend?: { available?: boolean; status_message?: string; packages?: Record<string, { available: boolean; error?: string | null }> };
  installed_weight_count?: number;
  missing_expected_weights?: Array<Record<string, unknown>>;
  available_scales?: number[];
  mambair_realsr_enabled?: boolean;
};

type IQAStatus = {
  available: boolean;
  status_message: string;
  metrics: string[];
  fallback?: string;
};

type AnalysisMode = "local" | "text_only" | "multimodal";

type RegionConstraint = {
  target: string;
  constraint_type: "avoid_overexposure" | "preserve_detail" | "reduce_noise";
  bbox: [number, number, number, number];
  confidence?: number;
  priority: "hard" | "soft";
  source?: "user" | "multimodal" | "local_highlight_detector" | "api";
  max_overexposed_ratio?: number;
  max_luminance_p95?: number;
  reason?: string;
};

type AIProviderStatus = {
  provider_id: string;
  display_name: string;
  implemented: boolean;
  configured: boolean;
  healthy: boolean;
  supports_image: boolean;
  current_model?: string;
  last_error?: string;
  error_code?: string;
};

type AIProviderHealth = {
  provider_id: string;
  implemented: boolean;
  configured: boolean;
  healthy: boolean;
  supports_image: boolean;
  current_model?: string;
  last_error?: string;
  error_code?: string;
};

type AIModelSuggestion = {
  model_id: string;
  score: number;
  reason: string;
};

type AICheckpointSuggestion = {
  model_id: string;
  checkpoint_id: string;
  score: number;
  reason: string;
};

type AIAnalysis = {
  provider: string;
  model: string;
  scene: string;
  subscene: string;
  scene_confidence: number;
  main_subjects: string[];
  important_light_sources: string[];
  critical_regions: string[];
  interpreted_intent: string[];
  model_candidates: AIModelSuggestion[];
  checkpoint_candidates: AICheckpointSuggestion[];
  reasoning_summary: string;
  warnings: string[];
  confidence: number;
  fallback_used: boolean;
  failure_reason?: string;
  validation_passed: boolean;
  validation_errors: string[];
  local_validation: Record<string, unknown>;
  adopted: boolean;
  adoption_reason: string;
  rejection_reason?: string;
  analysis_mode: AnalysisMode;
  sent_image: boolean;
  runtime_ms: number;
};

type AISettings = {
  enabled: boolean;
  provider: string;
  send_image: boolean;
  send_metrics: boolean;
  fallback_to_local: boolean;
  image_max_edge: number;
  image_quality: number;
  timeout_seconds: number;
  providers: Record<string, { configured: boolean; healthy: boolean; implemented: boolean; supports_image: boolean; current_model?: string }>;
};

const api = async <T,>(url: string, init?: RequestInit): Promise<T> => {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(await res.text());
  const body = (await res.json()) as Api<T>;
  return body.data;
};

const priorityLabels: Record<Priority, string> = {
  quality: "质量优先",
  balanced: "均衡模式",
  speed: "速度优先",
};

const modeLabels: Record<Mode, string> = {
  auto: "自动Agent",
  manual: "手动选择",
  compare: "多模型对比",
};

const statusText: Record<string, string> = {
  queued: "等待",
  analyzing: "分析图像",
  parsing_intent: "解析需求",
  inspecting_hardware: "检查资源",
  routing_model: "模型路由",
  routing_checkpoint: "权重路由",
  loading_model: "加载模型",
  running: "真实推理",
  evaluating: "评价结果",
  fallback_running: "备用策略",
  selecting_result: "选择结果",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [imageId, setImageId] = useState("");
  const [goal, setGoal] = useState("自然增强暗部，保持原始颜色，路灯不要过曝，质量优先，这是一张室外城市夜景。");
  const [mode, setMode] = useState<Mode>("auto");
  const [priority, setPriority] = useState<Priority>("quality");
  const [model, setModel] = useState("retinexformer");
  const [checkpoint, setCheckpoint] = useState("lol_v2_real");
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [intent, setIntent] = useState<Intent | null>(null);
  const [task, setTask] = useState<Task | null>(null);
  const [history, setHistory] = useState<Task[]>([]);
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("multimodal");
  const [analysisModeTouched, setAnalysisModeTouched] = useState(false);
  const [aiProviders, setAiProviders] = useState<AIProviderStatus[]>([]);
  const [aiSettings, setAiSettings] = useState<AISettings | null>(null);
  const [aiAnalysis, setAiAnalysis] = useState<AIAnalysis | null>(null);
  const [postprocessRecommendation, setPostprocessRecommendation] = useState<PostprocessRecommendation | null>(null);
  const [postprocessBusy, setPostprocessBusy] = useState(false);
  const [realesrganStatus, setRealesrganStatus] = useState<RealESRGANStatus | null>(null);
  const [iqaStatus, setIqaStatus] = useState<IQAStatus | null>(null);
  const [regionConstraints, setRegionConstraints] = useState<RegionConstraint[]>([]);
  const [tab, setTab] = useState("candidates");
  const [view, setView] = useState("workbench");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");

  const refresh = () => {
    api<ModelStatus[]>("/api/v1/models").then(setModels).catch((err) => setNotice(String(err)));
    api<SystemInfo>("/api/v1/system").then(setSystem).catch((err) => setNotice(String(err)));
    api<Task[]>("/api/v1/history").then(setHistory).catch(() => undefined);
    api<AIProviderStatus[]>("/api/v1/ai/providers").then(setAiProviders).catch(() => undefined);
    api<AISettings>("/api/v1/ai/settings").then(setAiSettings).catch(() => undefined);
    api<RealESRGANStatus>("/api/v2/models/realesrgan/status").then(setRealesrganStatus).catch(() => undefined);
    api<IQAStatus>("/api/v2/iqa/status").then(setIqaStatus).catch(() => undefined);
  };

  useEffect(refresh, []);

  useEffect(() => {
    const applyHash = () => {
      const next = window.location.hash.replace("#", "");
      if (["workbench", "models", "history", "settings"].includes(next)) setView(next);
    };
    applyHash();
    window.addEventListener("hashchange", applyHash);
    return () => window.removeEventListener("hashchange", applyHash);
  }, []);

  useEffect(() => {
    if (!task || taskIsPausedOrDone(task.status)) return;
    const id = window.setInterval(async () => {
      const next = await api<Task>(`/api/v1/tasks/${task.task_id}`);
      setTask(next);
      setAnalysis(next.analysis || null);
      setIntent(next.user_intent || null);
      setAiAnalysis(next.ai_analysis || null);
      if (taskIsPausedOrDone(next.status)) refresh();
    }, 1000);
    return () => window.clearInterval(id);
  }, [task?.task_id, task?.status]);

  useEffect(() => {
    if (!analysisModeTouched && aiSettings?.enabled && analysisMode === "local") {
      setAnalysisMode("multimodal");
    }
  }, [analysisModeTouched, aiSettings?.enabled, analysisMode]);

  useEffect(() => {
    if (task?.best_result?.output_file_id && ["completed", "awaiting_denoise_confirmation", "awaiting_sr_confirmation"].includes(task.status)) {
      if (task.postprocess_recommendation) setPostprocessRecommendation(task.postprocess_recommendation);
      loadPostprocessRecommendation(task).catch(() => undefined);
    }
  }, [task?.task_id, task?.status, task?.best_result?.output_file_id]);

  const selectedModel = models.find((item) => item.model_id === model);
  const weights = selectedModel?.capabilities?.weights || [];
  const installedWeights = models.flatMap((item) => item.capabilities?.weights || []).filter((item) => item.status === "found").length;
  const activeAnalysis = task?.analysis || analysis;
  const activeIntent = task?.user_intent || intent;
  const resultUrl = task?.best_result?.output_url;
  const previewUrl = preview || (task?.image_id ? `/api/v1/files/${task.image_id}` : "");
  const hardware = system?.hardware || task?.hardware_info || {};

  useEffect(() => {
    if (weights.length && !weights.some((item) => item.checkpoint_id === checkpoint)) {
      setCheckpoint(weights[0].checkpoint_id);
    }
  }, [model, models]);

  useEffect(() => {
    if (!task && !file && history.length) {
      const latest = history[0];
      setTask(latest);
      setAnalysis(latest.analysis || null);
      setIntent(latest.user_intent || null);
      setAiAnalysis(latest.ai_analysis || null);
      setGoal(latest.user_goal || goal);
    }
  }, [history, task, file]);

  const onFile = (f: File | null) => {
    setFile(f);
    setAnalysis(null);
    setIntent(null);
    setAiAnalysis(null);
    setPostprocessRecommendation(null);
    setRegionConstraints([]);
    setImageId("");
    setTask(null);
    setNotice("");
    if (preview) URL.revokeObjectURL(preview);
    setPreview(f ? URL.createObjectURL(f) : "");
  };

  const parseIntent = async () => {
    const parsed = await api<Intent>("/api/v1/intent/parse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: goal,
        priority,
        mode,
        model_id: mode === "manual" ? model : undefined,
        checkpoint_id: mode === "manual" ? checkpoint : undefined,
      }),
    });
    setIntent(parsed);
    return parsed;
  };

  const runAiAnalyze = async (id: string) => {
    const response = await api<{ analysis: AIAnalysis; preview: unknown }>("/api/v1/ai/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: id,
        user_request: goal,
        analysis_mode: analysisMode,
        manual_model: mode === "manual" ? model : undefined,
        manual_checkpoint: mode === "manual" ? checkpoint : undefined,
      }),
    });
    setAiAnalysis(response.analysis);
  };

  const uploadAnalyze = async () => {
    if (!file) return;
    setBusy(true);
    setNotice("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      const data = await api<{ file: { file_id: string }; analysis: Analysis }>("/api/v1/images/analyze", { method: "POST", body: fd });
      setImageId(data.file.file_id);
      setAnalysis(data.analysis);
      await parseIntent();
      await runAiAnalyze(data.file.file_id);
    } catch (err) {
      setNotice(String(err));
    } finally {
      setBusy(false);
    }
  };

  const loadPostprocessRecommendation = async (target: Task | null = task) => {
    if (!target?.task_id || !target.best_result?.output_file_id) return null;
    const response = await api<{ recommendation: PostprocessRecommendation | null; message?: string }>("/api/v2/tasks/" + target.task_id + "/recommendations");
    setPostprocessRecommendation(response.recommendation || null);
    return response.recommendation || null;
  };

  const runPostprocessDenoise = async (modelId: "lpdm" | "nafnet", checkpointId: string) => {
    if (!task?.task_id || !task.best_result?.output_file_id) return;
    setPostprocessBusy(true);
    setNotice("");
    try {
      const result = await api<PostprocessResult>("/api/v2/tasks/" + task.task_id + "/postprocess/decision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          operation: "denoise",
          decision: modelId === "lpdm" ? "accept" : "choose_model",
          model_id: modelId,
          parameters: { checkpoint_id: checkpointId, precision: "fp32", device: "cuda" },
        }),
      });
      const next = await api<Task>("/api/v2/tasks/" + task.task_id);
      setTask(next);
      setNotice(result.message);
      setTab("candidates");
      await loadPostprocessRecommendation(next);
      refresh();
    } catch (err) {
      setNotice(String(err));
    } finally {
      setPostprocessBusy(false);
    }
  };

  const runSuperResolution = async (scale = 2, allowX4 = false) => {
    if (!task?.task_id || !task.best_result?.output_file_id) return;
    setPostprocessBusy(true);
    setNotice("");
    try {
      const result = await api<PostprocessResult>("/api/v2/tasks/" + task.task_id + "/super-resolution/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scale, allow_x4: allowX4, parameters: { precision: "fp16", device: "cuda" } }),
      });
      const next = await api<Task>("/api/v2/tasks/" + task.task_id);
      setTask(next);
      setNotice(result.message);
      setTab("candidates");
      await loadPostprocessRecommendation(next);
      refresh();
    } catch (err) {
      setNotice(String(err));
    } finally {
      setPostprocessBusy(false);
    }
  };

  const skipSuperResolution = async () => {
    if (!task?.task_id) return;
    setPostprocessBusy(true);
    setNotice("");
    try {
      const result = await api<PostprocessResult>("/api/v2/tasks/" + task.task_id + "/super-resolution/skip", { method: "POST" });
      const next = await api<Task>("/api/v2/tasks/" + task.task_id);
      setTask(next);
      setNotice(result.message);
      refresh();
    } catch (err) {
      setNotice(String(err));
    } finally {
      setPostprocessBusy(false);
    }
  };

  const rollbackPostprocess = async () => {
    if (!task?.task_id || task.best_result?.parameters?.role !== "postprocess") return;
    setPostprocessBusy(true);
    setNotice("");
    try {
      const result = await api<PostprocessResult>("/api/v2/tasks/" + task.task_id + "/postprocess/rollback", { method: "POST" });
      const next = await api<Task>("/api/v2/tasks/" + task.task_id);
      setTask(next);
      setNotice(result.message);
      setTab("candidates");
      await loadPostprocessRecommendation(next);
      refresh();
    } catch (err) {
      setNotice(String(err));
    } finally {
      setPostprocessBusy(false);
    }
  };
  const start = async () => {
    if (!file) return;
    setBusy(true);
    setNotice("");
    try {
      let id = imageId;
      if (!id) {
        const fd = new FormData();
        fd.append("file", file);
        const uploaded = await api<{ file_id: string }>("/api/v1/images/upload", { method: "POST", body: fd });
        id = uploaded.file_id;
        setImageId(id);
      }
      const created = await api<Task>("/api/v2/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_id: id,
          user_goal: goal,
          mode,
          priority,
          analysis_mode: analysisMode,
          model_id: mode === "manual" ? model : undefined,
          checkpoint_id: mode === "manual" ? checkpoint : undefined,
          parameters: {
            region_constraints: regionConstraints,
          },
        }),
      });
      setTask(created);
      setTab("logs");
      setView("workbench");
    } catch (err) {
      setNotice(String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app-shell">
      <TopBar hardware={hardware} models={models} installedWeights={installedWeights} />
      <SideNav view={view} setView={setView} />
      <main className="content">
        {view === "workbench" && (
          <>
            <div className="center-column">
              <InputPanel
                file={file}
                preview={previewUrl}
                goal={goal}
                setGoal={setGoal}
                mode={mode}
                setMode={setMode}
                analysisMode={analysisMode}
                setAnalysisMode={(value) => {
                  setAnalysisModeTouched(true);
                  setAnalysisMode(value);
                }}
                aiSettings={aiSettings}
                priority={priority}
                setPriority={setPriority}
                models={models}
                model={model}
                setModel={setModel}
                checkpoint={checkpoint}
                setCheckpoint={setCheckpoint}
                weights={weights}
                regionConstraints={regionConstraints}
                setRegionConstraints={setRegionConstraints}
                busy={busy}
                onFile={onFile}
                onAnalyze={uploadAnalyze}
                onStart={start}
              />
              <ComparePanel preview={previewUrl} resultUrl={resultUrl} task={task} analysis={activeAnalysis} regionConstraints={(task?.region_constraints as RegionConstraint[] | undefined) || regionConstraints} />
              <BottomPanel tab={tab} setTab={setTab} task={task} preview={previewUrl} />
            </div>
            <DecisionCenter task={task} analysis={activeAnalysis} intent={activeIntent} goal={goal} models={models} aiAnalysis={task?.ai_analysis || aiAnalysis} analysisMode={task?.analysis_mode || analysisMode} aiProviders={aiProviders} aiSettings={aiSettings} postprocessRecommendation={postprocessRecommendation} postprocessBusy={postprocessBusy} realesrganStatus={realesrganStatus} iqaStatus={iqaStatus} onRefreshPostprocess={() => loadPostprocessRecommendation(task)} onRunPostprocess={runPostprocessDenoise} onRunSuperResolution={runSuperResolution} onSkipSuperResolution={skipSuperResolution} onRollbackPostprocess={rollbackPostprocess} />
          </>
        )}
        {view === "models" && <ModelsPage models={models} refresh={refresh} />}
        {view === "history" && <HistoryPage history={history} />}
        {view === "settings" && <SettingsPage system={system} models={models} aiSettings={aiSettings} aiProviders={aiProviders} refresh={refresh} setNotice={setNotice} />}
      </main>
      {notice && <div className="toast">{notice}</div>}
    </div>
  );
}

function TopBar({ hardware, models, installedWeights }: { hardware: Record<string, unknown>; models: ModelStatus[]; installedWeights: number }) {
  const gpuName = String(hardware.gpu_name || "未检测到 GPU");
  const cuda = Boolean(hardware.cuda_available);
  return (
    <header className="topbar">
      <div className="brand">
        <Brain />
        <div>
          <strong>VisionRestore Agent</strong>
          <span>多模型协同低照度图像增强系统</span>
        </div>
      </div>
      <div className="status-strip">
        <StatusPill icon={<Monitor />} text="本地模式" />
        <StatusPill tone="ok" icon={<ShieldCheck />} text="API正常" />
        <StatusPill tone="gpu" icon={<Cpu />} text={`GPU: ${gpuName}`} />
        <StatusPill tone={cuda ? "ok" : "warn"} icon={<Zap />} text={cuda ? "CUDA 可用" : "CUDA 不可用"} />
        <StatusPill icon={<Database />} text={`已安装模型 ${models.filter((item) => item.available).length}`} />
        <StatusPill icon={<Layers />} text={`权重 ${installedWeights}`} />
      </div>
    </header>
  );
}

function StatusPill({ icon, text, tone = "info" }: { icon: React.ReactNode; text: string; tone?: string }) {
  return <span className={`status-pill ${tone}`}>{icon}{text}</span>;
}

function SideNav({ view, setView }: { view: string; setView: (value: string) => void }) {
  const items = [
    ["workbench", "增强工作台", <Monitor key="i" />],
    ["models", "模型与权重", <Layers key="i" />],
    ["history", "历史任务", <History key="i" />],
    ["settings", "系统设置", <Settings key="i" />],
  ];
  return (
    <aside className="side-nav">
      {items.map(([id, label, icon]) => (
        <button key={String(id)} className={view === id ? "active" : ""} onClick={() => { setView(String(id)); window.location.hash = String(id); }}>
          {icon}
          <span>{label}</span>
        </button>
      ))}
    </aside>
  );
}

function InputPanel(props: {
  file: File | null;
  preview: string;
  goal: string;
  setGoal: (value: string) => void;
  mode: Mode;
  setMode: (value: Mode) => void;
  analysisMode: AnalysisMode;
  setAnalysisMode: (value: AnalysisMode) => void;
  aiSettings: AISettings | null;
  priority: Priority;
  setPriority: (value: Priority) => void;
  models: ModelStatus[];
  model: string;
  setModel: (value: string) => void;
  checkpoint: string;
  setCheckpoint: (value: string) => void;
  weights: Weight[];
  regionConstraints: RegionConstraint[];
  setRegionConstraints: (value: RegionConstraint[]) => void;
  busy: boolean;
  onFile: (file: File | null) => void;
  onAnalyze: () => void;
  onStart: () => void;
}) {
  return (
    <section className="panel input-panel">
      <div className="panel-title">
        <h2>输入图像与需求</h2>
        <span className="mini-note">RGB 静态图像</span>
      </div>
      <div className="input-grid">
        <label
          className={`upload-box ${props.preview ? "has-image" : ""}`}
          onDrop={(e) => {
            e.preventDefault();
            props.onFile(e.dataTransfer.files[0] || null);
          }}
          onDragOver={(e) => e.preventDefault()}
        >
          <input type="file" accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff" onChange={(e) => props.onFile(e.target.files?.[0] || null)} />
          {props.preview ? <img src={props.preview} /> : <><ImageUp /><span>拖拽图像到此处，或点击上传</span><small>仅支持 RGB 图像（.jpg .png .bmp .tif）</small></>}
        </label>
        <div className="input-controls">
          <textarea value={props.goal} maxLength={200} onChange={(e) => props.setGoal(e.target.value)} />
          <div className="counter">{props.goal.length} / 200</div>
          <ControlGroup label="优先级">
            {(Object.keys(priorityLabels) as Priority[]).map((item) => (
              <button key={item} className={props.priority === item ? "active" : ""} onClick={() => props.setPriority(item)}>{priorityLabels[item]}</button>
            ))}
          </ControlGroup>
          <ControlGroup label="增强模式">
            {(Object.keys(modeLabels) as Mode[]).map((item) => (
              <button key={item} className={props.mode === item ? "active" : ""} onClick={() => props.setMode(item)}>{modeLabels[item]}</button>
            ))}
          </ControlGroup>
          <ControlGroup label="分析方式">
            {(["local", "text_only", "multimodal"] as AnalysisMode[]).map((item) => {
              const cloudDisabled = item !== "local" && !props.aiSettings?.enabled;
              const label = item === "local" ? "纯本地分析" : item === "text_only" ? "文字AI分析" : "多模态AI分析";
              return <button key={item} className={props.analysisMode === item ? "active" : ""} disabled={cloudDisabled} title={cloudDisabled ? "尚未配置多模态AI API" : ""} onClick={() => props.setAnalysisMode(item)}>{label}</button>;
            })}
          </ControlGroup>
          {props.mode === "manual" && (
            <div className="manual-selects">
              <select value={props.model} onChange={(e) => props.setModel(e.target.value)}>
                {props.models.map((item) => <option key={item.model_id} value={item.model_id}>{item.display_name}</option>)}
              </select>
              <select value={props.checkpoint} onChange={(e) => props.setCheckpoint(e.target.value)}>
                {props.weights.map((item) => <option key={item.checkpoint_id} value={item.checkpoint_id}>{item.display_name}</option>)}
              </select>
            </div>
          )}
          <RegionConstraintEditor
            preview={props.preview}
            constraints={props.regionConstraints}
            setConstraints={props.setRegionConstraints}
          />
          <div className="action-row">
            <button onClick={props.onAnalyze} disabled={!props.file || props.busy}><Gauge />分析图像</button>
            <button className="primary-action" onClick={props.onStart} disabled={!props.file || props.busy}><Sparkles />开始智能增强</button>
          </div>
        </div>
      </div>
    </section>
  );
}

function ControlGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="control-group"><span>{label}</span><div>{children}</div></div>;
}

function RegionConstraintEditor({ preview, constraints, setConstraints }: { preview: string; constraints: RegionConstraint[]; setConstraints: (value: RegionConstraint[]) => void }) {
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [imageSize, setImageSize] = useState<[number, number] | null>(null);
  const [dragStart, setDragStart] = useState<[number, number] | null>(null);
  const [draft, setDraft] = useState<[number, number, number, number] | null>(null);
  const [target, setTarget] = useState("streetlight");
  const [threshold, setThreshold] = useState(0.08);

  const pointFromEvent = (event: React.MouseEvent<HTMLDivElement>): [number, number] | null => {
    const img = imageRef.current;
    if (!img || !imageSize) return null;
    const rect = img.getBoundingClientRect();
    const x = Math.round(((event.clientX - rect.left) / rect.width) * imageSize[0]);
    const y = Math.round(((event.clientY - rect.top) / rect.height) * imageSize[1]);
    return [Math.max(0, Math.min(imageSize[0], x)), Math.max(0, Math.min(imageSize[1], y))];
  };

  const begin = (event: React.MouseEvent<HTMLDivElement>) => {
    const point = pointFromEvent(event);
    if (!point) return;
    setDragStart(point);
    setDraft([point[0], point[1], point[0], point[1]]);
  };

  const move = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!dragStart) return;
    const point = pointFromEvent(event);
    if (!point) return;
    setDraft([dragStart[0], dragStart[1], point[0], point[1]]);
  };

  const end = () => {
    if (!draft) {
      setDragStart(null);
      return;
    }
    const [a, b, c, d] = draft;
    const bbox: [number, number, number, number] = [Math.min(a, c), Math.min(b, d), Math.max(a, c), Math.max(b, d)];
    setDragStart(null);
    setDraft(null);
    if ((bbox[2] - bbox[0]) < 6 || (bbox[3] - bbox[1]) < 6) return;
    setConstraints([...constraints, {
      target: target.trim() || "streetlight",
      constraint_type: "avoid_overexposure",
      bbox,
      priority: "hard",
      source: "user",
      confidence: 1,
      max_overexposed_ratio: threshold,
      max_luminance_p95: 248,
      reason: "manual ROI highlight protection",
    }]);
  };

  const remove = (index: number) => setConstraints(constraints.filter((_, itemIndex) => itemIndex !== index));

  return (
    <div className="region-editor">
      <div className="region-toolbar">
        <span>ROI约束</span>
        <input value={target} onChange={(event) => setTarget(event.target.value)} />
        <label>
          过曝阈值
          <input type="number" min="0.01" max="0.5" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value) || 0.08)} />
        </label>
      </div>
      {preview ? (
        <div className="region-stage" onMouseDown={begin} onMouseMove={move} onMouseUp={end} onMouseLeave={end}>
          <img
            ref={imageRef}
            src={preview}
            onLoad={(event) => {
              const img = event.currentTarget;
              setImageSize([img.naturalWidth, img.naturalHeight]);
            }}
            draggable={false}
          />
          <RegionOverlay constraints={constraints} imageSize={imageSize} draft={draft} />
        </div>
      ) : (
        <p className="note-line">上传图像后可拖拽框选需要保护的路灯、招牌或其他高光区域。</p>
      )}
      {constraints.length ? (
        <div className="region-list">
          {constraints.map((item, index) => (
            <button key={`${item.target}-${index}`} onClick={() => remove(index)}>
              <ShieldCheck />
              <span>{item.target} [{item.bbox.join(", ")}]</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function RegionOverlay({ constraints, imageSize, draft }: { constraints: RegionConstraint[]; imageSize?: [number, number] | null; draft?: [number, number, number, number] | null }) {
  const boxes = draft ? [...constraints, { target: "draft", bbox: draft, constraint_type: "avoid_overexposure", priority: "hard" } as RegionConstraint] : constraints;
  if (!imageSize || !boxes.length) return null;
  const [width, height] = imageSize;
  return (
    <div className="region-overlay">
      {boxes.map((item, index) => {
        const [a, b, c, d] = item.bbox;
        const x1 = Math.min(a, c);
        const y1 = Math.min(b, d);
        const x2 = Math.max(a, c);
        const y2 = Math.max(b, d);
        const style = {
          left: `${(x1 / width) * 100}%`,
          top: `${(y1 / height) * 100}%`,
          width: `${((x2 - x1) / width) * 100}%`,
          height: `${((y2 - y1) / height) * 100}%`,
        };
        return <span className={`region-box ${item.target === "draft" ? "draft" : ""}`} style={style} key={`${item.target}-${index}`}><b>{item.target}</b></span>;
      })}
    </div>
  );
}

function ComparePanel({ preview, resultUrl, task, analysis, regionConstraints }: { preview: string; resultUrl?: string; task: Task | null; analysis: Analysis | null | undefined; regionConstraints: RegionConstraint[] }) {
  const best = task?.best_result;
  return (
    <section className="panel compare-panel">
      <div className="panel-title">
        <h2>增强结果对比</h2>
        <span className={best?.is_mock ? "tag warn" : "tag ok"}>{best ? (best.is_mock ? "Mock测试结果" : "真实模型结果") : "等待结果"}</span>
      </div>
      <div className="compare-stage">
        <ImageSlot label="原始图像" src={preview} regions={regionConstraints} />
        <div className="split-handle"><ChevronRight /></div>
        <ImageSlot label={task?.error || "增强结果"} src={resultUrl} regions={regionConstraints} />
      </div>
      <div className="result-ribbon">
        <span>推荐结果：{best ? `${modelName(best.model_id)} / ${checkpointName(best.checkpoint_id)}` : "-"}</span>
        <span>综合评分 <b>{best ? fmt(best.score, 1) : "-"}</b></span>
        <span>推理时间 <b>{best ? seconds(best.runtime_ms) : "-"}</b></span>
        <span>峰值显存 <b>{best ? `${fmt(best.peak_memory_mb, 1)}MB` : "-"}</b></span>
        <span>尺寸 <b>{analysis ? `${analysis.width}×${analysis.height}` : "-"}</b></span>
      </div>
      <div className="action-row compact">
        {resultUrl && <a className="button" href={resultUrl} download><Download />保存图像</a>}
        {task?.task_id && <a className="button" href={`/api/v1/tasks/${task.task_id}/report`}><BarChart3 />导出报告</a>}
      </div>
    </section>
  );
}

function ImageSlot({ label, src, regions = [] }: { label: string; src?: string; regions?: RegionConstraint[] }) {
  const [imageSize, setImageSize] = useState<[number, number] | null>(null);
  return (
    <div className="image-slot">
      <span>{label}</span>
      {src ? (
        <div className="slot-image-wrap">
          <img src={src} onLoad={(event) => setImageSize([event.currentTarget.naturalWidth, event.currentTarget.naturalHeight])} />
          <RegionOverlay constraints={regions} imageSize={imageSize} />
        </div>
      ) : <div className="empty-image"><FileImage />{label}</div>}
    </div>
  );
}

function DecisionCenter({ task, analysis, intent, goal, models, aiAnalysis, analysisMode, aiProviders, aiSettings, postprocessRecommendation, postprocessBusy, realesrganStatus, iqaStatus, onRefreshPostprocess, onRunPostprocess, onRunSuperResolution, onSkipSuperResolution, onRollbackPostprocess }: { task: Task | null; analysis: Analysis | null | undefined; intent: Intent | null | undefined; goal: string; models: ModelStatus[]; aiAnalysis: AIAnalysis | null; analysisMode: AnalysisMode; aiProviders: AIProviderStatus[]; aiSettings: AISettings | null; postprocessRecommendation: PostprocessRecommendation | null; postprocessBusy: boolean; realesrganStatus: RealESRGANStatus | null; iqaStatus: IQAStatus | null; onRefreshPostprocess: () => void; onRunPostprocess: (modelId: "lpdm" | "nafnet", checkpointId: string) => void; onRunSuperResolution: (scale?: number, allowX4?: boolean) => void; onSkipSuperResolution: () => void; onRollbackPostprocess: () => void }) {
  return (
    <aside className="agent-panel">
      <div className="panel-title">
        <h2>Agent决策中心</h2>
        <span className="mini-note">{statusText[task?.status || "queued"] || "等待"}</span>
      </div>
      <AnalysisCard analysis={analysis} />
      <IntentCard intent={intent} goal={goal} />
      <SafeSemanticCard aiAnalysis={aiAnalysis} analysisMode={analysisMode} aiProviders={aiProviders} aiSettings={aiSettings} />
      <RegionMonitorCard task={task} />
      <PostprocessCard task={task} recommendation={postprocessRecommendation} busy={postprocessBusy} realesrganStatus={realesrganStatus} iqaStatus={iqaStatus} onRefresh={onRefreshPostprocess} onRun={onRunPostprocess} onRunSr={onRunSuperResolution} onSkipSr={onSkipSuperResolution} onRollback={onRollbackPostprocess} />
      <RouteCard title="模型架构路由" items={task?.model_candidates || []} selected={task?.plan?.selected_model} kind="model" />
      <WeightRouteCard items={task?.checkpoint_candidates || []} selected={task?.plan?.selected_checkpoint} />
      <TimelineCard task={task} />
      <TechDetails title="查看技术详情" rows={technicalRows(task?.best_result, models)} />
    </aside>
  );
}

function AnalysisCard({ analysis }: { analysis: Analysis | null | undefined }) {
  const rows = analysis ? [
    ["平均亮度", fmt(analysis.mean_luminance, 1), level(analysis.mean_luminance, [45, 120], ["低", "正常", "高"])],
    ["暗像素比例", pct(analysis.dark_pixel_ratio), level(analysis.dark_pixel_ratio, [0.25, 0.55], ["低", "中", "高"])],
    ["过曝风险", pct(analysis.overexposed_pixel_ratio), level(analysis.overexposed_pixel_ratio, [0.01, 0.04], ["低", "中", "高"])],
    ["噪声估计", fmt(analysis.noise_estimate, 2), level(analysis.noise_estimate, [8, 16], ["低", "中", "较高"])],
    ["清晰度", fmt(analysis.laplacian_sharpness, 2), level(analysis.laplacian_sharpness, [20, 80], ["偏低", "正常", "较高"])],
    ["场景判断", analysis.mean_luminance < 80 ? "室外夜景" : "常规场景", analysis.suggest_tile_inference ? "建议分块" : "整图处理"],
  ] : [];
  return (
    <Card title="图像分析" icon={<Eye />}>
      {analysis ? (
        <>
          <div className="metric-grid">
            <Metric label="分辨率" value={`${analysis.width}×${analysis.height}`} />
            <Metric label="格式" value={`${analysis.format.toUpperCase()} / ${analysis.bit_depth}bit`} />
            <Metric label="中位亮度" value={fmt(analysis.median_luminance, 1)} />
            <Metric label="动态范围" value={fmt(analysis.dynamic_range, 1)} />
          </div>
          <Histogram values={analysis.grayscale_histogram} />
          <div className="kv-list">{rows.map(([label, value, tone]) => <InfoRow key={label} label={label} value={value} tone={tone} />)}</div>
        </>
      ) : <EmptyText text="上传并分析后显示真实图像统计。" />}
    </Card>
  );
}

function IntentCard({ intent, goal }: { intent: Intent | null | undefined; goal: string }) {
  const prefs = intent?.preferences || {};
  const tags = [
    intent?.priority === "quality" ? "质量优先" : intent?.priority === "speed" ? "速度优先" : "均衡",
    sceneLabel(intent?.scene),
    prefs.preserve_color ? "色彩保持" : "",
    prefs.protect_highlights ? "高光保护" : "",
    prefs.recover_shadows ? "暗部恢复" : "",
    prefs.natural_result ? "自然结果" : "",
    prefs.reduce_noise ? "降噪控制" : "",
  ].filter(Boolean);
  return (
    <Card title="用户意图" icon={<ListChecks />}>
      <p className="intent-text">{intent?.raw_text || goal || "等待用户需求"}</p>
      <div className="tag-row">{tags.map((tag) => <span className="tag" key={tag}>{tag}</span>)}</div>
      {!!intent?.evidence?.length && <p className="evidence">{intent.evidence.join(" / ")}</p>}
    </Card>
  );
}

function RegionMonitorCard({ task }: { task: Task | null }) {
  const constraints = task?.region_constraints || [];
  const bestRegion = (task?.best_result?.metrics?.region_constraints || null) as Record<string, unknown> | null;
  if (!constraints.length && !bestRegion) {
    return (
      <Card title="区域监视" icon={<ShieldCheck />}>
        <EmptyText text="可在输入图上框选路灯、招牌等区域，Agent 会在真实输出后单独检查。." />
      </Card>
    );
  }
  const items = (bestRegion?.items || []) as Array<Record<string, unknown>>;
  return (
    <Card title="区域监视" icon={<ShieldCheck />}>
      {constraints.length ? (
        <div className="region-monitor-list">
          {constraints.map((item, index) => (
            <div className="region-monitor-item" key={`${item.target}-${index}`}>
              <strong>{item.target}</strong>
              <span>{item.constraint_type} / {item.priority}</span>
              <small>{item.bbox.join(", ")}</small>
            </div>
          ))}
        </div>
      ) : null}
      {bestRegion ? (
        <div className="region-eval-summary">
          <InfoRow label="最佳结果ROI" value={bestRegion.hard_failed ? "未通过" : "通过"} tone={bestRegion.hard_failed ? "warn" : "ok"} />
          <InfoRow label="区域得分" value={fmt(Number(bestRegion.score || 0) * 100, 1)} />
          {items.slice(0, 3).map((item, index) => (
            <p className="note-line" key={index}>{String(item.target || "ROI")}：过曝 {pct(Number(item.overexposed_ratio_after || 0))}，P95 {fmt(Number(item.luminance_p95_after || 0), 1)}</p>
          ))}
        </div>
      ) : <p className="note-line">任务运行后显示每个 ROI 的过曝和亮度检查。</p>}
    </Card>
  );
}

function SemanticCard({ aiAnalysis, analysisMode, aiProviders, aiSettings }: { aiAnalysis: AIAnalysis | null; analysisMode: AnalysisMode; aiProviders: AIProviderStatus[]; aiSettings: AISettings | null }) {
  const activeProvider = aiProviders.find((item) => item.provider_id === aiSettings?.provider);
  const modeText = analysisMode === "local" ? "纯本地分析" : analysisMode === "text_only" ? "文字AI分析" : "多模态AI分析";
  if (!aiAnalysis) {
    return (
      <Card title="场景语义" icon={<Brain />}>
        <div className="ai-status">
          <InfoRow label="分析方式" value={modeText} />
          <InfoRow label="AI提供商" value={activeProvider?.display_name || "disabled"} />
          <InfoRow label="状态" value={aiSettings?.enabled ? "等待分析" : "当前使用本地规则分析"} />
        </div>
        {!aiSettings?.enabled && <p className="note-line">多模态 AI 未启用；不会发送图像或文字到外部 API。</p>}
      </Card>
    );
  }
  const tags = [
    aiAnalysis.scene && `场景：${aiAnalysis.scene}`,
    aiAnalysis.subscene && `子场景：${aiAnalysis.subscene}`,
    ...aiAnalysis.main_subjects.slice(0, 3).map((item) => `主体：${item}`),
    ...aiAnalysis.important_light_sources.slice(0, 3).map((item) => `光源：${item}`),
    ...aiAnalysis.critical_regions.slice(0, 3).map((item) => `高风险：${item}`),
  ].filter(Boolean);
  return (
    <Card title="场景语义" icon={<Brain />}>
      <div className="metric-grid">
        <Metric label="分析方式" value={modeText} />
        <Metric label="提供商/模型" value={`${aiAnalysis.provider} / ${aiAnalysis.model}`} />
        <Metric label="是否发图" value={aiAnalysis.sent_image ? "发送缩略预览" : "未发送图像"} />
        <Metric label="置信度" value={fmt(aiAnalysis.confidence || aiAnalysis.scene_confidence, 2)} />
      </div>
      <div className="tag-row">{tags.map((tag) => <span className="tag" key={tag}>{tag}</span>)}</div>
      <p className="evidence">{aiAnalysis.reasoning_summary || "本地统计分析无法可靠识别复杂场景语义。"}</p>
      {!!aiAnalysis.warnings?.length && <p className="note-line">{aiAnalysis.warnings.join(" / ")}</p>}
      {aiAnalysis.fallback_used && <span className="tag warn">已回退本地规则</span>}
    </Card>
  );
}

function SafeSemanticCard({ aiAnalysis, analysisMode, aiProviders, aiSettings }: { aiAnalysis: AIAnalysis | null; analysisMode: AnalysisMode; aiProviders: AIProviderStatus[]; aiSettings: AISettings | null }) {
  const activeProvider = aiProviders.find((item) => item.provider_id === aiSettings?.provider);
  const modeText = analysisMode === "local" ? "纯本地分析" : analysisMode === "text_only" ? "文字 AI 分析" : "多模态 AI 分析";
  if (!aiAnalysis) {
    return (
      <Card title="多模态语义" icon={<Brain />}>
        <div className="ai-status">
          <InfoRow label="分析方式" value={modeText} />
          <InfoRow label="AI 提供商" value={activeProvider?.display_name || "disabled"} />
          <InfoRow label="状态" value={aiSettings?.enabled ? "等待分析" : "当前使用本地规则"} />
        </div>
        {!aiSettings?.enabled && <p className="note-line">多模态 API 未启用；不会发送图像或文字到外部 API。</p>}
      </Card>
    );
  }
  const tags = [
    aiAnalysis.scene && `场景：${aiAnalysis.scene}`,
    aiAnalysis.subscene && `子场景：${aiAnalysis.subscene}`,
    ...aiAnalysis.main_subjects.slice(0, 3).map((item) => `主体：${item}`),
    ...aiAnalysis.important_light_sources.slice(0, 3).map((item) => `光源：${item}`),
    ...aiAnalysis.critical_regions.slice(0, 3).map((item) => `高风险：${item}`),
  ].filter(Boolean);
  const suggestions = [
    ...(aiAnalysis.model_candidates || []).slice(0, 2).map((item) => ({
      id: item.model_id,
      title: modelName(item.model_id),
      meta: "模型建议",
      score: item.score,
      reason: item.reason,
    })),
    ...(aiAnalysis.checkpoint_candidates || []).slice(0, 3).map((item) => ({
      id: `${item.model_id}-${item.checkpoint_id}`,
      title: `${modelName(item.model_id)} / ${checkpointName(item.checkpoint_id)}`,
      meta: "权重建议",
      score: item.score,
      reason: item.reason,
    })),
  ];
  const conciseAdvice = compactAiAdvice(aiAnalysis);
  const validationRows: Array<[string, React.ReactNode, React.ReactNode?]> = [
    ["本地校验", aiAnalysis.validation_passed ? "通过" : "未通过", aiAnalysis.validation_passed ? "可作为建议" : "已回退"],
    ["最终采纳", aiAnalysis.adopted ? "采纳为有限加分依据" : "未采纳", aiAnalysis.adopted ? `最多 +${String(aiAnalysis.local_validation?.semantic_bonus_max ?? 20)} 分` : "本地规则优先"],
    ["未采纳原因", aiAnalysis.rejection_reason || aiAnalysis.failure_reason || "-", aiAnalysis.fallback_used ? "回退已生效" : undefined],
  ];
  return (
    <Card title="多模态语义" icon={<Brain />}>
      <div className="metric-grid">
        <Metric label="分析方式" value={modeText} />
        <Metric label="提供商/模型" value={`${aiAnalysis.provider} / ${aiAnalysis.model}`} />
        <Metric label="是否发图" value={aiAnalysis.sent_image ? "发送缩略预览" : "未发送图像"} />
        <Metric label="判断置信度" value={fmt(aiAnalysis.scene_confidence, 2)} />
      </div>
      <div className="tag-row">{tags.map((tag) => <span className="tag" key={tag}>{tag}</span>)}</div>
      <div className="ai-brief">
        <span>AI 精简建议</span>
        <b>{conciseAdvice}</b>
      </div>
      <div className="semantic-suggestions">
        {suggestions.length ? suggestions.map((item) => (
          <div className="suggestion-row" key={item.id}>
            <span>{item.meta}</span>
            <b>{item.title}</b>
            <em>{fmt(item.score, 1)}</em>
            <small>{item.reason || "来自多模态语义建议"}</small>
          </div>
        )) : <EmptyText text="暂无可展示的大模型建议。" />}
      </div>
      <div className="validation-grid">{validationRows.map(([label, value, tone]) => <InfoRow key={label} label={label} value={value} tone={tone} />)}</div>
      {!!aiAnalysis.validation_errors?.length && <p className="note-line">校验记录：{aiAnalysis.validation_errors.join(" / ")}</p>}
      <details className="ai-raw-summary">
        <summary>查看大模型原始摘要</summary>
        <p className="evidence">{aiAnalysis.reasoning_summary || "本地统计分析无法可靠识别复杂场景语义。"}</p>
      </details>
      {!!aiAnalysis.warnings?.length && <p className="note-line">{aiAnalysis.warnings.join(" / ")}</p>}
      <p className="note-line">多模态结果只作为受限建议展示，最终执行仍以本地模型注册、权重状态、硬件检查和路由规则为准。</p>
      {aiAnalysis.fallback_used && <span className="tag warn">已回退本地规则</span>}
    </Card>
  );
}

function compactAiAdvice(aiAnalysis: AIAnalysis) {
  const topModel = aiAnalysis.model_candidates?.[0];
  const topCheckpoint = aiAnalysis.checkpoint_candidates?.[0];
  const target = [
    topModel ? modelName(topModel.model_id) : "",
    topCheckpoint ? checkpointName(topCheckpoint.checkpoint_id) : "",
  ].filter(Boolean).join(" / ");
  const raw = (aiAnalysis.reasoning_summary || topModel?.reason || topCheckpoint?.reason || "建议结合本地评分选择候选结果。").replace(/\s+/g, " ").trim();
  const summary = raw.length > 120 ? `${raw.slice(0, 118)}...` : raw;
  return target ? `优先参考 ${target}；${summary}` : summary;
}

function PostprocessCard({ task, recommendation, busy, realesrganStatus, iqaStatus, onRefresh, onRun, onRunSr, onSkipSr, onRollback }: { task: Task | null; recommendation: PostprocessRecommendation | null; busy: boolean; realesrganStatus: RealESRGANStatus | null; iqaStatus: IQAStatus | null; onRefresh: () => void; onRun: (modelId: "lpdm" | "nafnet", checkpointId: string) => void; onRunSr: (scale?: number, allowX4?: boolean) => void; onSkipSr: () => void; onRollback: () => void }) {
  const ready = Boolean(task?.best_result?.output_file_id) && ["completed", "awaiting_denoise_confirmation", "awaiting_sr_confirmation"].includes(task?.status || "");
  const bestOperation = String(task?.best_result?.parameters?.operation || "");
  const alreadyPostprocessed = task?.best_result?.parameters?.role === "postprocess";
  const srReady = Boolean(realesrganStatus?.pytorch_backend?.available);
  const lpdmReady = false;
  const srPackages = realesrganStatus?.pytorch_backend?.packages || {};
  const srMissing = Object.entries(srPackages).filter(([, value]) => !value.available).map(([key]) => key).join(" / ");
  return (
    <Card title="后处理验证" icon={<Sparkles />}>
      <div className="postprocess-summary">
        <InfoRow label="去噪建议" value={recommendation?.preferred_denoiser ? modelName(recommendation.preferred_denoiser) : "NAFNet"} />
        <InfoRow label="去噪置信度" value={recommendation ? pct(recommendation.denoise_confidence) : "待计算"} />
        <InfoRow label="超分建议" value={recommendation?.preferred_sr_model && recommendation.preferred_sr_model !== "none" ? `${modelName(recommendation.preferred_sr_model)} x${recommendation.preferred_scale || 2}` : "暂不建议"} />
        <InfoRow label="Real-ESRGAN" value={srReady ? "可用" : "不可用"} tone={srReady ? "ok" : "warn"} />
        <InfoRow label="IQA" value={iqaStatus?.available ? "pyiqa 可用" : "本地指标回退"} tone={iqaStatus?.available ? "ok" : "warn"} />
        <InfoRow label="最终状态" value={alreadyPostprocessed ? (bestOperation === "super_resolution" ? "已采用超分结果" : "已采用去噪结果") : ready ? "等待用户确认" : "等待增强完成"} />
      </div>
      {recommendation?.denoise_reason?.length ? <div className="reason-list">{recommendation.denoise_reason.map((item) => <span key={item}>{item}</span>)}</div> : <p className="note-line">增强完成后会根据残余噪声、色偏和清晰度风险给出建议。</p>}
      {recommendation?.sr_reason?.length ? <div className="reason-list">{recommendation.sr_reason.map((item) => <span key={item}>{item}</span>)}</div> : null}
      {srMissing ? <p className="note-line">Real-ESRGAN 依赖缺失：{srMissing}，超分会保持禁用。</p> : null}
      {iqaStatus && !iqaStatus.available ? <p className="note-line">IQA 当前不可用：{iqaStatus.status_message}，将使用本地技术指标评分。</p> : null}
      <div className="action-row compact">
        <button onClick={() => onRun("lpdm", "lpdm_lol")} disabled={!lpdmReady || !ready || busy || alreadyPostprocessed} title="LPDM 在当前环境下全尺寸后处理不稳定，已暂停执行。">{busy ? "处理中" : "LPDM 暂停"}</button>
        <button onClick={() => onRun("nafnet", "sidd_width32")} disabled={!ready || busy || alreadyPostprocessed}>{busy ? "处理中" : "NAFNet 手动去噪"}</button>
        <button onClick={() => onRunSr(2, false)} disabled={!ready || busy || alreadyPostprocessed || !srReady}>{busy ? "处理中" : "Real-ESRGAN x2"}</button>
        <button onClick={() => onRunSr(4, true)} disabled={!ready || busy || alreadyPostprocessed || !srReady}>{busy ? "处理中" : "Real-ESRGAN x4"}</button>
        <button onClick={onSkipSr} disabled={!ready || busy}>跳过超分</button>
        <button onClick={onRefresh} disabled={!ready || busy}><RefreshCw />刷新建议</button>
        <button onClick={onRollback} disabled={!alreadyPostprocessed || busy}><RotateCcw />撤回后处理</button>
      </div>
      <p className="note-line">后处理结果会重新评分；如果分数下降或出现伪影，会自动回退到增强后的最佳结果。</p>
    </Card>
  );
}

function CandidatePlanCard({ task }: { task: Task | null }) {
  const items = task?.model_candidates || [];
  return (
    <Card title="候选规划" icon={<Layers />}>
      {items.length ? (
        <div className="v2-plan-list">
          {items.map((item, index) => {
            const modelId = String(item.model_id || "-");
            const checkpointId = String(item.checkpoint_id || "-");
            const reasons = Array.isArray(item.reason) ? item.reason : [String(item.reason || "")].filter(Boolean);
            return (
              <div className="v2-plan-item" key={`${modelId}-${checkpointId}-${index}`}>
                <div>
                  <span className="rank">{index + 1}</span>
                  <strong>{modelName(modelId)} / {checkpointName(checkpointId)}</strong>
                </div>
                <b>{scoreText(item.planning_score)} 规划分</b>
                <small>
                  {String(item.role || "enhancement")} · 权重匹配 {scoreText(item.checkpoint_score)}
                </small>
                <p>{reasons.join(" / ") || "由图像退化、用户需求、硬件资源和模型健康状态共同决定。"}</p>
              </div>
            );
          })}
        </div>
      ) : <EmptyText text="任务启动后显示 1-3 个互补增强候选。" />}
      <p className="note-line">规划分只决定哪些候选值得执行；最终推荐必须等待真实模型输出后的最终评分。</p>
    </Card>
  );
}

function CandidateResultsCard({ task }: { task: Task | null }) {
  const main = (task?.candidates || []).filter((item) => item.parameters?.role !== "postprocess");
  const bestId = task?.best_result?.output_file_id;
  return (
    <Card title="候选结果评分" icon={<BarChart3 />}>
      {main.length ? (
        <div className="v2-result-list">
          {main.map((item, index) => {
            const layers = (item.metrics?.score_layers || {}) as Record<string, unknown>;
            const region = (item.metrics?.region_constraints || null) as Record<string, unknown> | null;
            const finalScore = item.final_score ?? item.score;
            return (
              <div className={`v2-result-item ${bestId && item.output_file_id === bestId ? "selected" : ""}`} key={`${item.candidate_id || index}-${item.output_file_id || item.status}`}>
                <span className="rank">{index + 1}</span>
                <strong>{modelName(item.model_id)} / {checkpointName(item.checkpoint_id)}</strong>
                <b>{item.status === "completed" ? `${fmt(finalScore, 1)} 最终分` : "失败"}</b>
                <small>规划 {fmt(item.planning_score || Number(item.metrics?.planning_score || 0), 1)} · 画质 {fmt(Number(layers.image_quality || 0) * 100, 1)} · 恢复 {fmt(Number(layers.restoration || 0) * 100, 1)} · 约束 {fmt(Number(layers.constraint || 0) * 100, 1)} · 稳定 {fmt(Number(layers.stability || 0) * 100, 1)}</small>
                {region?.enabled ? <em className={region.hard_failed ? "roi-fail" : "roi-pass"}>ROI {region.hard_failed ? "未通过" : "通过"} · {fmt(Number(region.score || 0) * 100, 1)}</em> : null}
                {item.error && <p>{item.error}</p>}
              </div>
            );
          })}
        </div>
      ) : <EmptyText text="候选执行后显示真实输出评分。" />}
    </Card>
  );
}

function RouteCard({ title, items, selected, kind }: { title: string; items: Array<Record<string, unknown>>; selected?: string; kind: "model" | "weight" }) {
  return (
    <Card title={title} icon={<Layers />}>
      {items.length ? (
        <div className="route-list">
          {items.slice(0, 4).map((item, index) => {
            const id = String(item.model_id || item.checkpoint_id || item.id || index);
            return (
              <div className={`route-item ${selected === id ? "selected" : ""}`} key={id}>
                <span className="rank">{index + 1}</span>
                <strong>{kind === "model" ? modelName(id) : checkpointName(id)}</strong>
                <span>{scoreText(item.score)}</span>
                {selected === id && <em>已选择</em>}
                <small>{String(item.reason || item.status || item.domain || "候选方案")}</small>
              </div>
            );
          })}
        </div>
      ) : <EmptyText text="任务启动后显示路由排序。" />}
    </Card>
  );
}

function WeightRouteCard({ items, selected }: { items: Array<Record<string, unknown>>; selected?: string }) {
  return (
    <Card title="权重路由" icon={<Database />}>
      {items.length ? (
        <div className="weight-routes">
          {items.slice(0, 4).map((item, index) => {
            const id = String(item.checkpoint_id || item.id || index);
            return (
              <div className={`weight-route ${selected === id ? "selected" : ""}`} key={id}>
                <span>{index + 1}</span>
                <strong>{checkpointName(id)}</strong>
                <b>{scoreText(item.score)}</b>
                <em>{selected === id ? "首选" : index === 1 ? "备用" : "候选"}</em>
                <small>{String(item.domain || item.reason || "匹配当前图像域")}</small>
              </div>
            );
          })}
        </div>
      ) : <EmptyText text="选择模型后显示权重排序。" />}
    </Card>
  );
}

function TimelineCard({ task }: { task: Task | null }) {
  const steps = [
    ["分析输入图像", "analyzing"],
    ["解析用户需求", "parsing_intent"],
    ["检查GPU资源", "inspecting_hardware"],
    ["选择模型架构", "routing_model"],
    ["选择模型权重", "routing_checkpoint"],
    ["加载模型", "loading_model"],
    ["执行真实推理", "running"],
    ["评价增强结果", "evaluating"],
    ["检查备用策略", "fallback_running"],
    ["输出最佳结果", "completed"],
  ];
  const activeIndex = Math.max(0, steps.findIndex(([, state]) => state === task?.status));
  const done = task?.status === "completed" ? steps.length : activeIndex;
  return (
    <Card title="执行计划" icon={<Activity />}>
      <div className="timeline">
        {steps.map(([label, state], index) => {
          const status = task?.status === "failed" && index >= activeIndex ? "失败" : index < done ? "已完成" : index === activeIndex ? "进行中" : "等待";
          return <div className={`time-step ${status}`} key={state}><span>{index + 1}</span><b>{label}</b><em>{status}</em></div>;
        })}
      </div>
    </Card>
  );
}

function BottomPanel({ tab, setTab, task, preview }: { tab: string; setTab: (value: string) => void; task: Task | null; preview: string }) {
  const tabs = [
    ["candidates", "候选结果"],
    ["metrics", "质量指标"],
    ["logs", "Agent日志"],
    ["params", "运行参数"],
  ];
  return (
    <section className="panel bottom-panel">
      <div className="tabs">
        {tabs.map(([id, label]) => <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>{label}</button>)}
      </div>
      {tab === "candidates" && <CandidatesTab candidates={task?.candidates || []} />}
      {tab === "metrics" && <MetricsTab candidate={task?.best_result} />}
      {tab === "logs" && <LogsTab task={task} />}
      {tab === "params" && <ParamsTab task={task} preview={preview} />}
    </section>
  );
}

function CandidatesTab({ candidates }: { candidates: Candidate[] }) {
  if (!candidates.length) return <EmptyText text="任务运行后展示真实候选输出。" />;
  const postprocess = candidates.filter((item) => item.parameters?.role === "postprocess");
  const main = candidates.filter((item) => item.parameters?.role !== "postprocess");
  const renderCard = (item: Candidate) => (
    <article className={`candidate-card ${item.is_mock ? "mock" : ""}`} key={`${item.model_id}-${item.checkpoint_id}-${item.output_file_id}`}>
      {item.output_url ? <img src={item.output_url} /> : <div className="thumb-empty">{item.error || "未输出"}</div>}
      <div>
        <div className="candidate-head">
          <strong>{modelName(item.model_id)} / {checkpointName(item.checkpoint_id)}</strong>
          <span className={item.parameters?.role === "postprocess" ? "tag ok" : item.is_mock ? "tag warn" : "tag ok"}>{item.parameters?.role === "postprocess" ? "后处理" : item.is_mock ? "Mock结果" : "真实模型"}</span>
        </div>
        <div className="mini-metrics">
          <span>真实输出评分 <b>{fmt(item.score, 1)}</b></span>
          <span>推理时间 <b>{seconds(item.runtime_ms)}</b></span>
          <span>峰值显存 <b>{fmt(item.peak_memory_mb, 1)}MB</b></span>
          <span>设备 <b>{item.device || "-"}</b></span>
        </div>
        {item.output_url && <a className="button ghost" href={item.output_url} target="_blank">查看结果</a>}
      </div>
    </article>
  );
  return (
    <div className="candidate-sections">
      <div className="candidate-section">
        <div className="section-title-row"><h3>主增强候选</h3><span>参与自动评分和初始推荐。</span></div>
        <div className="candidate-grid">{main.map(renderCard)}</div>
      </div>
      <div className="candidate-section">
        <div className="section-title-row"><h3>后处理结果</h3><span>确认后由 NAFNet 基于推荐结果继续去噪。</span></div>
        {postprocess.length ? <div className="candidate-grid">{postprocess.map(renderCard)}</div> : <EmptyText text="暂无后处理结果。" />}
      </div>
    </div>
  );
}

function MetricsTab({ candidate }: { candidate?: Candidate }) {
  const metrics = candidate?.metrics || {};
  const rows = [
    ["平均亮度", "mean_luminance_before", "mean_luminance_after", ""],
    ["暗像素比例", "dark_pixel_ratio_before", "dark_pixel_ratio_after", "pct"],
    ["过曝比例", "overexposed_pixel_ratio_before", "overexposed_pixel_ratio_after", "pct"],
    ["RMS对比度", "rms_contrast_before", "rms_contrast_after", ""],
    ["动态范围", "dynamic_range_before", "dynamic_range_after", ""],
    ["清晰度", "laplacian_sharpness_before", "laplacian_sharpness_after", ""],
    ["噪声估计", "noise_estimate_before", "noise_estimate_after", ""],
    ["偏色指数", "color_cast_index_before", "color_cast_index_after", ""],
    ["信息熵", "image_entropy_before", "image_entropy_after", ""],
  ];
  if (!candidate) return <EmptyText text="推荐结果生成后显示前后质量指标。" />;
  return (
    <>
      <table className="quality-table">
        <thead><tr><th>指标</th><th>增强前</th><th>增强后</th><th>变化</th></tr></thead>
        <tbody>
          {rows.map(([label, beforeKey, afterKey, format]) => {
            const before = Number(metrics[beforeKey]);
            const after = Number(metrics[afterKey]);
            return <tr key={label}><td>{label}</td><td>{formatMetric(before, format)}</td><td>{formatMetric(after, format)}</td><td className={after >= before ? "trend-up" : "trend-down"}>{deltaText(after - before, format)}</td></tr>;
          })}
        </tbody>
      </table>
      <p className="note-line">{String(metrics.note || "无参考指标只能作为辅助判断，不能完全替代人工主观评价。")}</p>
    </>
  );
}

function LogsTab({ task }: { task: Task | null }) {
  const logs = task?.logs || [];
  if (!logs.length) return <EmptyText text="Agent 运行后显示步骤日志。" />;
  return (
    <div className="agent-log">
      {logs.map((line, index) => {
        const isError = line.includes("失败") || line.toLowerCase().includes("error");
        const isDetail = line.length > 180;
        return (
          <div className={`log-line ${isError ? "error" : ""}`} key={`${index}-${line.slice(0, 12)}`}>
            <Clock3 />
            <span>{String(index + 1).padStart(2, "0")}</span>
            {isDetail ? <details><summary>查看错误详情</summary><p>{line}</p></details> : <p>{line}</p>}
          </div>
        );
      })}
    </div>
  );
}

function ParamsTab({ task, preview }: { task: Task | null; preview: string }) {
  const best = task?.best_result;
  const plan = task?.plan;
  const rows = [
    ["模型", modelName(best?.model_id || plan?.selected_model)],
    ["权重", checkpointName(best?.checkpoint_id || plan?.selected_checkpoint)],
    ["设备", best?.device || String(plan?.parameters?.device || "-")],
    ["精度", best?.precision || String(plan?.parameters?.precision || "-")],
    ["输入尺寸", task?.analysis ? `${task.analysis.width}×${task.analysis.height}` : preview ? "已载入" : "-"],
    ["Padding", String(plan?.parameters?.padding || "按模型尺寸自动补边")],
    ["输出尺寸", task?.analysis && best ? `${task.analysis.width}×${task.analysis.height}` : "-"],
    ["tile_size", String(plan?.parameters?.tile_size || "-")],
    ["tile_overlap", String(plan?.parameters?.tile_overlap || "-")],
    ["最大重试", String(plan?.max_retries ?? "-")],
    ["策略", plan?.evaluation_strategy || "-"],
    ["备用模型/权重", `${(plan?.fallback_models || []).join(", ") || "-"} / ${(plan?.fallback_checkpoints || []).join(", ") || "-"}`],
  ];
  return <div className="param-grid">{rows.map(([label, value]) => <InfoRow key={label} label={label} value={value} />)}</div>;
}

function ModelsPage({ models, refresh }: { models: ModelStatus[]; refresh: () => void }) {
  const enhancementModels = models.filter((item) => !String(item.capabilities.task_type || "").includes("denoising") && !String(item.capabilities.task_type || "").includes("super_resolution"));
  const postprocessModels = models.filter((item) => String(item.capabilities.task_type || "").includes("denoising"));
  const superResolutionModels = models.filter((item) => String(item.capabilities.task_type || "").includes("super_resolution"));
  const groups = [
    { id: "enhancement", title: "增强模型", hint: "参与主流程候选增强、评分和最终结果选择。", items: enhancementModels },
    { id: "postprocess", title: "后处理去噪", hint: "用于增强后的真实图像去噪确认；NAFNet 不作为超分模型。", items: postprocessModels },
  ];

  return (
    <section className="page-panel" id="model-center">
      <div className="panel-title">
        <div>
          <h2>模型与权重</h2>
          <span className="mini-note">NAFNet 已接入后处理去噪，包含 SIDD width32 / width64 两套权重。</span>
        </div>
        <button onClick={refresh}><RefreshCw />重新扫描</button>
      </div>
      <div className="model-group-stack">
        {groups.map((group) => (
          <section className="model-section" key={group.id}>
            <div className="section-title-row">
              <h3>{group.title}</h3>
              <span>{group.hint}</span>
            </div>
            <div className="model-grid">
              {group.items.map((model) => (
                <article className={`model-card ${model.model_id === "nafnet" ? "featured-model" : ""}`} key={model.model_id}>
                  <div className="candidate-head">
                    <strong>{model.display_name}</strong>
                    <span className={model.available ? "tag ok" : "tag warn"}>{model.available ? "可用" : "不可用"}</span>
                  </div>
                  <p>{model.description}</p>
                  <small>{model.status_message}</small>
                  <div className="model-meta-row">
                    <span>{model.capabilities.task_type || "model"}</span>
                    <span>{model.capabilities.auto_route ? "自动路由" : "手动/确认"}</span>
                  </div>
                  <div className="weight-list">
                    {(model.capabilities.weights || []).map((weight) => (
                      <div className="weight-row" key={weight.checkpoint_id}>
                        <b>{weight.display_name}</b>
                        <span>{weight.status === "found" ? "已安装" : weight.status}</span>
                        <span>{weight.domain || "-"}</span>
                        <span>{fmtBytes(weight.size_bytes)}</span>
                      </div>
                    ))}
                  </div>
                </article>
              ))}
              {!group.items.length && <EmptyText text="暂无模型。" />}
            </div>
          </section>
        ))}
      </div>
    </section>
  );
}
function HistoryPage({ history }: { history: Task[] }) {
  return (
    <section className="page-panel">
      <div className="panel-title"><h2>历史任务</h2><span className="mini-note">最近 20 条</span></div>
      <div className="history-list">
        {history.slice(0, 20).map((item) => (
          <div className="history-row" key={item.task_id}>
            <span>{item.task_id.slice(0, 8)}</span>
            <b>{statusText[item.status] || item.status}</b>
            <em>{item.best_result ? `${modelName(item.best_result.model_id)} / ${checkpointName(item.best_result.checkpoint_id)}` : "-"}</em>
            <strong>{item.best_result ? fmt(item.best_result.score, 1) : "-"}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function SettingsPage({ system, models, aiSettings, aiProviders, refresh, setNotice }: { system: SystemInfo | null; models: ModelStatus[]; aiSettings: AISettings | null; aiProviders: AIProviderStatus[]; refresh: () => void; setNotice: (value: string) => void }) {
  const hw = system?.hardware || {};
  const settings = system?.settings || {};
  const [provider, setProvider] = useState(aiSettings?.provider || "disabled");
  const [enabled, setEnabled] = useState(Boolean(aiSettings?.enabled));
  const [model, setModel] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [sendImage, setSendImage] = useState(Boolean(aiSettings?.send_image));
  const [saving, setSaving] = useState(false);
  const [testingApi, setTestingApi] = useState(false);
  const [apiTestResult, setApiTestResult] = useState<AIProviderHealth | null>(null);
  useEffect(() => {
    setProvider(aiSettings?.provider || "disabled");
    setEnabled(Boolean(aiSettings?.enabled));
    setSendImage(Boolean(aiSettings?.send_image));
  }, [aiSettings?.provider, aiSettings?.enabled, aiSettings?.send_image]);
  const rows = [
    ["操作系统", String(hw.platform || "-")],
    ["API Python", String(hw.python || "-")],
    ["推理 Python", String(hw.inference_python || "-")],
    ["CPU", String(hw.cpu || "-")],
    ["GPU", String(hw.gpu_name || "未检测到")],
    ["CUDA", `${Boolean(hw.cuda_available) ? "可用" : "不可用"} / ${String(hw.cuda_version || "-")}`],
    ["总显存", hw.gpu_memory_mb ? `${hw.gpu_memory_mb} MB` : "-"],
    ["PyTorch", String(hw.torch || "-")],
    ["最大上传", `${String(settings.max_upload_mb || "-")} MB`],
    ["最大像素", String(settings.max_image_pixels || "-")],
    ["IntentParser", "本地规则"],
    ["LLM", "未启用"],
    ["可用模型", String(models.filter((item) => item.available).length)],
  ];
  const selectedStatus = aiProviders.find((item) => item.provider_id === provider);
  const saveAi = async () => {
    setSaving(true);
    setNotice("");
    try {
      await api<AISettings>(`/api/v1/ai/providers/${provider}/config`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          enabled,
          model,
          base_url: baseUrl,
          api_key: apiKey,
          send_image: sendImage,
          send_metrics: true,
          fallback_to_local: true,
        }),
      });
      setApiKey("");
      setNotice("多模态 API 配置已保存；后端已刷新本地配置。");
      refresh();
    } catch (err) {
      setNotice(String(err));
    } finally {
      setSaving(false);
    }
  };
  const testAiProvider = async () => {
    setTestingApi(true);
    setNotice("");
    setApiTestResult(null);
    try {
      const result = await api<AIProviderHealth>(`/api/v1/ai/providers/${provider}/health-check`, { method: "POST" });
      setApiTestResult(result);
      setNotice(result.healthy ? "API 检测通过。" : `API 检测未通过：${result.error_code || result.last_error || "未知错误"}`);
      refresh();
    } catch (err) {
      setApiTestResult({
        provider_id: provider,
        implemented: false,
        configured: false,
        healthy: false,
        supports_image: false,
        last_error: String(err),
        error_code: "HEALTH_CHECK_REQUEST_FAILED",
      });
      setNotice(String(err));
    } finally {
      setTestingApi(false);
    }
  };
  return (
    <section className="page-panel" id="system-settings">
      <div className="panel-title"><h2>系统设置</h2><span className="tag ok">本地 FastAPI</span></div>
      <div className="settings-grid">{rows.map(([label, value]) => <InfoRow key={label} label={label} value={value} />)}</div>
      <div className="settings-section">
        <div className="panel-title"><h2>多模态 API 配置</h2><span className={aiSettings?.enabled ? "tag ok" : "tag warn"}>{aiSettings?.enabled ? "已启用" : "纯本地默认"}</span></div>
        <div className="api-config-grid">
          <label><span>启用外部语义分析</span><select value={enabled ? "true" : "false"} onChange={(e) => setEnabled(e.target.value === "true")}><option value="false">关闭，使用纯本地规则</option><option value="true">开启，可选择供应商</option></select></label>
          <label><span>供应商</span><select value={provider} onChange={(e) => setProvider(e.target.value)}>{aiProviders.map((item) => <option key={item.provider_id} value={item.provider_id}>{item.display_name}</option>)}</select></label>
          <label><span>模型名</span><input value={model} placeholder={selectedStatus?.current_model || "例如 gpt-4o / qwen-vl / vendor-model"} onChange={(e) => setModel(e.target.value)} /></label>
          <label><span>Base URL</span><input value={baseUrl} placeholder={provider === "openai" ? "默认 https://api.openai.com/v1" : "兼容接口地址，例如 https://api.xxx.com/v1"} onChange={(e) => setBaseUrl(e.target.value)} disabled={provider === "anthropic" || provider === "gemini" || provider === "disabled"} /></label>
          <label><span>API Key</span><input value={apiKey} type="password" placeholder={selectedStatus?.configured ? "已配置；留空则不修改" : "仅写入本机 .env，不会回显"} onChange={(e) => setApiKey(e.target.value)} disabled={provider === "disabled"} /></label>
          <label><span>发送图像</span><select value={sendImage ? "true" : "false"} onChange={(e) => setSendImage(e.target.value === "true")}><option value="false">不发送图像，只发文字和指标</option><option value="true">发送去元数据缩略预览图</option></select></label>
        </div>
        <div className="provider-status-grid">
          <InfoRow label="当前状态" value={selectedStatus?.configured ? "已配置" : "未配置"} tone={selectedStatus?.healthy ? "健康" : selectedStatus?.error_code || "未检测"} />
          <InfoRow label="接口实现" value={selectedStatus?.implemented ? "已实现" : "预留接口"} />
          <InfoRow label="支持图像" value={selectedStatus?.supports_image ? "支持" : "不支持"} />
          <InfoRow label="当前模型" value={selectedStatus?.current_model || "未设置"} />
        </div>
        {apiTestResult && (
          <div className={`api-test-result ${apiTestResult.healthy ? "ok" : "warn"}`}>
            <div>
              <strong>{apiTestResult.healthy ? "接口检测通过" : "接口检测未通过"}</strong>
              <span>{apiTestResult.provider_id} / {apiTestResult.current_model || selectedStatus?.current_model || "未设置模型"}</span>
            </div>
            <p>{apiTestResult.healthy ? "当前 API Key、模型名和接口地址可以正常访问。" : apiTestResult.last_error || apiTestResult.error_code || "供应商返回了异常状态。"}</p>
            {apiTestResult.error_code && <em>{apiTestResult.error_code}</em>}
          </div>
        )}
        <p className="note-line">Key 只会写入本机 `.env`；前端和 API 响应不会读取或回显完整 Key。多模态模式只发送缩略预览图，原图仍只用于本地增强。</p>
        <div className="action-row compact">
          <button className="primary-action" onClick={saveAi} disabled={saving}>{saving ? "保存中" : "保存 API 配置"}</button>
          <button onClick={testAiProvider} disabled={testingApi || provider === "disabled"}>{testingApi ? "检测中" : "检测 API 接口"}</button>
        </div>
      </div>
    </section>
  );
}

function Card({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return <section className="decision-card"><h3>{icon}{title}</h3>{children}</section>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="metric"><span>{label}</span><b>{value}</b></div>;
}

function Histogram({ values }: { values: number[] }) {
  const max = Math.max(...values, 1);
  const points = values.map((value, index) => `${(index / Math.max(1, values.length - 1)) * 100},${38 - (value / max) * 36}`).join(" ");
  return (
    <svg className="histogram" viewBox="0 0 100 40" preserveAspectRatio="none">
      <polyline points={points} />
    </svg>
  );
}

function InfoRow({ label, value, tone }: { label: string; value: React.ReactNode; tone?: React.ReactNode }) {
  return <div className="info-row"><span>{label}</span><b>{value}</b>{tone && <em>{tone}</em>}</div>;
}

function EmptyText({ text }: { text: string }) {
  return <div className="empty-text">{text}</div>;
}

function TechDetails({ title, rows }: { title: string; rows: Array<[string, React.ReactNode]> }) {
  if (!rows.length) return null;
  return (
    <details className="tech-details">
      <summary>{title}</summary>
      <div className="kv-list">{rows.map(([label, value]) => <InfoRow key={label} label={label} value={value || "-"} />)}</div>
    </details>
  );
}

function technicalRows(best: Candidate | undefined, models: ModelStatus[]): Array<[string, React.ReactNode]> {
  if (!best) return [];
  return [
    ["adapter_class", best.adapter_class],
    ["model_id", best.model_id],
    ["checkpoint_id", best.checkpoint_id],
    ["checkpoint_path", best.checkpoint_path],
    ["checkpoint_sha256", best.checkpoint_sha256],
    ["input_sha256", best.input_sha256],
    ["output_sha256", best.output_sha256],
    ["已注册真实模型", models.filter((item) => item.available).map((item) => item.model_id).join(", ")],
  ];
}

function fmt(value: unknown, digits = 2) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : "-";
}

function pct(value: unknown) {
  const num = Number(value);
  return Number.isFinite(num) ? `${(num * 100).toFixed(2)}%` : "-";
}

function seconds(ms?: number) {
  return Number.isFinite(Number(ms)) ? `${(Number(ms) / 1000).toFixed(2)}s` : "-";
}

function level(value: number, cuts: [number, number], labels: [string, string, string]) {
  if (value < cuts[0]) return labels[0];
  if (value < cuts[1]) return labels[1];
  return labels[2];
}


function taskIsPausedOrDone(status: string) {
  return ["completed", "failed", "cancelled", "awaiting_denoise_confirmation", "awaiting_sr_confirmation"].includes(status);
}

function scoreText(value: unknown) {
  const num = Number(value);
  return Number.isFinite(num) ? `${num.toFixed(0)}分` : "-";
}

function formatMetric(value: number, mode: string) {
  if (!Number.isFinite(value)) return "-";
  return mode === "pct" ? pct(value) : fmt(value, 2);
}

function deltaText(value: number, mode: string) {
  if (!Number.isFinite(value)) return "-";
  const sign = value >= 0 ? "+" : "";
  return mode === "pct" ? `${sign}${(value * 100).toFixed(2)}%` : `${sign}${value.toFixed(2)}`;
}

function fmtBytes(n?: number) {
  if (!n) return "-";
  if (n > 1024 * 1024 * 1024) return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

function modelName(id?: string) {
  const names: Record<string, string> = { retinexformer: "Retinexformer", darkir: "DarkIR", hvi_cidnet: "HVI-CIDNet", flol: "FLOL", lpdm: "LPDM", nafnet: "NAFNet", realesrgan: "Real-ESRGAN", sci: "SCI", zero_dce: "Zero-DCE", mock_model: "MockModel" };
  return id ? names[id] || id : "-";
}

function checkpointName(id?: string) {
  const names: Record<string, string> = {
    lol_v2_real: "LOL-v2-real",
    sdsd_outdoor: "SDSD-outdoor",
    sdsd_indoor: "SDSD-indoor",
    medium: "medium",
    difficult: "difficult",
    epoch99: "Epoch99",
    real_lsrw: "real-LSRW",
    lol_blur: "LOLBlur",
    lol_blur_w64: "LOLBlur width64",
    all_lol: "All-LOL",
    sice: "SICE",
    fivek: "FiveK",
    sid: "SID",
    uhd_ll: "UHD-LL",
    mock_checkpoint: "Mock checkpoint",
    sidd_width32: "SIDD width32",
    sidd_width64: "SIDD width64",
  };
  return id ? names[id] || id : "-";
}

function sceneLabel(scene?: string) {
  const value = String(scene || "unknown");
  if (value.includes("outdoor")) return "室外";
  if (value.includes("indoor")) return "室内";
  if (value.includes("night")) return "夜景";
  return "场景自动";
}

createRoot(document.getElementById("root")!).render(<App />);

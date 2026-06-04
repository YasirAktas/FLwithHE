import React, { useState, useEffect } from 'react';
import { 
  Server, Laptop, Shield, Lock, Play, Square, ChevronDown, 
  ShieldCheck, Activity, Clock, Database, ArrowRight, ShieldAlert, Info
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip as RechartsTooltip, ResponsiveContainer, 
  BarChart, Bar, Cell
} from 'recharts';

// --- UI COMPONENTS ---
const Card = ({ children, className = "" }) => (
  <div className={`bg-[#1A202C] border border-[#2D3748] rounded-lg p-5 shadow-lg shadow-black/20 ${className}`}>
    {children}
  </div>
);

const Label = ({ children }) => (
  <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
    {children}
  </label>
);

const TradeoffBar = ({ label, level, value, colorClass }) => (
  <div className="mb-4 last:mb-0">
    <div className="flex justify-between text-xs mb-1.5">
      <span className="text-slate-400 font-medium">{label}</span>
      <span className={`font-semibold ${colorClass.text}`}>{level}</span>
    </div>
    <div className="h-1.5 w-full bg-[#0B0E14] rounded-full overflow-hidden border border-[#2D3748]">
      <div className={`h-full ${colorClass.bg} rounded-full transition-all duration-1000`} style={{ width: value }}></div>
    </div>
  </div>
);

export default function App() {
  // --- FORM STATE ---
  const [dataset, setDataset] = useState('MNIST');
  const [clients, setClients] = useState(10);
  const [rounds, setRounds] = useState(10);
  const [method, setMethod] = useState('Hybrid');
  const [dpMech, setDpMech] = useState('Gaussian Noise');
  const [epsilon, setEpsilon] = useState(2.45);
  const [scheme, setScheme] = useState('CKKS (Approximate)');

  // --- LIVE METRICS STATE ---
  const [isRunning, setIsRunning] = useState(false);
  const [statusText, setStatusText] = useState('System Idle');
  const [activeStep, setActiveStep] = useState(1);
  const [currentRound, setCurrentRound] = useState(0);
  
  const [metricsHistory, setMetricsHistory] = useState([]);
  const [latestMetrics, setLatestMetrics] = useState({
    accuracy: 0, loss: 0, total_round_time: 0, payload_nbytes: 0
  });

  // --- API POLLING LOGIC ---
  useEffect(() => {
    let interval;
    if (isRunning) {
      interval = setInterval(async () => {
        try {
          // 1. Fetch Status
          const statRes = await fetch('/api/status');
          const statData = await statRes.json();
          setIsRunning(statData.is_running);
          setStatusText(statData.current_action);

          // Map Backend string to visual stepper (1-5)
          if (statData.current_action.includes("Training")) setActiveStep(2);
          else if (statData.current_action.includes("Encrypting")) setActiveStep(4);
          else if (statData.current_action.includes("Aggregation")) setActiveStep(5);
          else setActiveStep(1);

          // 2. Fetch Metrics CSV Data
          const metRes = await fetch('/api/metrics');
          const metData = await metRes.json();
          
          if (metData.latest) {
            setLatestMetrics(metData.latest);
            setCurrentRound(metData.latest.round);
            
            // Map CSV data directly to Recharts format
            const formattedHistory = metData.history.map(row => ({
              round: row.round,
              acc: parseFloat((row.accuracy * 100).toFixed(2)),
              Train: parseFloat(row.training_time.toFixed(2)),
              Encrypt: parseFloat(row.encrypt_time.toFixed(2)),
              Aggregation: parseFloat(row.aggregate_time.toFixed(2)),
              PayloadMB: parseFloat((row.payload_nbytes / (1024 * 1024)).toFixed(2))
            }));
            setMetricsHistory(formattedHistory);
          }
        } catch (e) {
          console.error("Polling error", e);
        }
      }, 1500);
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  // --- START / STOP HANDLERS ---
  const handleStart = async () => {
    setMetricsHistory([]);
    setCurrentRound(0);
    setActiveStep(1);
    setIsRunning(true);
    setStatusText("Initializing FL Environment...");

    await fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset, num_clients: clients, rounds, method, dp_mechanism: dpMech, epsilon, scheme
      })
    });
  };

  const handleStop = async () => {
    await fetch('/api/stop', { method: 'POST' });
    setIsRunning(false);
    setStatusText("Experiment Aborted.");
    setActiveStep(1);
  };

  // Stepper Config
  const steps = [
    { id: 1, label: 'Server Dist.', icon: Server },
    { id: 2, label: 'Local Train', icon: Laptop },
    { id: 3, label: 'Add Noise', icon: Shield },
    { id: 4, label: 'Encrypt', icon: Lock },
    { id: 5, label: 'Secure Agg.', icon: Server },
  ];

  return (
    <div className="min-h-screen bg-[#0B0E14] text-slate-300 font-sans p-6 selection:bg-[#00D2FF] selection:text-black">
      
      {/* HEADER */}
      <header className="mb-8 flex justify-between items-center border-b border-[#2D3748] pb-4">
        <h1 className="text-2xl font-bold text-white tracking-widest flex items-center gap-3">
          <ShieldCheck size={28} className="text-[#00D2FF]" />
          FL+HE <span className="text-[#00D2FF] font-light">NEXUS</span>
        </h1>
        <div className="flex items-center gap-6 text-sm font-mono">
          <span className="flex items-center gap-2 text-[#00D2FF]">
            <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-400 animate-pulse shadow-[0_0_8px_rgba(74,222,128,0.8)]' : 'bg-slate-500'}`}></span> 
            {isRunning ? 'SYSTEM RUNNING' : 'SYSTEM IDLE'}
          </span>
          <span className="text-slate-400 bg-[#1A202C] px-3 py-1 rounded border border-[#2D3748]">
            ROUND: <span className="text-white font-bold">{currentRound}/{rounds}</span>
          </span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
        
        {/* LEFT SIDEBAR: CONTROLS */}
        <div className="lg:col-span-3 flex flex-col gap-5">
          <Card className="flex flex-col gap-6">
            <div>
              <Label>Dataset</Label>
              <select value={dataset} onChange={e => setDataset(e.target.value)} className="w-full bg-[#0B0E14] border border-[#2D3748] rounded px-3 py-2.5 text-sm text-white appearance-none outline-none focus:border-[#00D2FF]">
                <option>MNIST</option>
                <option>CIFAR10</option>
                <option>PTBXL</option>
              </select>
            </div>

            <div>
              <Label>Clients & Rounds</Label>
              <div className="flex gap-2">
                <input type="number" value={clients} onChange={e => setClients(e.target.value)} className="w-1/2 bg-[#0B0E14] border border-[#2D3748] rounded px-3 py-2 text-sm text-white" placeholder="Clients"/>
                <input type="number" value={rounds} onChange={e => setRounds(e.target.value)} className="w-1/2 bg-[#0B0E14] border border-[#2D3748] rounded px-3 py-2 text-sm text-white" placeholder="Rounds"/>
              </div>
            </div>

            <div>
              <Label>Privacy Method</Label>
              <div className="flex flex-col gap-3 bg-[#0B0E14] border border-[#2D3748] rounded p-3">
                {['Baseline', 'DP Only', 'HE Only', 'Hybrid'].map(opt => (
                  <label key={opt} className="flex items-center gap-3 text-sm cursor-pointer group">
                    <div className={`w-4 h-4 rounded-full border flex items-center justify-center transition-colors ${method === opt ? 'border-[#00D2FF]' : 'border-[#2D3748]'}`}>
                      {method === opt && <div className="w-2 h-2 rounded-full bg-[#00D2FF]" />}
                    </div>
                    <span className={method === opt ? 'text-white font-medium' : 'text-slate-400'}>{opt}</span>
                    <input type="radio" className="hidden" checked={method === opt} onChange={() => setMethod(opt)} />
                  </label>
                ))}
              </div>
            </div>

            <div className={`border-t border-[#2D3748] pt-5 ${(method === 'Baseline' || method === 'HE Only') ? 'opacity-30 pointer-events-none' : ''}`}>
              <Label>DP Configuration</Label>
              <select value={dpMech} onChange={e => setDpMech(e.target.value)} className="w-full bg-[#0B0E14] border border-[#2D3748] rounded px-3 py-2 text-sm text-white mb-4">
                <option>Gaussian Noise</option>
                <option>Laplace Noise</option>
              </select>
              <div className="bg-[#0B0E14] border border-[#2D3748] rounded px-3 py-2">
                <div className="flex justify-between text-xs text-slate-400 mb-2">
                  <span>Epsilon (ε)</span>
                  <span className="font-mono text-[#00D2FF]">{epsilon.toFixed(2)}</span>
                </div>
                <input type="range" min="0.1" max="10" step="0.05" value={epsilon} onChange={(e) => setEpsilon(parseFloat(e.target.value))} className="w-full accent-[#00D2FF] h-1 bg-[#2D3748] rounded-lg appearance-none cursor-pointer" />
              </div>
            </div>

            <div className={`border-t border-[#2D3748] pt-5 ${(method === 'Baseline' || method === 'DP Only') ? 'opacity-30 pointer-events-none' : ''}`}>
              <Label>HE Configuration</Label>
              <select value={scheme} onChange={e => setScheme(e.target.value)} className="w-full bg-[#0B0E14] border border-[#2D3748] rounded px-3 py-2 text-sm text-white">
                <option>CKKS (Approximate)</option>
                <option>Paillier (Partial)</option>
              </select>
            </div>

            <div className="pt-2">
              {!isRunning ? (
                <button onClick={handleStart} className="w-full bg-[#00D2FF] text-black font-bold py-3 rounded hover:bg-[#00b8e6] flex items-center justify-center gap-2 mb-3 shadow-[0_0_15px_rgba(0,210,255,0.2)]">
                  <Play size={16} className="fill-black" /> START EXPERIMENT
                </button>
              ) : (
                <button onClick={handleStop} className="w-full border border-red-900 text-red-500 bg-[#0B0E14] font-bold py-3 rounded hover:bg-red-950/40 hover:border-red-700 flex items-center justify-center gap-2 mb-3">
                  <Square size={14} className="fill-red-500" /> STOP RUNNING
                </button>
              )}
            </div>
          </Card>
        </div>

        {/* CENTER CONTENT: GRAPHS & FLOW */}
        <div className="lg:col-span-6 flex flex-col gap-6">
          
          <Card className="flex flex-col items-center py-8">
            <h2 className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] mb-8">Federated Learning Round Flow</h2>
            
            <div className="flex items-center justify-between relative w-full max-w-lg mb-8">
              <div className="absolute left-[10%] right-[10%] top-1/2 -translate-y-1/2 h-px bg-[#2D3748] -z-10" />
              {steps.map((step) => {
                const isActive = step.id === activeStep;
                const isPast = step.id < activeStep;
                const Icon = step.icon;
                
                return (
                  <div key={step.id} className="flex flex-col items-center gap-3 bg-[#1A202C] px-2 relative z-10">
                    <div className={`w-14 h-14 rounded-full flex items-center justify-center border-2 transition-all duration-500
                      ${isActive ? 'border-[#00D2FF] bg-[#00D2FF]/10 text-[#00D2FF] shadow-[0_0_20px_rgba(0,210,255,0.3)] scale-110' : 
                        isPast ? 'border-slate-600 bg-[#0B0E14] text-slate-400' : 'border-[#2D3748] bg-[#0B0E14] text-slate-600'}`}>
                      <Icon size={24} className={isActive ? 'animate-pulse' : ''} />
                    </div>
                    <span className={`text-[10px] font-bold uppercase tracking-wider ${isActive ? 'text-[#00D2FF]' : isPast ? 'text-slate-400' : 'text-slate-600'}`}>
                      {step.label}
                    </span>
                  </div>
                );
              })}
            </div>
            
            <p className="text-center text-xs text-slate-300 bg-[#0B0E14] px-5 py-2.5 rounded-full border border-[#2D3748] flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-[#00D2FF] animate-pulse' : 'bg-slate-500'}`}></span>
              <strong className="text-[#00D2FF] font-medium mr-1">Status:</strong> {statusText}
            </p>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 flex-1">
            
            {/* Global Accuracy Chart */}
            <Card className="flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Activity size={16} className="text-[#00D2FF]" />
                  <h3 className="text-sm font-semibold text-white">Global Accuracy</h3>
                </div>
                <span className="text-xs font-mono text-[#00D2FF]">{(latestMetrics.accuracy * 100).toFixed(2)}%</span>
              </div>
              <div className="flex-1 min-h-[160px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metricsHistory} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2D3748" vertical={false} />
                    <XAxis dataKey="round" stroke="#4A5568" fontSize={10} tickLine={false} axisLine={false} />
                    <YAxis stroke="#4A5568" fontSize={10} tickLine={false} axisLine={false} domain={[0, 100]} />
                    <RechartsTooltip contentStyle={{ backgroundColor: '#0B0E14', border: '1px solid #2D3748', borderRadius: '6px' }} />
                    <Line type="monotone" dataKey="acc" stroke="#00D2FF" strokeWidth={3} dot={false} activeDot={{ r: 5, fill: '#00D2FF' }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* Runtime Breakdown Chart */}
            <Card className="flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Clock size={16} className="text-orange-400" />
                  <h3 className="text-sm font-semibold text-white">Runtime Breakdown</h3>
                </div>
              </div>
              <div className="flex-1 min-h-[160px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={metricsHistory} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2D3748" vertical={false} />
                    <XAxis dataKey="round" stroke="#4A5568" fontSize={10} tickLine={false} axisLine={false} />
                    <YAxis stroke="#4A5568" fontSize={10} tickLine={false} axisLine={false} />
                    <RechartsTooltip cursor={{fill: '#2D3748', opacity: 0.3}} contentStyle={{ backgroundColor: '#0B0E14', border: '1px solid #2D3748', borderRadius: '6px' }} />
                    <Bar dataKey="Train" stackId="a" fill="#38B2AC" />
                    <Bar dataKey="Encrypt" stackId="a" fill="#00D2FF" />
                    <Bar dataKey="Aggregation" stackId="a" fill="#2B6CB0" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>

          </div>
        </div>

        {/* RIGHT SIDEBAR: SUMMARY */}
        <div className="lg:col-span-3 flex flex-col gap-5">
          <Card className="flex-1">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-5 flex items-center gap-2 border-b border-[#2D3748] pb-3">
              <Activity size={16} className="text-[#00D2FF]" /> Current Run Metrics
            </h3>
            
            <div className="space-y-4 mb-8">
              <div className="flex justify-between items-end border-b border-[#2D3748]/50 pb-2">
                <span className="text-xs text-slate-400 uppercase">Test Accuracy</span>
                <span className="font-mono text-lg text-white">{(latestMetrics.accuracy * 100).toFixed(2)}%</span>
              </div>
              <div className="flex justify-between items-end border-b border-[#2D3748]/50 pb-2">
                <span className="text-xs text-slate-400 uppercase">Latest Loss</span>
                <span className="font-mono text-lg text-white">{latestMetrics.loss ? latestMetrics.loss.toFixed(4) : "0.0000"}</span>
              </div>
              <div className="flex justify-between items-end border-b border-[#2D3748]/50 pb-2">
                <span className="text-xs text-slate-400 uppercase">Round Runtime</span>
                <span className="font-mono text-lg text-orange-400">{latestMetrics.total_round_time ? latestMetrics.total_round_time.toFixed(1) : 0}s</span>
              </div>
              <div className="flex justify-between items-end border-b border-[#2D3748]/50 pb-2">
                <span className="text-xs text-slate-400 uppercase">Payload Size</span>
                <span className="font-mono text-lg text-[#00D2FF]">{(latestMetrics.payload_nbytes / (1024 * 1024)).toFixed(2)} MB</span>
              </div>
            </div>

            <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-5 flex items-center gap-2 border-b border-[#2D3748] pb-3">
              <ShieldAlert size={16} className="text-orange-400" /> Active Configuration
            </h3>
            
            <div className="bg-[#0B0E14] border border-[#2D3748] p-4 rounded-lg text-xs text-slate-400 space-y-2 relative overflow-hidden shadow-inner">
              <div className="absolute left-0 top-0 bottom-0 w-1 bg-[#00D2FF] opacity-80"></div>
              <ul className="list-disc pl-4 space-y-2 marker:text-slate-600">
                <li><strong className="text-slate-300">Methodology:</strong> {method}</li>
                {(method === 'DP Only' || method === 'Hybrid') && (
                  <li><strong className="text-slate-300">Privacy:</strong> {dpMech} (ε={epsilon})</li>
                )}
                {(method === 'HE Only' || method === 'Hybrid') && (
                  <li><strong className="text-slate-300">Security:</strong> {scheme}</li>
                )}
              </ul>
            </div>
          </Card>
        </div>

      </div>
    </div>
  );
}
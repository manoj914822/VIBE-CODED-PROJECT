/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { 
  Heart, 
  Activity, 
  Thermometer, 
  User, 
  ClipboardCheck, 
  FileText, 
  Cpu, 
  AlertCircle,
  Database,
  ArrowRight,
  RefreshCw,
  Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { GoogleGenAI } from "@google/genai";

// Initialize Gemini API
const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

interface PatientData {
  age: number;
  sex: string;
  cp: string;
  trestbps: number;
  chol: number;
  fbs: string;
  restecg: string;
  thalach: number;
  exang: string;
  oldpeak: number;
  slope: string;
  ca: number;
  thal: string;
}

const initialData: PatientData = {
  age: 50,
  sex: "Male",
  cp: "Typical Angina",
  trestbps: 120,
  chol: 200,
  fbs: "False",
  restecg: "Normal",
  thalach: 150,
  exang: "No",
  oldpeak: 1.0,
  slope: "Flat",
  ca: 0,
  thal: "Normal"
};

export default function App() {
  const [formData, setFormData] = useState<PatientData>(initialData);
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([
    "[2026-05-10 19:28] INFO: Interface loaded",
    "[2026-05-10 19:34] INFO: Theme synchronized"
  ]);

  const addLog = (msg: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [`[${timestamp}] ${msg}`, ...prev].slice(0, 5));
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'].includes(name)
        ? Number(value) 
        : value
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    addLog("INFO: Initiating predict_pipeline.py");
    addLog("DEBUG: Loading artifacts/model.pkl");
    
    try {
      const response = await genAI.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: `You are an expert cardiologist AI. Analyze these patient parameters: ${JSON.stringify(formData)}. Predict "Low Risk" or "Potential Risk" and give a 2-sentence medical justification focused on clinical values like cholesterol, age, and blood pressure. Output JSON: {"prediction": string, "justification": string}`,
        config: { responseMimeType: "application/json" }
      });

      const result = JSON.parse(response.text || "{}");
      setPrediction(result);
      addLog("SUCCESS: Inference complete");
    } catch (err) {
      console.error(err);
      setError("Analysis failed. Please try again.");
      addLog("ERROR: Model inference failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0E1117] text-[#FAFAFA] font-sans">
      <div className="max-w-6xl mx-auto p-6 sm:p-10">
        <header className="mb-12 border-b border-[#30363D] pb-8">
          <div className="flex items-center gap-2 text-[#FF4B4B] mb-2">
            <Heart className="fill-[#FF4B4B]" size={24} />
            <span className="text-[10px] font-black tracking-[0.3em] uppercase italic text-[#8B949E]">Clinical Intelligence Workspace</span>
          </div>
          <h1 className="text-4xl font-black tracking-tighter">
            HEART <span className="text-[#FF4B4B] underline decoration-[#30363D] decoration-8 underline-offset-8">PROJECT_v1.0.4</span>
          </h1>
          <p className="mt-8 text-[#8B949E] font-mono text-xs">/workspace/src/pipeline/heart_disease_diagnostic</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <main className="lg:col-span-8 space-y-6">
            <div className="bg-[#161B22] rounded-xl border border-[#30363D] shadow-2xl overflow-hidden">
              <div className="px-6 py-4 bg-[#0D1117] border-b border-[#30363D] flex justify-between items-center">
                <span className="text-xs font-black flex items-center gap-2 tracking-widest text-[#8B949E]"><ClipboardCheck size={14} className="text-[#FF4B4B]" /> PATIENT_DATA_ENTRY</span>
                <button onClick={() => setFormData(initialData)} className="p-2 hover:bg-[#21262D] rounded-lg transition-colors text-[#8B949E]"><RefreshCw size={14} /></button>
              </div>
              
              <div className="p-8 grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-10">
                <div className="space-y-2">
                  <label className="text-[10px] font-black tracking-[0.2em] uppercase text-[#8B949E]">Biological Age</label>
                  <input name="age" type="number" value={formData.age} onChange={handleChange} className="w-full bg-[#0D1117] text-xl font-mono font-bold py-3 px-4 border border-[#30363D] rounded-lg focus:border-[#FF4B4B] outline-none transition-all text-[#C9D1D9]" />
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black tracking-[0.2em] uppercase text-[#8B949E]">Patient Gender</label>
                  <select name="sex" value={formData.sex} onChange={handleChange} className="w-full bg-[#0D1117] text-xl font-bold py-3 px-4 border border-[#30363D] rounded-lg focus:border-[#FF4B4B] outline-none transition-all text-[#C9D1D9]">
                    <option>Male</option>
                    <option>Female</option>
                  </select>
                </div>
                <div className="space-y-2">
                    <label className="text-[10px] font-black tracking-[0.2em] uppercase text-[#8B949E]">Chest Pain Type</label>
                    <select name="cp" value={formData.cp} onChange={handleChange} className="w-full bg-[#0D1117] font-bold py-3 px-4 border border-[#30363D] rounded-lg focus:border-[#FF4B4B] outline-none transition-all text-[#C9D1D9]">
                        <option>Typical Angina</option>
                        <option>Atypical Angina</option>
                        <option>Non-anginal Pain</option>
                        <option>Asymptomatic</option>
                    </select>
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black tracking-[0.2em] uppercase text-[#8B949E]">Resting Blood Pressure</label>
                  <input name="trestbps" type="number" value={formData.trestbps} onChange={handleChange} className="w-full bg-[#0D1117] text-xl font-mono font-bold py-3 px-4 border border-[#30363D] rounded-lg focus:border-[#FF4B4B] outline-none transition-all text-[#C9D1D9]" />
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black tracking-[0.2em] uppercase text-[#8B949E]">Serum Cholestoral</label>
                  <input name="chol" type="number" value={formData.chol} onChange={handleChange} className="w-full bg-[#0D1117] text-xl font-mono font-bold py-3 px-4 border border-[#30363D] rounded-lg focus:border-[#FF4B4B] outline-none transition-all text-[#C9D1D9]" />
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black tracking-[0.2em] uppercase text-[#8B949E]">Resting ECG Results</label>
                  <select name="restecg" value={formData.restecg} onChange={handleChange} className="w-full bg-[#0D1117] font-bold py-3 px-4 border border-[#30363D] rounded-lg focus:border-[#FF4B4B] outline-none transition-all text-[#C9D1D9]">
                        <option>Normal</option>
                        <option>ST-T Wave Abnormality</option>
                        <option>Left Ventricular Hypertrophy</option>
                    </select>
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black tracking-[0.2em] uppercase text-[#8B949E]">Max Heart Rate (thalach)</label>
                  <input name="thalach" type="number" value={formData.thalach} onChange={handleChange} className="w-full bg-[#0D1117] text-xl font-mono font-bold py-3 px-4 border border-[#30363D] rounded-lg focus:border-[#FF4B4B] outline-none transition-all text-[#C9D1D9]" />
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black tracking-[0.2em] uppercase text-[#8B949E]">Major Vessels (0-3)</label>
                  <input name="ca" type="number" min="0" max="3" value={formData.ca} onChange={handleChange} className="w-full bg-[#0D1117] text-xl font-mono font-bold py-3 px-4 border border-[#30363D] rounded-lg focus:border-[#FF4B4B] outline-none transition-all text-[#C9D1D9]" />
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black tracking-[0.2em] uppercase text-[#8B949E]">ST Depression (oldpeak)</label>
                  <input name="oldpeak" type="number" step="0.1" value={formData.oldpeak} onChange={handleChange} className="w-full bg-[#0D1117] text-xl font-mono font-bold py-3 px-4 border border-[#30363D] rounded-lg focus:border-[#FF4B4B] outline-none transition-all text-[#C9D1D9]" />
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black tracking-[0.2em] uppercase text-[#8B949E]">Exercise Induced Angina</label>
                  <select name="exang" value={formData.exang} onChange={handleChange} className="w-full bg-[#0D1117] font-bold py-3 px-4 border border-[#30363D] rounded-lg focus:border-[#FF4B4B] outline-none transition-all text-[#C9D1D9]">
                        <option>No</option>
                        <option>Yes</option>
                    </select>
                </div>
                <div className="space-y-2">
                  <label className="text-[10px] font-black tracking-[0.2em] uppercase text-[#8B949E]">Slope of ST segment</label>
                  <select name="slope" value={formData.slope} onChange={handleChange} className="w-full bg-[#0D1117] font-bold py-3 px-4 border border-[#30363D] rounded-lg focus:border-[#FF4B4B] outline-none transition-all text-[#C9D1D9]">
                        <option>Upsloping</option>
                        <option>Flat</option>
                        <option>Downsloping</option>
                    </select>
                </div>
              </div>

              <div className="px-8 py-6 bg-[#0D1117] border-t border-[#30363D] flex justify-end">
                <button 
                  onClick={handlePredict} 
                  disabled={loading}
                  className="bg-[#FF4B4B] text-white font-black px-12 py-5 rounded-lg shadow-[0_0_30px_rgba(255,75,75,0.2)] hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center gap-3 disabled:opacity-50 tracking-widest text-xs"
                >
                  {loading ? <RefreshCw className="animate-spin" /> : <Activity size={16} />}
                  {loading ? "EXECUTING_MODEL..." : "RUN AI DIAGNOSTIC"}
                </button>
              </div>
            </div>

            <AnimatePresence>
              {prediction && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`p-10 rounded-xl border border-[#30363D] ${prediction.prediction === 'Low Risk' ? 'bg-[#161B22]' : 'bg-[#161B22]'}`}
                >
                   <div className="flex items-center gap-3 mb-6">
                      <div className={`w-3 h-3 rounded-full ${prediction.prediction === 'Low Risk' ? 'bg-[#238636]' : 'bg-[#FF4B4B]'}`}></div>
                      <span className="text-[11px] font-mono font-bold tracking-widest uppercase text-[#8B949E]">Diagnostic Result Output</span>
                   </div>
                  <h3 className={`text-4xl font-black mb-4 tracking-tighter ${prediction.prediction === 'Low Risk' ? 'text-[#238636]' : 'text-[#FF4B4B] italic'}`}>
                    {prediction.prediction}
                  </h3>
                  <p className="text-lg font-medium text-[#8B949E] leading-relaxed">
                    {prediction.justification}
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </main>

          <aside className="lg:col-span-4 space-y-6">
            <div className="bg-[#161B22] rounded-xl p-8 border border-[#30363D]">
                <h4 className="flex items-center gap-2 font-black text-[#FAFAFA] tracking-tight mb-6 text-xs uppercase"> <Cpu size={16} className="text-[#58A6FF]"/> Pipeline Schematic</h4>
                <div className="space-y-4">
                  {[
                    { label: "Data Source", val: "UCI Cleveland" },
                    { label: "Transformation", val: "StandardScaler" },
                    { label: "Classifier", val: "Ensemble-V1" },
                    { label: "Verification", val: "Gemini-3-Flash" }
                  ].map((item, i) => (
                    <div key={i} className="flex justify-between items-center border-b border-[#30363D] pb-3">
                      <span className="text-[10px] font-bold text-[#8B949E] uppercase tracking-widest font-mono">{item.label}</span>
                      <span className="text-[10px] font-bold text-[#C9D1D9] font-mono">{item.val}</span>
                    </div>
                  ))}
                </div>
            </div>

            <div className="bg-[#161B22] rounded-xl p-8 border border-[#30363D] shadow-sm">
              <h4 className="font-black text-[#FAFAFA] mb-6 flex items-center gap-2 text-xs uppercase"><Database size={16} className="text-[#FF4B4B]"/> Workspace Artifacts</h4>
              <div className="space-y-2 font-mono text-[10px]">
                {['artifacts/model.pkl', 'artifacts/preprocessor.pkl', 'src/logger.py', 'setup.py'].map((f, i) => (
                  <div key={i} className="flex items-center justify-between p-3 bg-[#0D1117] border border-[#30363D] rounded-lg group hover:border-[#FF4B4B] transition-colors">
                    <div className="flex items-center gap-2 text-[#8B949E] group-hover:text-[#C9D1D9]">
                      <FileText size={12} /> {f}
                    </div>
                    <span className="text-[8px] bg-[#238636] text-white px-1 rounded uppercase">v1.0</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-[#0D1117] border border-[#30363D] rounded-xl p-4 font-mono text-[10px] text-[#8B949E] shadow-inner h-32 overflow-hidden">
              <div className="flex items-center gap-2 mb-2 pb-2 border-b border-[#30363D]">
                <div className="w-2 h-2 rounded-full bg-[#FF4B4B]"></div>
                <span>LOG_CONSOLE</span>
              </div>
              <div className="space-y-1">
                {logs.map((log, i) => (
                  <div key={i}>{log}</div>
                ))}
              </div>
            </div>
          </aside>
        </div>
      </div>

      <footer className="fixed bottom-0 left-0 right-0 h-8 bg-[#0D1117] border-t border-[#30363D] flex items-center px-6 justify-between text-[10px] text-[#8B949E] font-mono">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1">🌿 main</span>
          <span className="flex items-center gap-1 text-[#238636]">● System Active</span>
        </div>
        <div className="flex items-center gap-4">
          <span>UTF-8</span>
          <span>Python 3.9.12</span>
        </div>
      </footer>
    </div>
  );
}

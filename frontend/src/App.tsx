import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Battery, Bluetooth, Camera, Headphones, ShieldAlert, Mic, Activity, Navigation, Settings } from 'lucide-react';

export default function App() {
  const [status, setStatus] = useState<any>(null);
  const [connected, setConnected] = useState(false);
  const [videoUrl, setVideoUrl] = useState('');
  
  // Guardian Config State
  const [showGuardianConfig, setShowGuardianConfig] = useState(false);
  const [guardianName, setGuardianName] = useState('');
  const [guardianPhone, setGuardianPhone] = useState('');
  const [isCalling, setIsCalling] = useState(false);

  // Polling API for status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        setStatus(data);
      } catch (err) {
        console.error("API Error", err);
      }
    };
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleConnect = async () => {
    try {
      await fetch('/api/connect', { method: 'POST' });
      setConnected(true);
      setVideoUrl('http://127.0.0.1:5000/video_feed'); // Direct connection to avoid proxy streaming issues
    } catch (err) {
      console.error(err);
    }
  };

  const handleSOS = async () => {
    setIsCalling(true);
    try {
      await fetch('/api/sos', { method: 'POST' });
      // Keep calling overlay up for a bit
      setTimeout(() => setIsCalling(false), 5000);
    } catch (err) {
      console.error(err);
      setIsCalling(false);
    }
  };

  const handleSaveGuardian = async () => {
    try {
      await fetch('/api/guardian', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: guardianName, phone: guardianPhone })
      });
      setShowGuardianConfig(false);
      alert("Guardian configured successfully!");
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="min-h-screen flex flex-col p-4 md:p-6 gap-6">
      {/* TopNavBar */}
      <header className="fixed top-0 left-0 w-full z-50 bg-background/80 backdrop-blur-xl border-b border-border-glass shadow-sm flex justify-between items-center px-8 py-4">
        <div className="flex items-center gap-4">
          <span className="text-2xl font-bold tracking-tighter text-primary">NAVI</span>
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20">
            <div className={`w-2 h-2 rounded-full ${connected ? 'bg-primary' : 'bg-gray-500'}`}></div>
            <span className="text-xs font-semibold text-gray-500">{connected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
        <nav className="hidden md:flex gap-8">
          <Link className="text-primary border-b-2 border-primary pb-1 text-sm font-semibold" to="/">Dashboard</Link>
          <a className="text-gray-400 font-medium text-sm hover:text-primary transition-colors" href="#">Vision</a>
          <Link className="text-gray-400 font-medium text-sm hover:text-primary transition-colors" to="/guardian">Guardian</Link>
        </nav>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-gray-400">
            <Battery size={20} />
            <span className="text-sm font-semibold">85%</span>
          </div>
          <button onClick={() => setShowGuardianConfig(true)} className="hover:text-primary transition-colors text-gray-400">
            <Settings size={20} />
          </button>
        </div>
      </header>

      {/* Main Content Grid */}
      <main className="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-20 flex-grow">
        
        {/* Left Column */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          <section className="glass-card flex-grow overflow-hidden relative flex flex-col min-h-[400px]">
            <div className="p-4 border-b border-border-glass flex justify-between items-center bg-white/5">
              <h2 className="text-lg font-bold text-primary">System Info</h2>
            </div>
            <div className="p-4 flex flex-col gap-4 text-sm text-gray-300 flex-grow">
              <p><strong>Device:</strong> NAVI Glasses v1.0 Elite</p>
              <p><strong>Safe Direction:</strong> {status?.safe_direction || '--'}</p>
              <p><strong>Nearest Obstacle:</strong> {status?.nearest_object || 'None'} ({status?.nearest_distance || 0}m)</p>
              <p><strong>Danger Level:</strong> {status?.danger_level || 'Safe'}</p>
            </div>
            <div className="p-4 border-t border-white/10">
              <button 
                onClick={() => setShowGuardianConfig(true)}
                className="w-full py-2 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/20 rounded-lg font-bold text-sm transition-colors flex items-center justify-center gap-2"
              >
                <Settings size={16} />
                Configure Emergency Contact
              </button>
            </div>
          </section>

          <section className="glass-card p-4 glow-blue">
            <h3 className="text-xs font-bold text-secondary mb-4 uppercase tracking-widest">Device Connection</h3>
            <div className="flex items-center gap-4 p-3 rounded-lg bg-white/5 border border-white/10">
              <div className="w-10 h-10 rounded-full bg-secondary/20 flex items-center justify-center">
                <Headphones className="text-secondary" size={20} />
              </div>
              <div className="flex-grow">
                <p className="text-sm font-bold">Spatial Audio Buds</p>
                <p className="text-xs text-gray-400">Active • Low Latency</p>
              </div>
              <Bluetooth className="text-primary" size={16} />
            </div>
          </section>
        </div>

        {/* Center Column: Live Video Feed */}
        <div className="lg:col-span-6 flex flex-col gap-6">
          <section className="glass-card flex-grow overflow-hidden relative min-h-[400px] flex items-center justify-center bg-black">
            {!connected ? (
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleConnect}
                className="px-8 py-4 bg-primary text-black font-bold rounded-full flex items-center gap-4 glow-emerald shadow-2xl"
              >
                <Camera size={28} />
                CONNECT CAMERA
              </motion.button>
            ) : (
              <>
                <img src={videoUrl} alt="Live Video Feed" className="absolute inset-0 w-full h-full object-cover" />
                <div className="scan-line pointer-events-none"></div>
                <div className="absolute top-4 right-4 glass-card px-4 py-2 bg-black/60 text-primary font-bold text-xl flex items-center gap-2">
                  <span className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></span>
                  LIVE AI
                </div>
              </>
            )}
          </section>
        </div>

        {/* Right Column: Assistant & Safety */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          <section className="glass-card p-6 flex flex-col gap-4 glow-emerald h-1/2">
            <div className="flex items-center justify-between border-b border-border-glass pb-2">
              <h3 className="text-lg font-bold text-primary">Scene Understanding</h3>
              <Activity className="text-primary" size={20} />
            </div>
            <p className="text-sm font-bold">Scene: <span className="font-normal text-gray-400">{status?.scene_status || '--'}</span></p>
            <p className="text-sm font-bold">Objects: <span className="font-normal text-gray-400">{status?.obstacle_count || 0}</span></p>
            <div className="mt-2 p-3 rounded-lg bg-white/5 text-sm italic text-gray-400">
              "{status?.scene_understanding?.summary || 'Waiting for AI processing...'}"
            </div>
          </section>

          <section className="glass-card p-6 flex flex-col gap-4 h-1/2">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-bold text-secondary">Voice Assistant</h3>
              <Mic className="text-secondary" size={20} />
            </div>
            <div className="flex-grow flex flex-col items-center justify-center gap-4 py-8">
              <div className="flex items-end gap-1 h-12">
                <div className="w-1 bg-secondary rounded-full waveform-bar" style={{animationDelay: '0.1s'}}></div>
                <div className="w-1 bg-secondary rounded-full waveform-bar" style={{animationDelay: '0.3s'}}></div>
                <div className="w-1 bg-secondary rounded-full waveform-bar" style={{animationDelay: '0.2s'}}></div>
                <div className="w-1 bg-secondary rounded-full waveform-bar" style={{animationDelay: '0.5s'}}></div>
              </div>
              <p className="text-center text-sm font-medium text-gray-300">
                Listening...
              </p>
            </div>
          </section>
        </div>
      </main>

      {/* Emergency SOS Button */}
      <motion.button 
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        onClick={handleSOS}
        disabled={isCalling}
        className="fixed bottom-8 right-8 w-20 h-20 bg-danger rounded-full flex flex-col items-center justify-center text-white font-bold z-[100] glow-red shadow-2xl disabled:opacity-50"
      >
        <ShieldAlert size={32} />
        <span className="text-[10px] font-bold mt-1">EMERGENCY</span>
      </motion.button>

      {/* Calling Guardian Overlay */}
      {isCalling && (
        <div className="fixed inset-0 bg-red-950/90 backdrop-blur-md z-[150] flex flex-col items-center justify-center p-4">
          <motion.div 
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="flex flex-col items-center gap-6"
          >
            <div className="w-32 h-32 bg-red-500/20 rounded-full flex items-center justify-center relative">
              <div className="absolute inset-0 border-4 border-red-500 rounded-full animate-ping opacity-50"></div>
              <ShieldAlert size={64} className="text-red-500" />
            </div>
            <h2 className="text-4xl font-bold text-white tracking-widest text-center">SOS INITIATED</h2>
            <p className="text-xl text-red-300 font-mono animate-pulse text-center">
              Dialing configured Guardian contact...<br/>
              Broadcasting location & live feed...
            </p>
          </motion.div>
        </div>
      )}

      {/* Guardian Config Modal */}
      {showGuardianConfig && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[200] flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl p-6 w-full max-w-md shadow-2xl">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <ShieldAlert className="text-emerald-500" />
              Configure Guardian
            </h2>
            <p className="text-slate-400 text-sm mb-6">Set up the emergency contact who will receive automated calls and location access when an SOS is triggered.</p>
            
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Guardian Name</label>
                <input 
                  type="text" 
                  value={guardianName}
                  onChange={(e) => setGuardianName(e.target.value)}
                  placeholder="e.g. John Doe"
                  className="w-full bg-black/50 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-emerald-500 transition-colors"
                />
              </div>
              <div>
                <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Phone Number</label>
                <input 
                  type="tel" 
                  value={guardianPhone}
                  onChange={(e) => setGuardianPhone(e.target.value)}
                  placeholder="+1 (555) 000-0000"
                  className="w-full bg-black/50 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-emerald-500 transition-colors"
                />
              </div>
            </div>

            <div className="flex gap-4 mt-8">
              <button 
                onClick={() => setShowGuardianConfig(false)}
                className="flex-1 px-4 py-3 rounded-lg font-bold text-slate-300 bg-slate-800 hover:bg-slate-700 transition-colors"
              >
                Cancel
              </button>
              <button 
                onClick={handleSaveGuardian}
                className="flex-1 px-4 py-3 rounded-lg font-bold text-black bg-emerald-500 hover:bg-emerald-400 transition-colors shadow-lg shadow-emerald-500/20"
              >
                Save Contact
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

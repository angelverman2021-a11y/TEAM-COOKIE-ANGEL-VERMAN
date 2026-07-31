import { useState, useEffect, useRef } from 'react';
import { ShieldAlert, CheckCircle, Video, Activity, PhoneCall } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function Guardian() {
  const [emergencyMode, setEmergencyMode] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<string>('');
  const [sceneStatus, setSceneStatus] = useState<string>('');
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Poll for emergency status
  useEffect(() => {
    const pollStatus = async () => {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        
        if (data.emergency_mode !== emergencyMode) {
          setEmergencyMode(data.emergency_mode);
          if (data.emergency_mode) {
            // Play alarm sound if an emergency just started
            if (audioRef.current) {
              audioRef.current.play().catch(e => console.log('Audio autoplay blocked', e));
            }
          }
        }
        
        setSceneStatus(data.scene_status || 'Unknown');
        setLastUpdate(new Date().toLocaleTimeString());
      } catch (err) {
        console.error('Failed to poll status', err);
      }
    };

    const interval = setInterval(pollStatus, 1000);
    return () => clearInterval(interval);
  }, [emergencyMode]);

  const handleResolve = async () => {
    try {
      await fetch('/api/resolve_emergency', { method: 'POST' });
      setEmergencyMode(false);
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }
    } catch (err) {
      console.error('Failed to resolve', err);
    }
  };

  return (
    <div className={`min-h-screen transition-colors duration-1000 ${emergencyMode ? 'bg-red-950' : 'bg-slate-900'} text-white p-8 font-sans`}>
      {/* Hidden alarm audio */}
      <audio ref={audioRef} loop>
        <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg" />
      </audio>

      <div className="max-w-6xl mx-auto space-y-8">
        <header className="flex items-center justify-between border-b border-white/10 pb-6">
          <div className="flex items-center gap-4">
            <ShieldAlert size={40} className={emergencyMode ? "text-red-500 animate-pulse" : "text-emerald-500"} />
            <div>
              <h1 className="text-3xl font-bold tracking-tight">NAVI Guardian Portal</h1>
              <p className="text-slate-400">Remote Caretaker Access Terminal</p>
            </div>
          </div>
          <div className="text-right">
            <div className="flex items-center gap-2 justify-end">
              <Activity size={16} className="text-emerald-400" />
              <span className="text-sm text-slate-300">Live Connection</span>
            </div>
            <p className="text-xs text-slate-500 mt-1">Last Sync: {lastUpdate}</p>
          </div>
        </header>

        <AnimatePresence>
          {emergencyMode ? (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-red-900/50 border border-red-500/50 rounded-2xl p-8 backdrop-blur-md shadow-[0_0_50px_rgba(239,68,68,0.3)] relative overflow-hidden"
            >
              <div className="absolute top-0 left-0 w-full h-1 bg-red-500 animate-pulse"></div>
              
              <div className="flex flex-col md:flex-row gap-8">
                {/* Video Feed */}
                <div className="flex-1 space-y-4">
                  <div className="flex items-center justify-between">
                    <h2 className="text-2xl font-bold flex items-center gap-2">
                      <Video className="text-red-400" />
                      Live Emergency Feed
                    </h2>
                    <span className="px-3 py-1 bg-red-500 text-white text-xs font-bold rounded-full animate-pulse">
                      LIVE
                    </span>
                  </div>
                  
                  <div className="aspect-video bg-black rounded-xl overflow-hidden border-2 border-red-500/30 relative shadow-2xl">
                    <img 
                      src="http://127.0.0.1:5000/video_feed" 
                      alt="Remote Live Feed" 
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                        (e.target as HTMLImageElement).nextElementSibling?.classList.remove('hidden');
                      }}
                    />
                    <div className="hidden absolute inset-0 flex items-center justify-center text-slate-500">
                      Camera feed unavailable (User offline)
                    </div>
                    {/* Scanline overlay */}
                    <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_4px,3px_100%] opacity-50 z-10"></div>
                  </div>
                </div>

                {/* Status & Actions */}
                <div className="w-full md:w-80 space-y-6">
                  <div className="bg-black/40 rounded-xl p-6 border border-white/5 space-y-4">
                    <h3 className="text-lg font-semibold text-red-400 flex items-center gap-2">
                      <PhoneCall size={18} />
                      Alert Details
                    </h3>
                    <div className="space-y-3">
                      <div>
                        <p className="text-xs text-slate-400 uppercase tracking-wider">Trigger Reason</p>
                        <p className="font-medium text-white">Manual SOS Button</p>
                      </div>
                      <div>
                        <p className="text-xs text-slate-400 uppercase tracking-wider">AI Scene Analysis</p>
                        <p className="font-medium text-white text-sm line-clamp-3">{sceneStatus}</p>
                      </div>
                      <div>
                        <p className="text-xs text-slate-400 uppercase tracking-wider">Automated Call</p>
                        <p className="font-medium text-emerald-400 text-sm">Dispatched Successfully</p>
                      </div>
                    </div>
                  </div>

                  <button 
                    onClick={handleResolve}
                    className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-4 px-6 rounded-xl transition-all active:scale-95 flex items-center justify-center gap-2 shadow-lg shadow-emerald-900/20"
                  >
                    <CheckCircle size={20} />
                    Resolve Emergency
                  </button>
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="bg-white/5 border border-white/10 rounded-2xl p-12 text-center backdrop-blur-sm"
            >
              <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-emerald-500/10 text-emerald-400 mb-6">
                <CheckCircle size={48} />
              </div>
              <h2 className="text-2xl font-bold mb-2">All Systems Nominal</h2>
              <p className="text-slate-400 max-w-md mx-auto">
                The NAVI user is currently safe. If an emergency occurs, this dashboard will automatically switch to alert mode and display the live camera feed.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

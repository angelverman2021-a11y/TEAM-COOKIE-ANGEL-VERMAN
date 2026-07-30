document.addEventListener("DOMContentLoaded", () => {
    const btnConnect = document.getElementById("btn-connect");
    const connectionDot = document.getElementById("connection-dot");
    const connectionText = document.getElementById("connection-text");
    const videoStream = document.getElementById("video-stream");
    const emotionValue = document.getElementById("emotion-value");
    const btnSos = document.getElementById("btn-sos");
    
    let pollingInterval = null;

    const btnConnectCard = document.getElementById("btn-connect-card");
    const btnConnectMain = document.getElementById("btn-connect-main");

    // Connect to Glasses logic
    const connectToGlasses = async () => {
        console.log("CONNECT CLICKED");
        if(btnConnect) btnConnect.style.opacity = "0.5";
        if(btnConnectCard) btnConnectCard.style.opacity = "0.5";
        
        try {
            const response = await fetch('/api/connect', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                if(btnConnect) btnConnect.style.color = "#4ade80"; // primary color
                if(btnConnectCard) btnConnectCard.style.borderColor = "#4ade80";
                
                if(connectionDot) {
                    connectionDot.classList.remove("bg-gray-500");
                    connectionDot.classList.add("bg-primary", "pulse");
                }
                if(connectionText) {
                    connectionText.innerText = "Connected";
                    connectionText.classList.replace("text-gray-500", "text-primary");
                }
                
                if(btnConnectMain) btnConnectMain.style.display = "none";
                
                if(videoStream) {
                    videoStream.src = "/video_feed?t=" + Date.now();
                    videoStream.style.display = "block";
                }
                
                // Start polling telemetry status
                if(!pollingInterval) {
                    pollingInterval = setInterval(fetchStatus, 500);
                }
            } else {
                alert("Connection returned false: " + data.message);
            }
        } catch (err) {
            console.error(err);
            alert("Network connection failed! Is your Python server running? Error: " + err.message);
        }
    };

    if(btnConnect) btnConnect.addEventListener("click", connectToGlasses);
    if(btnConnectCard) btnConnectCard.addEventListener("click", connectToGlasses);
    if(btnConnectMain) btnConnectMain.addEventListener("click", connectToGlasses);
    
    // Fetch live telemetry from the Python Engine
    async function fetchStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if(emotionValue) {
                emotionValue.innerText = data.emotion.toUpperCase();
                
                // Color coding
                if (data.emotion.toLowerCase() === "angry") {
                    emotionValue.style.color = "#EF4444"; // danger
                } else if (data.emotion.toLowerCase() === "happy") {
                    emotionValue.style.color = "#4ade80"; // primary
                } else {
                    emotionValue.style.color = "#22D3EE"; // cyan
                }
            }
            
            // Populate Debug Diagnostics
            if(data.diagnostics) {
                const diag = data.diagnostics;
                const setHtml = (id, val) => {
                    let el = document.getElementById(id);
                    if(el) el.innerText = val;
                };
                setHtml("dbg-fps", (diag.cam_fps || 0).toFixed(1));
                setHtml("dbg-backend", diag.backend || "--");
                setHtml("dbg-resolution", diag.resolution ? diag.resolution.join("x") : "--");
                setHtml("dbg-aitime", (diag.ai_time || 0).toFixed(1));
                setHtml("dbg-detcount", diag.det_count || "0");
                setHtml("dbg-device", diag.device || "--");
                setHtml("dbg-camstatus", diag.cam_status || "--");
            }
            
            // Dynamic HUD Widgets based on Live AI Data
            const navMetrics = data.nav_metrics || {};
            let foundPerson = null;
            let foundStairs = null;
            let foundTrafficLight = null;
            
            for (const key in navMetrics) {
                const obj = navMetrics[key];
                const lbl = (obj.label || "").toLowerCase();
                if (lbl === "person") {
                    if (!foundPerson || obj.dist < foundPerson.dist) foundPerson = obj;
                } else if (lbl === "stairs") {
                    if (!foundStairs || obj.dist < foundStairs.dist) foundStairs = obj;
                } else if (lbl === "traffic light") {
                    if (!foundTrafficLight || obj.dist < foundTrafficLight.dist) foundTrafficLight = obj;
                }
            }
            
            const elPerson = document.getElementById("hud-person");
            const elPersonText = document.getElementById("hud-person-text");
            if (foundPerson && elPerson && elPersonText) {
                elPerson.classList.remove("hidden");
                elPersonText.innerText = `PERSON: ${(foundPerson.dist * 10).toFixed(1)}m`; // Pseudo-distance scaling
            } else if (elPerson) {
                elPerson.classList.add("hidden");
            }
            
            const elStairs = document.getElementById("hud-stairs");
            const elStairsText = document.getElementById("hud-stairs-text");
            if (foundStairs && elStairs && elStairsText) {
                elStairs.classList.remove("hidden");
                elStairsText.innerText = `Stairs Detected (${(foundStairs.dist * 10).toFixed(1)}m)`;
            } else if (elStairs) {
                elStairs.classList.add("hidden");
            }
            
            const elTraffic = document.getElementById("hud-traffic");
            const elTrafficText = document.getElementById("hud-traffic-text");
            if (foundTrafficLight && elTraffic && elTrafficText) {
                elTraffic.classList.remove("hidden");
                elTrafficText.innerText = `Traffic Light: ${(foundTrafficLight.dist * 10).toFixed(1)}m`;
            } else if (elTraffic) {
                elTraffic.classList.add("hidden");
            }
            
            // Voice Assistant Text
            const elVoice = document.getElementById("voice-assistant-text");
            if (elVoice) {
                let action = data.recommended_action || "Path clear";
                if (data.danger_level === "Critical") {
                    elVoice.innerHTML = `"<span class='text-danger font-bold'>${action.toUpperCase()}</span>"`;
                } else if (data.danger_level === "Warning") {
                    elVoice.innerHTML = `"<span class='text-accent-cyan font-bold'>${action}</span>"`;
                } else {
                    elVoice.innerHTML = `"${action}"`;
                }
            }
            
            // Scene Understanding Text
            if (data.scene_understanding) {
                const sc = data.scene_understanding;
                const setHtml = (id, val) => {
                    let el = document.getElementById(id);
                    if(el) el.innerText = val;
                };
                setHtml("scene-short", sc.scene || "--");
                setHtml("scene-objects", sc.objects ? sc.objects.join(", ") : "--");
                setHtml("scene-obstacles", sc.obstacles ? sc.obstacles.join(", ") : "--");
                setHtml("scene-dir", sc.walkable_direction || "--");
                setHtml("scene-summary", sc.summary || "Waiting for Florence-2...");
            }
            
        } catch (err) {
            console.error("Telemetry lost");
        }
    }
    
    // Manual SOS Button
    if(btnSos) {
        btnSos.addEventListener("click", async () => {
            const originalText = btnSos.innerHTML;
            btnSos.innerHTML = `<span class="material-symbols-outlined text-[32px]">warning</span><span class="text-[10px] font-bold">SENDING</span>`;
            
            try {
                const response = await fetch('/api/sos', { method: 'POST' });
                const data = await response.json();
                
                setTimeout(() => {
                    btnSos.innerHTML = originalText;
                }, 3000);
                
            } catch (err) {
                console.error(err);
                btnSos.innerHTML = `<span class="material-symbols-outlined text-[32px]">error</span><span class="text-[10px] font-bold">FAILED</span>`;
            }
        });
    }
});

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
                    videoStream.src = "/video_feed";
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

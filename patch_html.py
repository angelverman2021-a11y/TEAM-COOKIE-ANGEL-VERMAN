import os

with open("stitch_dashboard.html", "r", encoding="utf-8") as f:
    html = f.read()

# 1. Connection Indicator
html = html.replace(
    '<div class="w-2 h-2 rounded-full bg-primary pulse"></div>\n<span class="text-label-sm font-label-sm text-primary">Connected</span>',
    '<div id="connection-dot" class="w-2 h-2 rounded-full bg-gray-500"></div>\n<span id="connection-text" class="text-label-sm font-label-sm text-gray-500">Disconnected</span>'
)

# 2. Bluetooth Connect Button
html = html.replace(
    '<span class="material-symbols-outlined text-on-surface-variant cursor-pointer hover:text-primary">bluetooth_connected</span>',
    '<span id="btn-connect" class="material-symbols-outlined text-on-surface-variant cursor-pointer hover:text-primary" title="Connect Camera">bluetooth_connected</span>'
)

import re

# 3. Video Stream Replacement
# We use regex to find the img tag with HUD View
pattern = r'<img[^>]*HUD View[^>]*>'
new_img = '<img id="video-stream" class="w-full h-full object-cover opacity-100" style="display:none;" src="" />'
html = re.sub(pattern, new_img, html, count=1)




# 4. Telemetry Mood Text
html = html.replace(
    '<span class="text-[24px] font-bold text-primary">92%</span>\n<span class="text-[10px] text-on-surface-variant uppercase font-bold">Mood: Happy</span>',
    '<span class="text-[24px] font-bold text-primary">AI</span>\n<span class="text-[10px] text-on-surface-variant uppercase font-bold">Mood: <span id="emotion-value">Scanning</span></span>'
)

# 5. SOS Button
html = html.replace(
    '<button class="fixed bottom-margin-mobile',
    '<button id="btn-sos" class="fixed bottom-margin-mobile'
)

# 6. Inject JS
html = html.replace('</body>', '<script src="{{ url_for(\'static\', filename=\'script.js\') }}"></script>\n</body>')

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html)
    
print("Successfully patched Stitch HTML into templates/index.html")

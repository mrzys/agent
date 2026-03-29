"""
HTTP Service for viewing chat sessions
Run: python main.py
"""

import json
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

SESSIONS_DIR = Path(".sessions")


def get_all_sessions():
    """Get all session IDs from .sessions directory"""
    if not SESSIONS_DIR.exists():
        return []

    sessions = []
    for file_path in SESSIONS_DIR.glob("*.json"):
        session_id = file_path.stem
        # Get file modification time
        stat = file_path.stat()
        sessions.append(
            {
                "session_id": session_id,
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size": stat.st_size,
            }
        )

    # Sort by last modified time, newest first
    sessions.sort(key=lambda x: x["last_modified"], reverse=True)
    return sessions


def get_session_messages(session_id: str):
    """Get all messages for a specific session"""
    file_path = SESSIONS_DIR / f"{session_id}.json"

    if not file_path.exists():
        return []

    messages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    msg = json.loads(line)
                    messages.append(msg)
                except json.JSONDecodeError:
                    continue

    return messages


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat Sessions</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            height: 100vh;
            display: flex;
        }
        
        /* 左侧会话列表 */
        .sidebar {
            width: 300px;
            background: #fff;
            border-right: 1px solid #e0e0e0;
            overflow-y: auto;
            flex-shrink: 0;
        }
        
        .sidebar-header {
            padding: 20px;
            background: #2c3e50;
            color: white;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        .sidebar-header h1 {
            font-size: 18px;
            font-weight: 500;
        }
        
        .session-list {
            padding: 10px;
        }
        
        .session-item {
            padding: 15px;
            margin-bottom: 8px;
            background: #f8f9fa;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            border: 2px solid transparent;
        }
        
        .session-item:hover {
            background: #e9ecef;
            border-color: #3498db;
        }
        
        .session-item.active {
            background: #3498db;
            color: white;
        }
        
        .session-id {
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 5px;
            word-break: break-all;
        }
        
        .session-meta {
            font-size: 11px;
            opacity: 0.7;
        }
        
        /* 右侧对话区域 */
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #fff;
        }
        
        .chat-header {
            padding: 20px;
            background: #fff;
            border-bottom: 1px solid #e0e0e0;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        .chat-header h2 {
            font-size: 16px;
            color: #333;
            font-weight: 500;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #fafafa;
            display: flex;
            flex-direction: column;
        }
        
        /* 消息通用样式 */
        .message {
            margin-bottom: 20px;
            display: flex;
            flex-direction: column;
            max-width: 75%;
            min-width: 0;
        }
        
        /* User消息在左边 */
        .message.user {
            align-self: flex-start;
            margin-right: auto;
        }
        
        /* Assistant和Tool消息在右边 */
        .message.assistant,
        .message.tool {
            align-self: flex-end;
            margin-left: auto;
        }
        
        /* System消息居中 */
        .message.system {
            align-self: center;
            max-width: 85%;
        }
        
        .message-header {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 12px;
            color: #666;
        }
        
        .message-role {
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            margin-right: 10px;
            text-transform: uppercase;
            font-size: 11px;
        }
        
        .message-timestamp {
            font-size: 11px;
            color: #999;
            margin-right: 10px;
            font-family: 'Monaco', 'Menlo', monospace;
        }
        
        .role-system {
            background: #e74c3c;
            color: white;
        }
        
        .role-user {
            background: #3498db;
            color: white;
        }
        
        .role-assistant {
            background: #2ecc71;
            color: white;
        }
        
        .role-tool {
            background: #9b59b6;
            color: white;
        }
        
        .message-content {
            background: #fff;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            line-height: 1.6;
            word-wrap: break-word;
            overflow-wrap: break-word;
            min-width: 0;
        }
        
        .message-content pre {
            background: #f4f4f4;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            max-width: 100%;
        }
        
        .message-content code {
            background: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: monospace;
        }
        
        .message-content blockquote {
            border-left: 4px solid #3498db;
            margin: 0 0 16px 0;
            padding: 12px 16px;
            color: #555;
            background: #f8f9fa;
            border-radius: 0 8px 8px 0;
            font-style: italic;
        }
        
        .message-content blockquote p:last-child {
            margin-bottom: 0;
        }
        
        .message-content a {
            color: #3498db;
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: all 0.2s;
        }
        
        .message-content a:hover {
            color: #2980b9;
            border-bottom-color: #2980b9;
        }
        
        .message-content a:focus {
            outline: 2px solid #3498db;
            outline-offset: 2px;
        }
        
        .message-content ul, .message-content ol {
            margin-left: 20px;
            margin-bottom: 12px;
        }
        
        .message-content li {
            margin-bottom: 4px;
        }
        
        .message-content p {
            margin-bottom: 12px;
        }
        
        .message-content p:last-child {
            margin-bottom: 0;
        }
        
        .message-content h1, .message-content h2, .message-content h3,
        .message-content h4, .message-content h5, .message-content h6 {
            margin-top: 16px;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .message-content h1 { font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: 8px; }
        .message-content h2 { font-size: 1.3em; }
        .message-content h3 { font-size: 1.1em; }
        
        .message-content table {
            border-collapse: collapse;
            width: 100%;
            margin: 12px 0;
        }
        
        .message-content th, .message-content td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }
        
        .message-content th {
            background: #f5f5f5;
            font-weight: 600;
        }
        
        .message-content tr:nth-child(even) {
            background: #fafafa;
        }
        
        .footnotes {
            margin-top: 24px;
            padding-top: 16px;
            border-top: 1px solid #e0e0e0;
            font-size: 0.9em;
        }
        
        .footnotes-sep {
            display: none;
        }
        
        .footnotes ol {
            margin-left: 16px;
            padding-left: 0;
        }
        
        .footnotes li {
            margin-bottom: 8px;
            color: #666;
        }
        
        .footnote-ref {
            font-size: 0.75em;
            vertical-align: super;
            color: #3498db;
            text-decoration: none;
            font-weight: 600;
            padding: 0 2px;
        }
        
        .footnote-ref:hover {
            color: #2980b9;
        }
        
        .footnote-backref {
            color: #3498db;
            text-decoration: none;
            margin-left: 4px;
        }
        
        .footnote-backref:hover {
            color: #2980b9;
        }
        
        .message.user .message-content {
            background: #e3f2fd;
        }
        
        .message.assistant .message-content {
            background: #e8f5e9;
        }
        
        .message.system .message-content {
            background: #ffebee;
        }
        
        .message.tool .message-content {
            background: #f3e5f5;
        }
        
        /* Tool结果Card样式 - 可折叠 */
        .tool-card {
            margin-top: 10px;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e0e0e0;
            background: #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .tool-card-toggle {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 13px;
            font-weight: 600;
        }
        
        .tool-card-toggle:hover {
            background: linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%);
        }
        
        .tool-card-toggle-left {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .tool-card-toggle-icon {
            transition: transform 0.2s;
            font-size: 10px;
        }
        
        .tool-card-toggle.expanded .tool-card-toggle-icon {
            transform: rotate(90deg);
        }
        
        .tool-card-content {
            display: none;
            background: #fafafa;
            padding: 16px;
            font-size: 13px;
            line-height: 1.5;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #333;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }
        
        .tool-card-content.expanded {
            display: block;
        }
        
        /* 复制按钮样式 */
        .tool-card-copy-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            background: #667eea;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            transition: all 0.2s;
            opacity: 0.8;
        }
        
        .tool-card-copy-btn:hover {
            opacity: 1;
            background: #5a6fd6;
        }
        
        .tool-card-copy-btn.copied {
            background: #2ecc71;
        }
        
        .tool-card-content-wrapper {
            position: relative;
        }
        
        .empty-state {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #999;
            font-size: 16px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">
            <h1>💬 Chat Sessions</h1>
        </div>
        <div class="session-list" id="sessionList">
            <div class="loading">Loading sessions...</div>
        </div>
    </div>
    
    <div class="chat-container">
        <div class="chat-header">
            <h2 id="chatTitle">Select a session to view</h2>
        </div>
        <div class="chat-messages" id="chatMessages">
            <div class="empty-state">Select a session from the left to view conversations</div>
        </div>
    </div>
    
    <script>
        let currentSessionId = null;
        
        marked.setOptions({
            breaks: true,
            gfm: true
        });
        
        function processFootnotes(text) {
            const footnotes = {};
            const usedFootnotes = [];
            let counter = 1;
            
            // Step 1: Extract and remove footnote definitions [^id]: content
            text = text.replace(/\\[\\^([a-zA-Z0-9_-]+)\\]:\\s*(.+?)(?=\\n\\n|\\n\\[\\^|\\n*$)/gm, (match, id, content) => {
                footnotes[id] = content.trim();
                return '';
            });
            
            // Step 2: Replace footnote references [^id] with numbered links
            text = text.replace(/\\[\\^([a-zA-Z0-9_-]+)\\]/g, (match, id) => {
                if (footnotes[id]) {
                    const num = counter++;
                    usedFootnotes.push({ id, content: footnotes[id], num });
                    return `<sup class="footnote-ref"><a href="#fn-${id}" id="fnref-${id}">[${num}]</a></sup>`;
                }
                return match;
            });
            
            // Step 3: Add footnotes section
            if (usedFootnotes.length > 0) {
                text += '<div class="footnotes"><hr><ol>';
                usedFootnotes.forEach(fn => {
                    text += `<li id="fn-${fn.id}">${fn.content} <a href="#fnref-${fn.id}" class="footnote-backref">↩</a></li>`;
                });
                text += '</ol></div>';
            }
            
            return text;
        }
        
        // Load sessions list
        async function loadSessions() {
            try {
                const response = await fetch('/api/sessions');
                const sessions = await response.json();
                
                const sessionList = document.getElementById('sessionList');
                sessionList.innerHTML = '';
                
                sessions.forEach(session => {
                    const div = document.createElement('div');
                    div.className = 'session-item';
                    div.dataset.sessionId = session.session_id;
                    div.onclick = () => selectSession(session.session_id, div);
                    div.innerHTML = `
                        <div class="session-id">${session.session_id}</div>
                        <div class="session-meta">
                            ${new Date(session.last_modified).toLocaleString()} • ${formatBytes(session.size)}
                        </div>
                    `;
                    sessionList.appendChild(div);
                });
            } catch (error) {
                console.error('Error loading sessions:', error);
                document.getElementById('sessionList').innerHTML = '<div style="padding: 20px; color: red;">Failed to load sessions</div>';
            }
        }
        
        // Select a session and load its messages
        async function selectSession(sessionId, element) {
            currentSessionId = sessionId;
            
            // Update UI
            document.querySelectorAll('.session-item').forEach(item => {
                item.classList.remove('active');
            });
            element.classList.add('active');
            
            document.getElementById('chatTitle').textContent = `Session: ${sessionId}`;
            document.getElementById('chatMessages').innerHTML = '<div class="loading">Loading messages...</div>';
            
            try {
                const response = await fetch(`/api/sessions/${sessionId}/messages`);
                const messages = await response.json();
                
                const container = document.getElementById('chatMessages');
                container.innerHTML = '';
                
                messages.forEach((msg, index) => {
                    const div = document.createElement('div');
                    div.className = `message ${msg.role}`;
                    
                    // 处理Tool Calls（默认折叠）
                    let toolCallsHtml = '';
                    if (msg.tool_calls && msg.tool_calls.length > 0) {
                        const toolCallsId = `tool-calls-${index}`;
                        const toolCallsContent = msg.tool_calls.map(tc => `
                            <div class="tool-call">
                                <span class="tool-call-name">${tc.function.name}</span>
                                <pre>${tc.function.arguments}</pre>
                            </div>
                        `).join('');
                        
                        toolCallsHtml = `
                            <div class="tool-calls">
                                <div class="tool-calls-toggle" onclick="toggleToolCalls('${toolCallsId}', this)">
                                    <span>Tool Calls (${msg.tool_calls.length})</span>
                                </div>
                                <div class="tool-calls-content" id="${toolCallsId}">
                                    ${toolCallsContent}
                                </div>
                            </div>
                        `;
                    }
                    
                    // Tool消息作为Card，默认折叠
                    let contentHtml = '';
                    if (msg.role === 'tool') {
                        const toolCardId = `tool-card-${index}`;
                        const toolContentId = `tool-content-${index}`;
                        const toolContent = escapeHtml(msg.content || '');
                        // Tool消息作为折叠的card展示
                        div.innerHTML = `
                            <div class="tool-card">
                                <div class="tool-card-toggle" onclick="toggleToolCard('${toolCardId}', this)">
                                    <div class="tool-card-toggle-left">
                                        <span class="tool-card-toggle-icon">▶</span>
                                        <span>🔧 Tool Result: ${msg.name || 'Unknown Tool'}</span>
                                    </div>
                                    <span style="font-size: 11px; opacity: 0.8;">${formatTimestamp(msg.timestamp)} ${msg.tool_call_id || ''}</span>
                                </div>
                                <div class="tool-card-content" id="${toolCardId}">
                                    <div class="tool-card-content-wrapper">
                                        <button class="tool-card-copy-btn" onclick="copyToolContent('${toolContentId}', this)" title="复制内容">
                                            📋 复制
                                        </button>
                                        <div id="${toolContentId}">${toolContent}</div>
                                    </div>
                                </div>
                            </div>
                        `;
                    } else {
                        if (msg.role === 'assistant' || msg.role === 'system') {
                            const processedContent = processFootnotes(msg.content || '');
                            contentHtml = marked.parse(processedContent);
                        } else {
                            contentHtml = escapeHtml(msg.content || '');
                        }
                        
                        div.innerHTML = `
                            <div class="message-header">
                                <span class="message-role role-${msg.role}">${msg.role}</span>
                                <span class="message-timestamp">${formatTimestamp(msg.timestamp)}</span>
                                ${msg.name ? `<span>Tool: ${msg.name}</span>` : ''}
                                ${msg.tool_call_id ? `<span>Call ID: ${msg.tool_call_id}</span>` : ''}
                            </div>
                            <div class="message-content">${contentHtml}</div>
                            ${toolCallsHtml}
                        `;
                    }
                    
                    container.appendChild(div);
                });
                
                // Scroll to bottom
                container.scrollTop = container.scrollHeight;
            } catch (error) {
                console.error('Error loading messages:', error);
                document.getElementById('chatMessages').innerHTML = '<div style="padding: 20px; color: red;">Failed to load messages</div>';
            }
        }
        
        // Toggle tool calls visibility
        function toggleToolCalls(contentId, toggleElement) {
            const content = document.getElementById(contentId);
            const isExpanded = content.classList.contains('expanded');
            
            if (isExpanded) {
                content.classList.remove('expanded');
                toggleElement.classList.remove('expanded');
            } else {
                content.classList.add('expanded');
                toggleElement.classList.add('expanded');
            }
        }
        
        // Toggle tool card visibility
        function toggleToolCard(cardId, toggleElement) {
            const content = document.getElementById(cardId);
            const isExpanded = content.classList.contains('expanded');
            
            if (isExpanded) {
                content.classList.remove('expanded');
                toggleElement.classList.remove('expanded');
            } else {
                content.classList.add('expanded');
                toggleElement.classList.add('expanded');
            }
        }
        
        // Copy tool content to clipboard
        async function copyToolContent(contentId, btnElement) {
            const contentDiv = document.getElementById(contentId);
            const text = contentDiv.textContent;
            const originalText = btnElement.innerHTML;
            
            try {
                await navigator.clipboard.writeText(text);
                
                // Visual feedback
                btnElement.innerHTML = '✓ 已复制';
                btnElement.classList.add('copied');
                
                // Reset after 2 seconds
                setTimeout(() => {
                    btnElement.innerHTML = originalText;
                    btnElement.classList.remove('copied');
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
                btnElement.innerHTML = '✗ 失败';
                setTimeout(() => {
                    btnElement.innerHTML = originalText;
                }, 2000);
            }
        }
        
        // Format bytes to human readable
        function formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
        
        // Escape HTML
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Format timestamp to date string
        function formatTimestamp(timestamp) {
            if (!timestamp) return '';
            // Handle both seconds and milliseconds
            const ts = timestamp > 10000000000 ? timestamp : timestamp * 1000;
            const date = new Date(ts);
            return date.toLocaleString('zh-CN', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
        }
        
        // Auto refresh every 10 seconds
        setInterval(() => {
            if (!currentSessionId) {
                loadSessions();
            }
        }, 10000);
        
        // Initial load
        loadSessions();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Serve the main HTML page"""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/sessions")
def api_sessions():
    """API: Get all sessions"""
    sessions = get_all_sessions()
    return jsonify(sessions)


@app.route("/api/sessions/<session_id>/messages")
def api_session_messages(session_id: str):
    """API: Get messages for a specific session"""
    messages = get_session_messages(session_id)
    return jsonify(messages)


if __name__ == "__main__":
    print("=" * 60)
    print("Chat Session Viewer")
    print("=" * 60)
    print(f"Sessions directory: {SESSIONS_DIR.absolute()}")
    print("Open http://127.0.0.1:5001 in your browser")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5001)

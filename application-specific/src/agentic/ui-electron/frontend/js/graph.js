/**
 * Graph Module
 * Handles graph visualization
 */

class GraphManager {
    constructor() {
        this.canvas = document.getElementById('graph-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.scale = 0.75; // Zoomed out by default
        this.offsetX = 0;
        this.offsetY = 0;
        
        this.nodes = new Map();
        this.edges = [];
        this.activeNode = null;
        this.visitedNodes = new Set();
        
        // Drag state
        this.isDragging = false;
        this.dragStartX = 0;
        this.dragStartY = 0;
        this.lastOffsetX = 0;
        this.lastOffsetY = 0;
        
        this.initializeCanvas();
        this.createDefaultGraph();
        this.centerGraph(); // Center the graph initially
        this.initializeEventListeners();
        this.initializeDragPan();
    }

    initializeCanvas() {
        // Set canvas size
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        
        // Handle resize
        window.addEventListener('resize', () => {
            this.canvas.width = container.clientWidth;
            this.canvas.height = container.clientHeight;
            this.draw();
        });
    }

    centerGraph() {
        // Approximate center of the graph content
        const graphCenterX = 400;
        const graphCenterY = 350;
        
        // Center in the canvas
        this.offsetX = (this.canvas.width / 2) - (graphCenterX * this.scale);
        this.offsetY = (this.canvas.height / 2) - (graphCenterY * this.scale);
        this.draw();
    }

    initializeEventListeners() {
        document.getElementById('btn-zoom-in').addEventListener('click', () => this.zoomIn());
        document.getElementById('btn-zoom-out').addEventListener('click', () => this.zoomOut());
        document.getElementById('btn-reset-graph').addEventListener('click', () => this.reset());
        
        if (window.electronAPI) {
            window.electronAPI.onResetGraph(() => this.reset());
        }
    }

    initializeDragPan() {
        // Mouse down - start dragging
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.dragStartX = e.clientX;
            this.dragStartY = e.clientY;
            this.lastOffsetX = this.offsetX;
            this.lastOffsetY = this.offsetY;
            this.canvas.style.cursor = 'grabbing';
        });

        // Mouse move - drag
        this.canvas.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const dx = e.clientX - this.dragStartX;
                const dy = e.clientY - this.dragStartY;
                this.offsetX = this.lastOffsetX + dx;
                this.offsetY = this.lastOffsetY + dy;
                this.draw();
            }
        });

        // Mouse up - stop dragging
        this.canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.canvas.style.cursor = 'grab';
        });

        // Mouse leave - stop dragging
        this.canvas.addEventListener('mouseleave', () => {
            this.isDragging = false;
            this.canvas.style.cursor = 'grab';
        });

        // Mouse wheel - zoom
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            this.scale *= delta;
            this.scale = Math.max(0.1, Math.min(3, this.scale)); // Limit zoom
            this.draw();
        });

        // Set initial cursor
        this.canvas.style.cursor = 'grab';
    }

    createDefaultGraph() {
        // Define node positions - Complete workflow with Direct RTL synthesis option
        const positions = {
            'START': {x: 400, y: 50, label: 'START'},
            'master_triage_router': {x: 400, y: 150, label: 'Router'},
            
            // Left path - Traditional workflow
            'create_project_node': {x: 150, y: 250, label: 'Project'},
            
            'design_loop_router': {x: 400, y: 350, label: 'Design Loop'},
            'design_arch_node': {x: 250, y: 470, label: 'Architect'},
            'config_params_node': {x: 550, y: 470, label: 'Config'},
            'ask_design_confirmation_node': {x: 400, y: 540, label: 'Confirm?'},
            'set_design_confirmed_node': {x: 400, y: 620, label: 'Confirmed'},
            
            'generate_code_node': {x: 650, y: 250, label: 'Code Gen'},
            'train_node': {x: 650, y: 620, label: 'Train'},
            
            'ask_new_design_node': {x: 650, y: 730, label: 'Next?'},
            'post_training_router': {x: 650, y: 840, label: 'Post-Train'},
            'test_image_node': {x: 450, y: 960, label: 'Test Image'},
            
            // Right path - Direct RTL synthesis (with sub-steps)
            'direct_rtl_upload_node': {x: 850, y: 250, label: 'Upload .pt'},
            'rtl_synthesis_node': {x: 850, y: 380, label: 'RTL Config'},
            'configure_hls_node': {x: 850, y: 510, label: 'HLS C++'},
            'verify_hls_node': {x: 850, y: 640, label: 'Verify C++'},
            'synthesize_rtl_node': {x: 850, y: 770, label: 'Synthesize'},
            
            'END': {x: 650, y: 900, label: 'END'}
        };
        
        // Create nodes
        for (const [id, data] of Object.entries(positions)) {
            this.nodes.set(id, {
                id: id,
                x: data.x,
                y: data.y,
                label: data.label,
                radius: 38 // Slightly larger for better visibility
            });
        }
        
        // Define edges - Complete workflow with Direct RTL synthesis path
        this.edges = [
            // Start -> Router
            ['START', 'master_triage_router'],
            
            // Router -> Phase Nodes (traditional workflow)
            ['master_triage_router', 'create_project_node'],
            ['master_triage_router', 'design_loop_router'],
            ['master_triage_router', 'generate_code_node'],
            ['master_triage_router', 'train_node'],
            ['master_triage_router', 'post_training_router'],
            
            // Router -> Direct RTL path
            ['master_triage_router', 'direct_rtl_upload_node'],
            ['master_triage_router', 'rtl_synthesis_node'],
            ['master_triage_router', 'END'],
            
            // Project Loop
            ['create_project_node', 'master_triage_router'],
            
            // Design Loop
            ['design_loop_router', 'design_arch_node'],
            ['design_loop_router', 'config_params_node'],
            ['design_loop_router', 'ask_design_confirmation_node'],
            ['design_loop_router', 'set_design_confirmed_node'],
            
            // Design Backflows
            ['design_arch_node', 'design_loop_router'],
            ['config_params_node', 'design_loop_router'],
            
            // Confirmation Flow
            ['ask_design_confirmation_node', 'END'], // Waits for user
            ['set_design_confirmed_node', 'master_triage_router'],
            
            // Code Gen & Train
            ['generate_code_node', 'train_node'],
            ['train_node', 'ask_new_design_node'],
            
            // Post-Training Options
            ['ask_new_design_node', 'END'], // Waits for user
            ['post_training_router', 'test_image_node'],
            ['post_training_router', 'rtl_synthesis_node'],
            ['post_training_router', 'design_loop_router'], // New design
            ['post_training_router', 'END'],
            
            // Direct RTL path - Upload -> Config -> Steps
            ['direct_rtl_upload_node', 'rtl_synthesis_node'],
            ['direct_rtl_upload_node', 'master_triage_router'],
            
            // RTL Synthesis Pipeline (sequential steps)
            ['rtl_synthesis_node', 'configure_hls_node'],
            ['configure_hls_node', 'verify_hls_node'],
            ['verify_hls_node', 'synthesize_rtl_node'],
            ['synthesize_rtl_node', 'END'],
            
            // Post-action loops
            ['test_image_node', 'END']
        ];
        
        this.draw();
    }

    draw() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        ctx.save();
        ctx.translate(this.offsetX, this.offsetY);
        ctx.scale(this.scale, this.scale);
        
        // Get theme colors
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        const colors = {
            bg: isDark ? '#1a1a1a' : '#ffffff',
            edge: isDark ? '#4a4a4a' : '#d1d1d6',
            edgeActive: isDark ? '#00b8b8' : '#009999',
            edgeInactive: isDark ? '#3a3a3a' : '#e0e0e0',
            nodeInactive: isDark ? '#2a2a2a' : '#ffffff',
            nodeActive: isDark ? '#00b8b8' : '#009999',
            nodeVisited: isDark ? '#3a3a3a' : '#f0f0f0',
            text: isDark ? '#eeeeee' : '#1a1a1a',
            textSecondary: isDark ? '#b8b8b8' : '#54565a',
            border: isDark ? '#4a4a4a' : '#d1d1d6',
            borderActive: isDark ? '#00d4d4' : '#007a7a'
        };
        
        // Draw edges
        ctx.strokeStyle = colors.edge;
        ctx.lineWidth = 2;
        
        for (const [fromId, toId] of this.edges) {
            const from = this.nodes.get(fromId);
            const to = this.nodes.get(toId);
            
            if (from && to) {
                this.drawEdge(ctx, from, to, colors);
            }
        }
        
        // Draw nodes
        for (const node of this.nodes.values()) {
            this.drawNode(ctx, node, colors);
        }
        
        ctx.restore();
    }

    drawNode(ctx, node, colors) {
        const isActive = this.activeNode === node.id;
        const isVisited = this.visitedNodes.has(node.id);
        
        // Advanced visualization: Glow effect
        if (isActive || isVisited) {
            ctx.shadowColor = isActive ? colors.nodeActive : 'rgba(0, 0, 0, 0.2)';
            ctx.shadowBlur = isActive ? 20 : 10;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 4;
        }
        
        // Node Background (Gradient)
        const gradient = ctx.createRadialGradient(node.x, node.y, node.radius * 0.2, node.x, node.y, node.radius);
        
        if (isActive) {
            gradient.addColorStop(0, colors.nodeActive);
            gradient.addColorStop(1, this.adjustColor(colors.nodeActive, -20)); // Darker edge
            ctx.fillStyle = gradient;
            ctx.strokeStyle = '#ffffff'; // White border for contrast
            ctx.lineWidth = 3;
        } else if (isVisited) {
            gradient.addColorStop(0, colors.nodeVisited);
            gradient.addColorStop(1, this.adjustColor(colors.nodeVisited, -10));
            ctx.fillStyle = gradient;
            ctx.strokeStyle = colors.edgeActive;
            ctx.lineWidth = 2;
        } else {
            gradient.addColorStop(0, colors.nodeInactive);
            gradient.addColorStop(1, this.adjustColor(colors.nodeInactive, -10));
            ctx.fillStyle = gradient;
            ctx.strokeStyle = colors.border;
            ctx.lineWidth = 1.5;
        }
        
        // Draw Main Circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        
        // Active Node Pulse Ring (Visual Effect)
        if (isActive) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius + 6, 0, 2 * Math.PI);
            ctx.strokeStyle = colors.nodeActive;
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.5;
            ctx.stroke();
            ctx.globalAlpha = 1;
        }
        
        // Reset shadow
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
        
        // Label Background (for readability)
        const labelWidth = ctx.measureText(node.label).width + 12;
        ctx.fillStyle = isActive ? 'rgba(0,0,0,0.2)' : 'transparent';
        
        // Label
        ctx.fillStyle = isActive ? '#ffffff' : colors.text;
        ctx.font = isActive ? 'bold 14px "Siemens Sans", "Segoe UI", sans-serif' : '500 13px "Siemens Sans", "Segoe UI", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.label, node.x, node.y);
    }
    
    // Helper to darken/lighten hex color
    adjustColor(color, amount) {
        if (!color.startsWith('#')) return color;
        
        const num = parseInt(color.replace('#', ''), 16);
        let r = (num >> 16) + amount;
        let b = ((num >> 8) & 0x00FF) + amount;
        let g = (num & 0x0000FF) + amount;
        
        r = Math.max(Math.min(255, r), 0);
        b = Math.max(Math.min(255, b), 0);
        g = Math.max(Math.min(255, g), 0);
        
        return '#' + (g | (b << 8) | (r << 16)).toString(16).padStart(6, '0');
    }

    drawEdge(ctx, from, to, colors) {
        const isActiveEdge = this.visitedNodes.has(from.id) && this.visitedNodes.has(to.id);
        const isSourceActive = this.activeNode === from.id;
        const isSourceVisited = this.visitedNodes.has(from.id);
        
        // Calculate angle and arrow position
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx);
        const arrowSize = 12;
        
        // Determine edge style
        const lineWidth = isActiveEdge ? 3.5 : (isSourceVisited ? 2.5 : 2);
        const strokeStyle = isActiveEdge ? colors.edgeActive : (isSourceVisited ? colors.edgeActive : colors.edge);
        const alpha = isActiveEdge ? 1 : (isSourceVisited ? 0.5 : 0.4);
        
        // Draw shadow for active edges
        if (isActiveEdge || isSourceActive) {
            ctx.shadowColor = isActiveEdge ? colors.edgeActive : 'rgba(0, 184, 184, 0.3)';
            ctx.shadowBlur = 8;
        }
        
        // Use curved lines for better aesthetics
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        
        // Calculate control points for cubic Bezier curve
        // Use perpendicular offset for cleaner curved edges
        const curvature = 0.15; // Adjust this for more/less curve
        const midX = (from.x + to.x) / 2;
        const midY = (from.y + to.y) / 2;
        
        // Perpendicular direction for curve offset
        const perpAngle = angle + Math.PI / 2;
        const offset = distance * curvature;
        const cp1x = midX + Math.cos(perpAngle) * offset * 0.5;
        const cp1y = midY + Math.sin(perpAngle) * offset * 0.5;
        
        // Use quadratic curve for smoother edges
        ctx.quadraticCurveTo(cp1x, cp1y, to.x, to.y);
        
        ctx.strokeStyle = strokeStyle;
        ctx.lineWidth = lineWidth;
        ctx.globalAlpha = alpha;
        ctx.stroke();
        ctx.globalAlpha = 1;
        
        // Reset shadow
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
        
        // Draw arrowhead at the end (filled triangle)
        // Recalculate angle at the endpoint for curved edge
        const arrowBaseX = to.x - Math.cos(angle) * (to.radius || 35);
        const arrowBaseY = to.y - Math.sin(angle) * (to.radius || 35);
        
        ctx.beginPath();
        ctx.moveTo(arrowBaseX, arrowBaseY);
        ctx.lineTo(
            arrowBaseX - arrowSize * Math.cos(angle - Math.PI / 6),
            arrowBaseY - arrowSize * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
            arrowBaseX - arrowSize * Math.cos(angle + Math.PI / 6),
            arrowBaseY - arrowSize * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        
        ctx.fillStyle = strokeStyle;
        ctx.globalAlpha = alpha;
        ctx.fill();
        ctx.globalAlpha = 1;
    }

    setActiveNode(nodeId) {
        this.activeNode = nodeId;
        if (nodeId) {
            this.visitedNodes.add(nodeId);
        }
        this.draw();
    }

    reset() {
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.activeNode = null;
        this.visitedNodes.clear();
        this.draw();
    }

    zoomIn() {
        this.scale *= 1.2;
        this.draw();
    }

    zoomOut() {
        this.scale /= 1.2;
        this.draw();
    }
}

// Initialize graph manager
const graphManager = new GraphManager();
